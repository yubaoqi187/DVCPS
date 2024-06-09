import argparse
import logging
import os
import random 
import sys 

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime


from dataloaders.dataset import (BaseDataSets,DVCPS_BaseDataSets,
                                 TwoStreamBatchSampler)
from networks.net_factory import net_factory
from utils import losses
from val_2D import test_single_volume_dvcps

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='data/Task05_Prostate', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='Task05_Prostate/dvcps', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_dvcps', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=42, help='random seed')
parser.add_argument('--num_classes', type=int,  default=3,      # 3 for Prostate  4 for ACDC and miniACDC
                    help='output channel of network')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=3,
                    help='labeled data')
# costs
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')

parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, 
                    default=200.0, help='consistency_rampup')
parser.add_argument('--conf_thresh', type=float,
                    default=0.95, help='conf_thresh')
 
args = parser.parse_args()


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"1": 32, "3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "Task05_Prostate" in dataset:
        ref_dict = {"1": 15, "2": 35, "4":70 , 
                    "8": 150, "10" : 181, "21": 564}
    elif "miniAC" in dataset:
        ref_dict = {"1": 32, "2": 48, "4": 84,
                    "21": 396}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d_%H:%M:%S'
    return datetime.today().strftime(fmt)

def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=1,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model
    model = create_model()
    ema_model = create_model(ema=True)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train =  DVCPS_BaseDataSets(base_dir=args.root_path, split="train",num=None,size=args.patch_size)
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)

    print("Total silices is: {}, labeled slices is: {}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))

    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,num_workers=1)

    model.train()
    ema_model.train()
    optimizer = optim.SGD(model.parameters(), lr=base_lr,momentum=0.9, weight_decay=0.0001)
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = losses.DiceLoss_ignore(n_classes=num_classes)
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    cur_time = time_str() 
            
    num_lb = num_ulb = args.labeled_bs
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sample in enumerate(trainloader):
            img_w = sample["image_w"].cuda()
            lab_w = sample["label_w"].cuda()
            img_w_m = sample["image_w_m"].cuda()
            img_s = sample["image_s1"].cuda()
            img_s_m = sample["image_s1_m"].cuda()
            cutmix_box1 = sample["cutmix_box1"].cuda() 
            img_s[cutmix_box1.unsqueeze(1) == 1] = img_s_m[cutmix_box1.unsqueeze(1) == 1]
            
            with torch.no_grad():
                pseudo_1, pseudo_2, _ = ema_model(torch.cat((img_w[num_ulb:],img_w_m[num_ulb:])))
                pseudo_w1, pseudo_w_m1 = pseudo_1.detach().softmax(dim=1).split([num_lb, num_ulb])
                pseudo_w2, pseudo_w_m2 = pseudo_2.detach().softmax(dim=1).split([num_lb, num_ulb])

            pseudo1_s,pseudo2_s = pseudo_w1.clone(),pseudo_w2.clone()
            pseudo1_s[cutmix_box1[num_ulb:].unsqueeze(1).repeat(1,num_classes,1,1) == 1] \
                = pseudo_w_m1[cutmix_box1[num_ulb:].unsqueeze(1).repeat(1,num_classes,1,1) == 1]
            pseudo2_s[cutmix_box1[num_ulb:].unsqueeze(1).repeat(1,num_classes,1,1) == 1] \
                = pseudo_w_m2[cutmix_box1[num_ulb:].unsqueeze(1).repeat(1,num_classes,1,1) == 1]
            
            pred_w1,pred_w2,deepout1 = model(img_w) 
            pred_s1,pred_s2,deepout_s1 = model(img_s[num_ulb:]) 

            los1 = dice_loss(deepout1[4].softmax(dim=1)[:num_lb], lab_w[:num_lb].unsqueeze(1).float())
            los2 = dice_loss(deepout1[3].softmax(dim=1)[:num_lb], lab_w[:num_lb].unsqueeze(1).float())
            los3 = dice_loss(deepout1[2].softmax(dim=1)[:num_lb], lab_w[:num_lb].unsqueeze(1).float())
            los4 = dice_loss(deepout1[1].softmax(dim=1)[:num_lb], lab_w[:num_lb].unsqueeze(1).float())
            los5 = dice_loss(deepout1[0].softmax(dim=1)[:num_lb], lab_w[:num_lb].unsqueeze(1).float())
            loss_ds = (0.8*los1 + 0.5*los2 + 0.4*los3 + 0.2*los4 + 0.1*los5)/2
 
            loss_sup1 = dice_loss(pred_w1.softmax(dim=1)[:num_lb], lab_w[:num_lb].unsqueeze(1).float())
            loss_sup2 = ce_loss(pred_w2[:num_lb], lab_w[:num_lb].long())
            loss_sup = (loss_sup1 + loss_sup2)/2

            loss_cons1 = losses.loss_cot(pred_w1[num_ulb:],pred_w2[num_ulb:],bs*n*x*y/2)
            loss_cons2 = losses.loss_cot(pred_s1,pred_s2,bs*n*x*y/2) 
            loss_con = (loss_cons1 + loss_cons2)/2

            loss_ua_s1 = dice_loss(pred_s1[:num_lb].softmax(dim=1), pseudo2_s.max(dim=1)[1].unsqueeze(1).float(),
                                    ignore=(pseudo2_s.max(dim=1)[0] < args.conf_thresh).float())
            loss_ub_s1 = dice_loss(pred_s2[:num_lb].softmax(dim=1), pseudo1_s.max(dim=1)[1].unsqueeze(1).float(),
                                    ignore=(pseudo1_s.max(dim=1)[0] < args.conf_thresh).float())
            loss_ua_w = dice_loss(pred_w1[num_ulb:].softmax(dim=1), pseudo_w2.max(dim=1)[1].unsqueeze(1).float(),
                                    ignore=(pseudo_w2.max(dim=1)[0] < args.conf_thresh).float())
            loss_ub_w = dice_loss(pred_w2[num_ulb:].softmax(dim=1), pseudo_w1.max(dim=1)[1].unsqueeze(1).float(),
                                    ignore=(pseudo_w1.max(dim=1)[0] < args.conf_thresh).float())
            loss_ps = (loss_ua_s1 + loss_ub_s1 + loss_ua_w + loss_ub_w)/6

            loss = loss_sup + loss_con + loss_ds + loss_ps
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)
 
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ds', loss_ds, iter_num) 
            writer.add_scalar('info/loss_sup', loss_sup, iter_num)
            writer.add_scalar('info/loss_ds', loss_ds, iter_num) 
            writer.add_scalar('info/loss_con', loss_con, iter_num) 
            writer.add_scalar('info/loss_ps', loss_ps, iter_num)
            writer.add_scalar('info/loss_cons1', loss_cons1, iter_num)  
            writer.add_scalar('info/loss_cons2', loss_cons2, iter_num)  
            writer.add_scalar('info/loss_sup1', loss_sup1, iter_num)
            writer.add_scalar('info/loss_sup2', loss_sup2, iter_num)
            writer.add_scalar('info/los1', los1, iter_num)
            writer.add_scalar('info/los2', los2, iter_num)
            writer.add_scalar('info/los3', los3, iter_num)
            writer.add_scalar('info/los4', los4, iter_num)
            writer.add_scalar('info/los5', los5, iter_num)
            writer.add_scalar('info/loss_ua_s1', loss_ua_s1, iter_num) 
            writer.add_scalar('info/loss_ub_s1', loss_ub_s1, iter_num) 
            writer.add_scalar('info/loss_ua_w', loss_ua_w, iter_num)
            writer.add_scalar('info/loss_ub_w', loss_ub_w, iter_num) 
            logging.info(
                'iteration %d: loss:%f loss_sup1:%f loss_sup2:%f loss_ds:%f loss_con:%f' %
                (iter_num, loss.item() ,loss_sup1.item(),loss_sup2.item(),loss_ds.item(),loss_con.item()))

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_dvcps(sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1),metric_list[class_i, 1], iter_num)
                performance = np.mean(metric_list, axis=0)[0]
                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)
                    logging.info('iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "******/{}_{}_labeled".format(
        args.exp, args.labeled_num)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path) 

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)

