#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import os
import gc
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from data import ModelNet40
from vtctestregistrationdata import VTCTestRegistrationData
import numpy as np
from torch.utils.data import DataLoader
from model import PRNet

torch.cuda.empty_cache()
def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    #os.system('cp main.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
    #os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    #os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')

import torch
import copy
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import argparse
import numpy as np
# import open3d as o3d
import os
from scipy.spatial.transform import Rotation as R
import h5py



def train(args, net, train_loader, test_loader):
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(net.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)

    epoch_factor = args.epochs / 100.0

    scheduler = MultiStepLR(opt,
                            milestones=[int(30*epoch_factor), int(60*epoch_factor), int(80*epoch_factor)],
                            gamma=0.1)

    info_test_best = None

    for epoch in range(args.epochs):
        scheduler.step()
        info_train = net._train_one_epoch(epoch=epoch, train_loader=train_loader, opt=opt)
        info_test = net._test_one_epoch(epoch=epoch, test_loader=test_loader)

        if info_test_best is None or info_test_best['loss'] > info_test['loss']:
            info_test_best = info_test
            info_test_best['stage'] = 'best_test'

            net.save('checkpoints/%s/models/model.best.t7' % args.exp_name)

            torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': info_test_best
            }, 'checkpoints/%s/models/model.best.state_dicts.tar' % args.exp_name)

            
            
        net.logger.write(info_test_best)

        net.save('checkpoints/%s/models/model.%d.t7' % (args.exp_name, epoch))
        gc.collect()


def test(args, net, test_loader):

    info_test = net._test_one_epoch(epoch=0, test_loader=test_loader)
    print(info_test)
        



def main():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='prnet', metavar='N',
                        choices=['prnet'],
                        help='Model to use, [prnet]')
    parser.add_argument('--emb_nn', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Embedding to use, [pointnet, dgcnn]')
    parser.add_argument('--attention', type=str, default='transformer', metavar='N',
                        choices=['identity', 'transformer'],
                        help='Head to use, [identity, transformer]')
    parser.add_argument('--head', type=str, default='mlp', metavar='N',
                        choices=['mlp', 'svd'],
                        help='Head to use, [mlp, svd]')
    parser.add_argument('--n_emb_dims', type=int, default=512, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--n_blocks', type=int, default=1, metavar='N',
                        help='Num of blocks of encoder&decoder')
    parser.add_argument('--n_heads', type=int, default=4, metavar='N',
                        help='Num of heads in multiheadedattention')
    parser.add_argument('--n_iters', type=int, default=3, metavar='N',
                        help='Num of iters to run inference')
    parser.add_argument('--discount_factor', type=float, default=0.9, metavar='N',
                        help='Discount factor to compute the loss')
    parser.add_argument('--n_ff_dims', type=int, default=1024, metavar='N',
                        help='Num of dimensions of fc in transformer')
    parser.add_argument('--n_keypoints', type=int, default=512, metavar='N',
                        help='Num of keypoints to use')
    parser.add_argument('--temp_factor', type=float, default=100, metavar='N',
                        help='Factor to control the softmax precision')
    parser.add_argument('--cat_sampler', type=str, default='gumbel_softmax', choices=['softmax', 'gumbel_softmax'],
                        metavar='N', help='use gumbel_softmax to get the categorical sample')
    parser.add_argument('--dropout', type=float, default=0.05, metavar='N',
                        help='Dropout ratio in transformer')
    parser.add_argument('--batch_size', type=int, default=2, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate the model')
    parser.add_argument('--cycle_consistency_loss', type=float, default=0.1, metavar='N',
                        help='cycle consistency loss')
    parser.add_argument('--feature_alignment_loss', type=float, default=0.1, metavar='N',
                        help='feature alignment loss')
    parser.add_argument('--gaussian_noise', type=bool, default=True, metavar='N',
                        help='Wheter to add gaussian noise')
    parser.add_argument('--unseen', type=bool, default=True, metavar='N',
                        help='Wheter to test on unseen category')
    parser.add_argument('--n_points', type=int, default=1024, metavar='N',
                        help='Num of points to use')
    parser.add_argument('--n_workers', type=int, default=2, metavar='N',
                        help='Num of points to use')
    parser.add_argument('--n_subsampled_points', type=int, default=768, metavar='N',
                        help='Num of subsampled points to use')
    parser.add_argument('--dataset', type=str, default='modelnet40', choices=['modelnet40'], metavar='N',
                        help='dataset to use')
    parser.add_argument('--rot_factor', type=float, default=4, metavar='N',
                        help='Divided factor of rotation')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--vtc_testing', type=bool, default=True, metavar='N',
                        help='VTC Testing')

    args = parser.parse_args('')
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    _init_(args)

    if args.dataset == 'modelnet40':
        train_loader = DataLoader(ModelNet40(num_points=args.n_points,
                                             num_subsampled_points=args.n_subsampled_points,
                                             partition='train', gaussian_noise=args.gaussian_noise,
                                             unseen=args.unseen, rot_factor=args.rot_factor),
                                  batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.n_workers)
        
        if args.vtc_testing:
            vtc_path = 'vtc_testing_dataset.hdf5'
            test_loader = DataLoader(VTCTestRegistrationData(vtc_path, n_points=args.n_points, meter_scaled=True),
                                 batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=args.n_workers)
        else:
            test_loader = DataLoader(ModelNet40(num_points=args.n_points,
                                                num_subsampled_points=args.n_subsampled_points,
                                                partition='test', gaussian_noise=args.gaussian_noise,
                                                unseen=args.unseen, rot_factor=args.rot_factor),
                                    batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=args.n_workers)
    else:
        raise Exception("not implemented")

    if args.model == 'prnet':
        net = PRNet(args).cuda()
        if args.eval:
            if args.model_path == '':
                model_path = 'checkpoints' + '/' + args.exp_name + '/models/model.best.t7'
            else:
                model_path = args.model_path
            if not os.path.exists(model_path):
                print("can't find pretrained model")
                return
            else:
                net = torch.load(model_path)
                net.eval()
    else:
        raise Exception('Not implemented')
    if not args.eval:
        train(args, net, train_loader, test_loader)
    else:
        test(args, net, test_loader)

    print('FINISH')


if __name__ == '__main__':
    main()
