#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import os
# import dependencies
from data_generator import Generator
from aligned_data_generator import Aligned_Generator
from model import Siamese_GNN, Siamese_Matcher, GNN_Matcher
from Logger import Logger
import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

#Pytorch requirements
import unicodedata
import string
import re
import random
import argparse

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from MatchingLayer import MatchingLayer
import pickle

parser = argparse.ArgumentParser()

###############################################################################
#                             General Settings                                #
###############################################################################

parser.add_argument('--lr', nargs='?', const=1, type=float,
                    default=1e-3)
parser.add_argument('--quad_reg', nargs='?', const=1, type=float,
                    default=1e-4)
parser.add_argument('--align', nargs='?', const=1, type=int,
                    default=0)
parser.add_argument('--num_examples_train', nargs='?', const=1, type=int,
                    default=int(20000))
parser.add_argument('--num_examples_test', nargs='?', const=1, type=int,
                    default=int(1000))
parser.add_argument('--edge_density', nargs='?', const=1, type=float,
                    default=0.2)
parser.add_argument('--random_noise', action='store_true')
parser.add_argument('--noise', nargs='?', const=1, type=float, default=0.03)
parser.add_argument('--noise_model', nargs='?', const=1, type=int, default=2)
parser.add_argument('--generative_model', nargs='?', const=1, type=str,
                    default='ErdosRenyi')
parser.add_argument('--iterations', nargs='?', const=1, type=int,
                    default=int(20000))
parser.add_argument('--batch_size', nargs='?', const=1, type=int, default=1)
parser.add_argument('--mode', nargs='?', const=1, type=str, default='train')
parser.add_argument('--path_dataset', nargs='?', const=1, type=str, default='')
parser.add_argument('--path_logger', nargs='?', const=1, type=str, default='')
parser.add_argument('--print_freq', nargs='?', const=1, type=int, default=100)
parser.add_argument('--test_freq', nargs='?', const=1, type=int, default=500)
parser.add_argument('--save_freq', nargs='?', const=1, type=int, default=2000)
parser.add_argument('--clip_grad_norm', nargs='?', const=1, type=float,
                    default=40.0)

###############################################################################
#                                 GNN Settings                                #
###############################################################################

parser.add_argument('--num_features', nargs='?', const=1, type=int,
                    default=20)
parser.add_argument('--num_layers', nargs='?', const=1, type=int,
                    default=20)
parser.add_argument('--J', nargs='?', const=1, type=int, default=4)

args = parser.parse_args()

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
    torch.cuda.manual_seed(0)
else:
    dtype = torch.FloatTensor
    dtype_l = torch.LongTensor
    torch.manual_seed(0)

batch_size = args.batch_size
criterion = nn.CrossEntropyLoss()
template1 = '{:<10} {:<10} {:<10} {:<15} {:<10} {:<10} {:<10} '
template2 = '{:<10} {:<10.5f} {:<10.5f} {:<15} {:<10} {:<10} {:<10.3f} \n'

def compute_loss(pred, labels):
    pred = pred.view(-1, pred.size()[-1])
    labels = labels.view(-1)
    return criterion(pred, labels)

#def train(siamese_gnn, logger, gen):
def train(siamese_gnn, logger, gen):
    labels = (Variable(torch.arange(0, gen.N).unsqueeze(0).expand(batch_size,
              gen.N)).type(dtype_l))
    #labels have this format
    #tensor([[0, 1, 2],
    #        [0, 1, 2]]) batch size 2 - JK
    optimizer = torch.optim.Adamax(siamese_gnn.parameters(), lr=args.lr)    #1e-3)
    for it in range(args.iterations):
        start = time.time()
        input = gen.sample_batch(batch_size, cuda=torch.cuda.is_available())
        # 'input' from sample_batch return has this form:
        # [WW, X], [WW_noise, X_noise] where WW, X, WW_noise, X_noise are all batches (tens)
        #temp
        #    pickle.dump( input, open('pickle_input.p','wb') )
        #    quit("Pickle saved")
        #

        pred = siamese_gnn(*input)
        #? what form do the predictions have?
        #pred =
        #tensor([[[22.6977,  0.0827,  0.1377,  ...,  8.2557,  8.9004,  4.8111],
        #         [ 2.0629, 22.5110,  5.8868,  ...,  2.8347,  7.7718, 14.2134],
        #         [-1.5174, 10.6303, 14.9685,  ..., 12.7365,  8.1801,  7.6513],
        #         ...,
        #         [ 8.6330,  5.8210,  7.1014,  ..., 15.7287,  9.9970,  6.4952],
        #         [ 5.1720,  8.9540, 12.8027,  ...,  6.6614, 12.6969, 10.2977],
        #         [ 2.9169,  8.3249, -0.3027,  ...,  8.4153,  5.3743, 14.3038]]],
        #       device='cuda:0', grad_fn=<BmmBackward>)
        #pred[0] =
        #tensor([[22.6977,  0.0827,  0.1377,  ...,  8.2557,  8.9004,  4.8111],
        #        [ 2.0629, 22.5110,  5.8868,  ...,  2.8347,  7.7718, 14.2134],
        #        [-1.5174, 10.6303, 14.9685,  ..., 12.7365,  8.1801,  7.6513],
        #        ...,
        #        [ 8.6330,  5.8210,  7.1014,  ..., 15.7287,  9.9970,  6.4952],
        #        [ 5.1720,  8.9540, 12.8027,  ...,  6.6614, 12.6969, 10.2977],
        #        [ 2.9169,  8.3249, -0.3027,  ...,  8.4153,  5.3743, 14.3038]],
        #       device='cuda:0', grad_fn=<SelectBackward>)
        #pred.size() =
        #torch.Size([1, 50, 50])
        #gen.N =
        #50
        """
        print("pred = ")
        print( pred )
        print("pred[0] = ")
        print( pred[0] )
        print("pred.size() = ")
        print( pred.size() )
        print("gen.N = ")
        print( gen.N )
        """
        if it%100 == 0:
            print("Solved iteration {}".format(it))
        #
        loss = compute_loss(pred, labels)
        siamese_gnn.zero_grad()
        loss.backward()
        #nn.utils.clip_grad_norm(siamese_gnn.parameters(), args.clip_grad_norm)    JK: deprecated
        nn.utils.clip_grad_norm_(siamese_gnn.parameters(), args.clip_grad_norm)
        optimizer.step()
        logger.add_train_loss(loss)
        logger.add_train_accuracy(pred, labels)
        elapsed = time.time() - start
        if it % logger.args['print_freq'] == 0:
            logger.plot_train_accuracy()
            logger.plot_train_loss()
            loss = loss.data.cpu().numpy()#[0]
            info = ['iteration', 'loss', 'accuracy', 'edge_density',
                    'noise', 'model', 'elapsed']
            out = [it, loss.item(), logger.accuracy_train[-1].item(), args.edge_density,
                   args.noise, args.generative_model, elapsed]
            print(template1.format(*info))
            print(template2.format(*out))
            # test(siamese_gnn, logger, gen)
        if it % logger.args['save_freq'] == 0:
            logger.save_model(siamese_gnn)
            logger.save_results()
    print('Optimization finished.')

if __name__ == '__main__':
    logger = Logger(args.path_logger)
    logger.write_settings(args)
    #siamese_gnn = Siamese_GNN(args.num_features, args.num_layers, args.J + 2)

    align = args.align
    print('align = ', align)
    if align:
        print('Using {} '.format('Aligned_Generator'))
        gen = Aligned_Generator(args.path_dataset)
    else:
        print('Using {} '.format('Generator'))
        gen = Generator(args.path_dataset)

    # generator setup
    gen.num_examples_train = args.num_examples_train
    gen.num_examples_test = args.num_examples_test
    gen.J = args.J
    gen.edge_density = args.edge_density
    gen.random_noise = args.random_noise
    gen.noise = args.noise
    gen.noise_model = args.noise_model
    gen.generative_model = args.generative_model
    # load dataset
    # print(gen.random_noise)
    gen.load_dataset()

    matching_layer = MatchingLayer(nNodes=gen.N, eps=args.quad_reg)            #      eps=1e-4)

    if align:
        siamese_gnn = GNN_Matcher(args.num_features, args.num_layers, args.J + 2, matching_layer = matching_layer)

    else:
        siamese_gnn = Siamese_Matcher(args.num_features, args.num_layers, args.J + 2, matching_layer = matching_layer)



    if torch.cuda.is_available():
        siamese_gnn.cuda()

    if args.mode == 'train':
        train(siamese_gnn, logger, gen)
    # elif args.mode == 'test':
    #     test(siamese_gnn, logger, gen)
