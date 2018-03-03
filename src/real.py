from __future__ import division

from torch.autograd import Variable
import torch
import torch.autograd as autograd
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as tvutils

import argparse
import random
import numpy as np
import os
import sys

import input
import image_saver

sys.path.append(os.getcwd())

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0

###################
# GLOBAL VARIABLES
###################
BATCH_SIZE = 500 # Batch size
OUTPUT_DIM = 784 # Number of pixels in MNIST (28*28)

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=1)
ARGS = parser.parse_args()
ARGS.samplepth = '../samples/reals'

if torch.cuda.is_available() and not ARGS.cuda:
    print("WARNING: You have a CUDA device, but it is not enabled.")
cudnn.benchmark = True

# ==================Definition Start======================

def generate_image(x):
    input.save(x.data, 'mnist', ARGS.samplepth, 'reals', 10)

# ==================Definition End======================

dataset = input.get_dataset('mnist', BATCH_SIZE)

sorted = torch.zeros(100, 1, 28, 28)

data = iter(dataset)
_data, _label = data.next()

sample = 0 # current sample
for i in range(10): # all digits
    class_sample = 0 # track num of sample for each class
    for j in range(len(_label)): # search through labels until we find a match
        if(class_sample == 10):
            break
        elif(_label[j] == i):
            sorted[sample, :] = _data[j]
            sample += 1
            digit_sample += 1
            j += 1

# generate_image(Variable(sorted))
