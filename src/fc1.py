from __future__ import division
from __future__ import print_function

from torch.autograd import Variable
import torch
import torch.autograd as autograd
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as tvutils

import argparse
import random
import numpy as np
import os
import sys
sys.path.append(os.getcwd())

import input
import image_saver
import lossplot

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0

###################
# GLOBAL VARIABLES
###################
DIM = 64 # Model dimensionality
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
OUTPUT_DIM = 784 # Number of pixels in MNIST (28*28)

parser = argparse.ArgumentParser()
parser.add_argument('--sample', action='store_true')
parser.add_argument('--loadweights', action='store_true')
parser.add_argument('--cuda', type=int, default=1)
ARGS = parser.parse_args()
ARGS.ckpts = '../ckpts/fc1'
ARGS.samplepth = '../samples/fc1'
ARGS.plotspth = '../plots/plot_fc1full'

EPOCH_MAX = 15
BATCH_SIZE = 32
ARGS.lr = 1e-4
ARGS.hfeat = 64
ARGS.Glayer = 0
ARGS.Dlayer = 0

if torch.cuda.is_available() and not ARGS.cuda:
    print("WARNING: You have a CUDA device, but it is not enabled.")
cudnn.benchmark = True

# ==================Definition Start======================

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 4*DIM, 2)
        self.conv2 = nn.Conv2d(4*DIM, 4*DIM, 2)
        self.conv3 = nn.Conv2d(4*DIM, 8*DIM, 2)
        self.fc1 = nn.Linear(8*DIM, 1)

    def forward(self, x):
        x = x.view(-1, 1, 8, 8)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
        x = self.fc1(x)
        # print(x.size())
        return x.view(-1)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(ARGS.hfeat, 5*5*DIM)
        self.deconv1 = nn.ConvTranspose2d(DIM, 8*DIM, 5)
        self.deconv2 = nn.ConvTranspose2d(8*DIM, 4*DIM, 5)
        self.deconv3 = nn.ConvTranspose2d(4*DIM, 4*DIM, 7)
        self.deconv4 = nn.ConvTranspose2d(4*DIM, 1, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x.view(-1, DIM, 5, 5)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = self.deconv4(x)
        # print(x.size())
        return x.view(-1, OUTPUT_DIM)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 7)
        self.conv2 = nn.Conv2d(64, 128, 7)
        self.conv3 = nn.Conv2d(128, 256, 7)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        outs = [] # for storing outputs from different layers

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = x.view(-1, x.size(1)*x.size(2)*x.size(3))

        x = F.relu(self.fc1(x))
        outs.append(x)

        x = self.fc2(x)
        outs.append(x)

        return outs

def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(real_data.size(0), 1) # uniformly random [0,1)
    alpha = alpha.expand(real_data.size()) # expands across columns
    alpha = alpha.cuda(gpu) if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda(gpu)
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def sampler(e1, e2, e3, iters, cols, filename='samples'):
    ''' MALA-approx sampler  with parameters e1, e2, e3.  Each sample is
    run iteratively iters times. Cols is  the total number of columns in
    the figure that this function outputs.
    '''
    netG.eval()
    classes = 10
    N = cols * classes # total number of samples
    size = [1, ARGS.hfeat]
    samples = torch.zeros(N, size[1])
    if(use_cuda):
        samples = samples.cuda()
    i = 0 # rows in samples array

    sys.stdout.write('\r[fc1] [sample] [0/%d]' %(N))
    sys.stdout.write("\033[K")
    sys.stdout.flush()

    for c in range(classes):
        for _ in range(cols):
            h_val = np.random.normal(loc=0, scale=1, size=size)
            h = Variable(torch.FloatTensor(h_val).cuda(), requires_grad=True)
            for _ in range(iters):
                x = netG(h).view(-1, 1, 28, 28)
                C_c = F.log_softmax(netE(x)[-1])[0,c]
                R_h = netE(x)[ARGS.Glayer].cpu().data.numpy()
                norm = np.random.normal(loc=0, scale=e3, size=size)

                # derivatives
                netE.zero_grad()
                netG.zero_grad()
                C_c.backward()
                dC_c = h.grad.cpu().data.numpy()

                # MALA approx
                h_val += e1 * (R_h - h_val) + e2 * dC_c + norm

                h = Variable(
                    torch.FloatTensor(h_val).cuda(), requires_grad=True)

            # append samples
            samples[i] = h.data

            i += 1

            sys.stdout.write('\r[fc1] [sample] [%d/%d]' %(i,N))
            sys.stdout.flush()

    samples = Variable(samples, volatile=True)
    samples = netG(samples).view(-1, 1, 28, 28)
    if(use_cuda):
        samples = samples.cpu().data
    else:
        samples = samples.data
    image_saver.save(samples, 'mnist', ARGS.samplepth, filename, cols)
    netG.train() # needed if samples created during training

# ==================Definition End======================

netE = Encoder()
netG = Generator()
netD = Discriminator()

try:
    netE.load_state_dict(torch.load('{0}/E.pth'.format('../ckpts/encoder')))
    netE.eval()
except:
    print('No Encoder weights found. Exiting.')
    # do not continue without pretrained encoder
    sys.exit(1)

if(ARGS.loadweights or ARGS.sample):
    try:
        netG.load_state_dict(torch.load('{0}/G.pth'.format(ARGS.ckpts)))
        print('Generator weights loaded.')
    except:
        print('No Generator weights found. Initializing model.')
    if(not ARGS.sample):
        try:
            netD.load_state_dict(torch.load('{0}/D.pth'.format(ARGS.ckpts)))
            print('Discriminator weights loaded.')
        except:
            print('No Discriminator weights found. Initializing model.')
else:
    print('Intializing model weights.')

if use_cuda:
    netD = netD.cuda(gpu)
    netE = netE.cuda(gpu)
    netG = netG.cuda(gpu)

if(ARGS.sample):
    sampler(1e-2, 1e0, 1e-15, 200, 10, 'fc1full')
    print('')
    sys.exit(1)

optimizerD = optim.Adam(netD.parameters(), lr=ARGS.lr, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=ARGS.lr, betas=(0.5, 0.9))

one = torch.FloatTensor([1])
mone = one * -1
if use_cuda:
    one = one.cuda(gpu)
    mone = mone.cuda(gpu)

dataset = input.get_dataset('mnist', BATCH_SIZE)
MSE = nn.MSELoss() # mean sq loss

# number of training steps
D_step = 0
G_step = 0

# running wasserstein
wasserstein = []

print('')

for epoch in range(1, EPOCH_MAX+1):
    data = iter(dataset)
    datalen = len(dataset)
    i = 0

    if(epoch > 1):
        sampler(1e-2, 1e0, 1e-15, 200, 1, str(epoch-1))

    while i < datalen:
        ############################
        # (1) Update D network
        ###########################
        if(G_step < 25 or G_step % 500 == 0):
            Diters = 100
        else:
            Diters = 5
        j = 0
        ###########################
        # Train discriminator
        ###########################
        while j < Diters and i < len(dataset):
            netD.zero_grad()
            netE.zero_grad()
            netG.zero_grad()

            # real data
            j += 1
            _data, _ = data.next()
            _data = _data.view(-1, 28*28)
            i += 1
            real_data = torch.Tensor(_data)
            if use_cuda:
                real_data = real_data.cuda(gpu)
            real_data_v = Variable(real_data)

            # real input h for D
            h = netE(real_data_v.view(-1, 1, 28, 28))[ARGS.Dlayer].data
            if use_cuda:
                h = h.cuda(gpu)

            # train with real
            hv = Variable(h)
            D_real = netD(hv)
            D_real = D_real.mean()
            D_real.backward(mone)

            # train with fake
            hv = Variable(h)
            x_fake = netG(hv)
            fake = netE(x_fake.view(-1, 1, 28, 28))[ARGS.Dlayer]
            inputv = Variable(fake.data, requires_grad=False)
            D_fake = netD(inputv)
            D_fake = D_fake.mean()
            D_fake.backward(one)

            gradient_penalty = calc_gradient_penalty(netD, hv.data, fake.data)
            gradient_penalty.backward()

            optimizerD.step()
            D_step += 1

        # for printing loss
        D_cost = D_fake.data[0] - D_real.data[0] + gradient_penalty.data[0]

        # wasserstein estimate
        wasserstein.append(D_real.data[0] - D_fake.data[0])

        ############################
        # (2) Update G network
        ############################
        netD.zero_grad()
        netE.zero_grad()
        netG.zero_grad()

        # input for G
        h = netE(real_data_v.view(-1, 1, 28, 28))[ARGS.Glayer].data
        if use_cuda:
            h = h.cuda(gpu)

        # L_gan
        hv = Variable(h)
        x_fake = netG(hv)
        fake = netE(x_fake.view(-1, 1, 28, 28))[ARGS.Dlayer]
        G = netD(fake)
        L_gan = G.mean()
        L_gan.backward(mone)
        G_cost = -L_gan.data[0]

        # L_x
        hv = Variable(h, requires_grad=False)
        w_x = 2e0
        fake = netG(hv)
        L_x = w_x * MSE(fake, real_data_v)
        L_x.backward()

        # L_h
        hv = Variable(h, requires_grad=False)
        w_h = 1e-1
        fake = netG(hv).view(-1, 1, 28, 28)
        h_fake = netE(fake)[ARGS.Glayer]
        L_h = w_h * MSE(h_fake, hv)
        L_h.backward()

        optimizerG.step()
        G_step += 1


        sys.stdout.write(
                '\r[fc1] [Epoch: %d/%d] [%d] [L_D: %f] '\
                '[L_gan: %f  L_x: %f  L_h: %f]'\
                %(
                    epoch, EPOCH_MAX, G_step, D_cost, G_cost,
                    L_x.data[0], L_h.data[0]
                )
        )
        sys.stdout.write("\033[K")
        sys.stdout.flush()

    torch.save(netG.state_dict(), '{0}/G.pth'.format(ARGS.ckpts))
    torch.save(netD.state_dict(), '{0}/D.pth'.format(ARGS.ckpts))

    print('\n')

lossplot.wassersteinplot(wasserstein, ARGS.plotspth)
sampler(1e-2, 1e0, 1e-15, 200, 1, EPOCH_MAX)
