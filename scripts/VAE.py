from pathlib import Path
from omegaconf import OmegaConf
from synthgen_pytorch.data.CelebA import CelebADataset

cfg = Path("/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/SynthGenDiffusion/secrets.yaml")
config = OmegaConf.load(cfg)

from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
from glob import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchsummary import summary



train_dataset = CelebADataset()
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=config.CELEBA_BATCH_SIZE,
                          shuffle=True,
                          num_workers=4)


parser = argparse.ArgumentParser(description='PyTorch VAE')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()
args.cuda = True
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 4, 2, 1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.2)
    def forward(self, x):
        return self.leaky_relu(self.batch_norm(self.conv(x)))

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, eps=1e-3):
        super().__init__()
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.pad = nn.ReplicationPad2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1)
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=eps)
        self.leakyrelu = nn.LeakyReLU(0.2) 
    def forward(self, x):
        return self.leakyrelu(self.batch_norm(self.conv(self.pad(self.upsample(x)))))

class VAE_CLEAN(nn.Module):
    """
    Vanilla Variational Autoencoder
    """
    def __init__(self, in_channels, encoder_shape, decoder_shape, latent_dim):
        super(VAE_CLEAN, self).__init__()
        self.in_channels = in_channels
        self.encoder_shape = encoder_shape
        self.decoder_shape = decoder_shape
        self.latent_dim = latent_dim
        self.encoder_mults = [(1, 2), (2, 4), (4, 8), (8, 8)]
        self.encoder = nn.Sequential(ConvBlock(self.in_channels, self.encoder_shape))
        for m in self.encoder_mults:
            self.encoder.append(
                ConvBlock(self.encoder_shape * m[0], self.encoder_shape * m[1])
                )
        self.linear_mu = nn.Linear(self.encoder_shape*8*4*4, latent_dim)
        self.linear_logvar = nn.Linear(self.encoder_shape*8*4*4, latent_dim)
        self.decoder_mults = [(16, 8), (8, 4), (4, 2), (2, 1)]
        self.decoder = nn.Sequential()
        for m in self.decoder_mults:
            self.decoder.append(
                DecoderBlock(self.decoder_shape * m[0], self.decoder_shape * m[1])
            )
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.linear_decode = nn.Linear(self.latent_dim, self.decoder_shape*8*2*4*4)
        self.out_conv = nn.Conv2d(self.decoder_shape, self.in_channels, 3, 1)
        self.out_pad = nn.ReplicationPad2d(1)
        self.out_upsample = nn.UpsamplingNearest2d(scale_factor=2)
    def encode(self, x):
        # convblocks
        for layer in self.encoder:
            x = layer(x)
        # linear transform
        x = x.view(-1, self.encoder_shape*8*4*4)
        return self.linear_mu(x), self.linear_logvar(x)
    def decode(self, z):
        # linear transform from latent shape
        h1 = self.relu(self.linear_decode(z))
        h1 = h1.view(-1, self.decoder_shape*8*2, 4, 4)
        for layer in self.decoder:
            h1 = layer(h1)
        # transform to normalized vector
        return self.sigmoid(self.out_conv(self.out_pad(self.out_upsample(h1))))
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if args.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    def get_latent_var(self, x):
        mu, logvar = self.encode(x.view(-1, self.in_channels, self.encoder_shape, self.decoder_shape))
        z = self.reparametrize(mu, logvar)
        return z
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.in_channels, self.encoder_shape, self.decoder_shape))
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return res, mu, logvar



# WLOG
model = VAE_CLEAN(in_channels=3, encoder_shape=128, decoder_shape=128, latent_dim=500)



if args.cuda:
    model.cuda()

reconstruction_function = nn.BCELoss()
reconstruction_function.size_average = False
def loss_function(recon_x, x, mu, logvar):
    BCE = reconstruction_function(recon_x, x)
    # https://arxiv.org/abs/1312.6114 (Appendix B)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return BCE + KLD

optimizer = optim.Adam(model.parameters(), lr=3e-4)

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), (len(train_loader)*128),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / (len(train_loader)*128)))
    return train_loss / (len(train_loader)*128)



for epoch in range(0, 100, 1):
    train_loss = train(epoch)
    torch.save(model.state_dict(), 
        '/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/checkpoints/Epoch_{}_Train_loss_{:.4f}.pth'.format(epoch, train_loss))




























# def test(epoch):
#     model.eval()
#     test_loss = 0
#     for batch_idx in test_loader:
#         data = load_batch(batch_idx, False)
#         data = Variable(data, volatile=True)
#         if args.cuda:
#             data = data.cuda()
#         recon_batch, mu, logvar = model(data)
#         test_loss += loss_function(recon_batch, data, mu, logvar).data[0]

#         torchvision.utils.save_image(data.data, '../imgs/Epoch_{}_data.jpg'.format(epoch), nrow=8, padding=2)
#         torchvision.utils.save_image(recon_batch.data, '../imgs/Epoch_{}_recon.jpg'.format(epoch), nrow=8, padding=2)

#     test_loss /= (len(test_loader)*128)
#     print('====> Test set loss: {:.4f}'.format(test_loss))
#     return test_loss


def perform_latent_space_arithmatics(items): # input is list of tuples of 3 [(a1,b1,c1), (a2,b2,c2)]
    load_last_model()
    model.eval()
    data = [im for item in items for im in item]
    data = [totensor(i) for i in data]
    data = torch.stack(data, dim=0)
    data = Variable(data, volatile=True)
    if args.cuda:
        data = data.cuda()
    z = model.get_latent_var(data.view(-1, model.nc, model.ndf, model.ngf))
    it = iter(z.split(1))
    z = zip(it, it, it)
    zs = []
    numsample = 11
    for i,j,k in z:
        for factor in np.linspace(0,1,numsample):
            zs.append((i-j)*factor+k)
    z = torch.cat(zs, 0)
    recon = model.decode(z)

    it1 = iter(data.split(1))
    it2 = [iter(recon.split(1))]*numsample
    result = zip(it1, it1, it1, *it2)
    result = [im for item in result for im in item]

    result = torch.cat(result, 0)
    torchvision.utils.save_image(result.data, '../imgs/vec_math.jpg', nrow=3+numsample, padding=2)


def latent_space_transition(items): # input is list of tuples of  (a,b)
    load_last_model()
    model.eval()
    data = [im for item in items for im in item[:-1]]
    data = [totensor(i) for i in data]
    data = torch.stack(data, dim=0)
    data = Variable(data, volatile=True)
    if args.cuda:
        data = data.cuda()
    z = model.get_latent_var(data.view(-1, model.nc, model.ndf, model.ngf))
    it = iter(z.split(1))
    z = zip(it, it)
    zs = []
    numsample = 11
    for i,j in z:
        for factor in np.linspace(0,1,numsample):
            zs.append(i+(j-i)*factor)
    z = torch.cat(zs, 0)
    recon = model.decode(z)

    it1 = iter(data.split(1))
    it2 = [iter(recon.split(1))]*numsample
    result = zip(it1, it1, *it2)
    result = [im for item in result for im in item]

    result = torch.cat(result, 0)
    torchvision.utils.save_image(result.data, '../imgs/trans.jpg', nrow=2+numsample, padding=2)


def rand_faces(num=5):
    load_last_model()
    model.eval()
    z = torch.randn(num*num, model.latent_variable_size)
    z = Variable(z, volatile=True)
    if args.cuda:
        z = z.cuda()
    recon = model.decode(z)
    torchvision.utils.save_image(recon.data, '../imgs/rand_faces.jpg', nrow=num, padding=2)

def load_last_model():
    models = glob('../models/*.pth')
    model_ids = [(int(f.split('_')[1]), f) for f in models]
    start_epoch, last_cp = max(model_ids, key=lambda item:item[0])
    print('Last checkpoint: ', last_cp)
    model.load_state_dict(torch.load(last_cp))
    return start_epoch, last_cp

def resume_training():
    start_epoch, _ = load_last_model()

    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        train_loss = train(epoch)
        test_loss = test(epoch)
        torch.save(model.state_dict(), '../models/Epoch_{}_Train_loss_{:.4f}_Test_loss_{:.4f}.pth'.format(epoch, train_loss, test_loss))

def last_model_to_cpu():
    _, last_cp = load_last_model()
    model.cpu()
    torch.save(model.state_dict(), '../models/cpu_'+last_cp.split('/')[-1])

if __name__ == '__main__':
    resume_training()
    # last_model_to_cpu()
    # load_last_model()
    # rand_faces(10)
    # da = load_pickle(test_loader[0])
    # da = da[:120]
    # it = iter(da)
    # l = zip(it, it, it)
    # # latent_space_transition(l)
    # perform_latent_space_arithmatics(l)