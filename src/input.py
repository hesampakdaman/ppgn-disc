import torchvision.datasets as dset
import torchvision.utils as tvutils
import torchvision.transforms as transforms
import torch

def get_dataset(s, batch_size):
    if(s == 'mnist'):
        dataset = dset.MNIST(
                root='../data/mnist', download=True,
                transform=transforms.Compose([
                    # transforms.Scale(28),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
        )
    if(s == 'cifar10'):
        dataset = dset.MNIST(
                root='../data/cifar10', download=True,
                transform=transforms.Compose([
                    # transforms.Scale(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
        )
    dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size,shuffle=True, num_workers=2, drop_last=True)
    return dataloader

def save(x, dataset, loc, filename, nrows=8):
    if(dataset == 'mnist'):
        x = x * 0.3081 + 0.1307
        tvutils.save_image(x,'{0}/{1}.jpg'.format(loc, filename), nrows)
