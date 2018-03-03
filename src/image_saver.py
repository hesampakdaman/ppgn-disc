import torchvision.utils as tvutils

def save(x, dataset, loc, filename, nrows=8):
    if(dataset == 'mnist'):
        # x = x * 0.3081 + 0.1307
        tvutils.save_image(x,'{0}/{1}.jpg'.format(loc, filename), nrows)
