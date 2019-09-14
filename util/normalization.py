import torch
from models.vgg_extractor import get_VGG_features

def get_normalization_data(dataset, opt):
    mean = 0.
    std = 0.
    nb_samples = 0.
    for _, data in enumerate(dataset):
        data = data['A']
        batch_samples = data.size(0)
        data = get_VGG_features(data, opt.vgg_relu).detach()
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    mean = mean[None,:,None,None]
    std = std[None,:,None,None]

    return mean, std