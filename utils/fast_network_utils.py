import torch
from models.resnet import resnet
from models.wide import wide_resnet

def get_network(network, depth, dataset):

    if dataset == 'cifar10':
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).cuda()
        std = torch.tensor([0.2023, 0.1994, 0.2010]).cuda()
    elif dataset == 'cifar100':
        mean = torch.tensor([0.5071, 0.4867, 0.4408]).cuda()
        std = torch.tensor([0.2675, 0.2565, 0.2761]).cuda()

    if network == 'resnet':
        model = resnet(depth=depth, dataset=dataset, mean=mean, std=std)
    elif network == 'wide':
        model = wide_resnet(depth=depth, widen_factor=10, dataset=dataset, mean=mean, std=std)
    else:
        raise NotImplementedError

    return model
