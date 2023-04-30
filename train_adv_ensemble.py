# Import built-in module
import argparse
import warnings
import copy
from contextlib import nullcontext
warnings.filterwarnings(action='ignore')

# Import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.autograd import Variable

# Import Custom Utils
from utils.fast_network_utils import get_network
from utils.fast_data_utils import get_fast_dataloader
from utils.utils import *

# attack loader
# from attack.attack import attack_loader
from attack.fastattack import attack_loader

# Accelerating forward and backward
from torch.cuda.amp import GradScaler, autocast

# Import loss for ensemble training
from utils.loss import trades_loss

# import sam optimizer
from utils.sam import SAM

#Import composer 
import composer.functional as cf

torch.backends.cudnn.benchmark = True
torch.autograd.profiler.emit_nvtx(False)
torch.autograd.profiler.profile(False)
torch.autograd.set_detect_anomaly(False)
torch.set_flush_denormal(True)

# fetch args
parser = argparse.ArgumentParser()

# model parameter
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--network', default='resnet', type=str)
parser.add_argument('--depth', default=18, type=int)
parser.add_argument('--num_models', default=3, type=int)
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--port', default='12356', type=str)
parser.add_argument('--resume', default=False, type=bool)

# learning parameter
parser.add_argument('--learning_rate', default=0.1, type=float)
parser.add_argument('--weight_decay', default=0.0002, type=float)
parser.add_argument('--batch_size', default=128, type=float)
parser.add_argument('--test_batch_size', default=256, type=float)
parser.add_argument('--epoch', default=10, type=int)

# attack parameter only for CIFAR-10 and SVHN
parser.add_argument('--attack', default='pgd', type=str)
parser.add_argument('--eps', default=0.03, type=float)
parser.add_argument('--steps', default=10, type=int)

# loss parameters
parser.add_argument('--beta', default=5.0, type=float)
parser.add_argument('--lamda', default=0.1, type=float)
parser.add_argument('--log_det_lamda', default=1.0, type=float)
parser.add_argument('--label_smoothing', default=0.1, type=float)
parser.add_argument('--num_classes', default=10, type=int)

# composer addons
parser.add_argument('--blurpool', default=True, type=bool)
parser.add_argument('--EMA', default=True, type=bool)
parser.add_argument('--SAM', default=True, type=bool)


args = parser.parse_args()

# the number of gpus for multi-process
gpu_list = list(map(int, args.gpu.split(',')))
ngpus_per_node = len(gpu_list)

# cuda visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = args.port

# global best_acc
best_acc = 0

# Mix Training
scaler = GradScaler()
sam_scaler = GradScaler()

class Ensemble(nn.Module):
    def __init__(self, models):
        super(Ensemble, self).__init__()
        self.models = models
        assert len(self.models) > 0

    def forward(self, x):
        if len(self.models) > 1:
            outputs = 0
            for model in self.models:
                outputs += F.softmax(model(x), dim=-1)
            output = outputs / len(self.models)
            output = torch.clamp(output, min=1e-40)
            return torch.log(output)
        else:
            return self.models[0](x)
         

def train(nets, ema_nets, trainloader, optimizer, lr_scheduler, scaler, attack):
    for net in nets:
        net.train()
    train_loss = 0
    adv_correct = 0
    correct = 0
    total = 0

    desc = ('[Train/LR=%.3f] Loss: %.3f | Acc: %.3f%%' %
            (lr_scheduler.get_lr()[0], 0, 0))

    prog_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=desc, leave=True)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.cuda(), targets.cuda()
        with track_bn_stats(nets, False):
            adv_inputs = attack(inputs, targets)

        # Accerlating forward propagation
        optimizer.zero_grad(set_to_none=True)

        if args.SAM:
            loss, _, _ = trades_loss(
                inputs, 
                adv_inputs, 
                targets, 
                nets,
                beta = args.beta,
                lamda = args.lamda,
                log_det_lamda = args.log_det_lamda,
                num_classes = args.num_classes,
                label_smoothing = args.label_smoothing
            )
            sam_scaler.scale(loss).backward()
            sam_scaler.unscale_(optimizer)
            optimizer.first_step(zero_grad=True)
            sam_scaler.update()
        
        # second forward-backward pass
        with track_bn_stats(nets, False) if args.SAM else nullcontext():
            loss, nat_outputs, adv_outputs = trades_loss(
                inputs, 
                adv_inputs, 
                targets, 
                nets,
                beta = args.beta,
                lamda = args.lamda,
                log_det_lamda = args.log_det_lamda,
                num_classes = args.num_classes,
                label_smoothing = args.label_smoothing
            )

        # Accerlating backward propagation
        scaler.scale(loss).backward()
        if args.SAM: optimizer.second_step(zero_grad=True)
        scaler.step(optimizer)
        scaler.update()

        for net, ema_net in zip(nets, ema_nets):
            cf.compute_ema(net, ema_net, smoothing=0.99)


        # scheduling for Cyclic LR
        lr_scheduler.step()

        train_loss += loss.item()
        _, predicted = nat_outputs.max(1)
        _, adv_predicted = adv_outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        adv_correct += adv_predicted.eq(targets).sum().item()

        desc = ('[Train/LR=%.3f] Loss: %.3f | Acc: %.3f%% | Adv Acc: %.3f%%' %
                (lr_scheduler.get_lr()[0], train_loss / (batch_idx + 1), 100. * correct / total, 100. * adv_correct / total))
        prog_bar.set_description(desc, refresh=True)



def test(nets, testloader, attack, rank):
    global best_acc
    for net in nets:
        net.eval()
    test_loss = 0
    correct = 0
    total = 0
    desc = ('[Test/Clean] Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(0+1), 0, correct, total))

    ensemble = Ensemble(nets)

    prog_bar = tqdm(enumerate(testloader), total=len(testloader), desc=desc, leave=False)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.cuda(), targets.cuda()

        # Accerlating forward propagation
        with autocast():
            outputs = ensemble(inputs)
            loss = F.cross_entropy(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        desc = ('[Test/Clean] Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)

    # Save clean acc.
    clean_acc = 100. * correct / total

    test_loss = 0
    correct = 0
    total = 0

    desc = ('[Test/PGD] Loss: %.3f | Acc: %.3f%%'
            % (test_loss / (0 + 1), 0))

    prog_bar = tqdm(enumerate(testloader), total=len(testloader), desc=desc, leave=False)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs = attack(inputs, targets)
        inputs, targets = inputs.cuda(), targets.cuda()

        # Accerlating forward propagation
        with autocast():
            outputs = ensemble(inputs)
            loss = F.cross_entropy(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        desc = ('[Test/PGD] Loss: %.3f | Acc: %.3f%%'
                % (test_loss / (batch_idx + 1), 100. * correct / total))
        prog_bar.set_description(desc, refresh=True)

    # Save adv acc.
    adv_acc = 100. * correct / total

    # compute acc
    acc = (clean_acc + adv_acc)/2

    rprint('Current Accuracy is {:.2f}/{:.2f}!!'.format(clean_acc, adv_acc), rank)

    if acc > best_acc:
        state = {
            'nets': [net.state_dict() for net in nets],
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')

        best_acc = acc
        if rank == 0:
            torch.save(state, './checkpoint/%s/%s_adv_%s%s_best.t7' % (args.dataset, args.dataset,
                                                                                args.network,
                                                                                args.depth))
            print('Saving~ ./checkpoint/%s/%s_adv_%s%s_best.t7' % (args.dataset, args.dataset,
                                                                            args.network,
                                                                            args.depth))


def main_worker(rank, ngpus_per_node=ngpus_per_node):

    # print configuration
    print_configuration(args, rank)

    # setting gpu id of this process
    torch.cuda.set_device(rank)

    # DDP environment settings
    print("Use GPU: {} for training".format(gpu_list[rank]))
    dist.init_process_group(backend='nccl', world_size=ngpus_per_node, rank=rank)

    # init model and Distributed Data Parallel
    nets = []
    ema_nets = []
    for i in range(args.num_models):
        net = get_network(network=args.network,
                        depth=args.depth,
                        dataset=args.dataset)

        # composer Algorithms
        if args.blurpool: cf.apply_blurpool(net, replace_convs=True, replace_maxpools=True, blur_first=True)

        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = net.to(memory_format=torch.channels_last).cuda()
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank], output_device=[rank], broadcast_buffers=False)
        nets.append(net)
        if args.EMA:
            ema_nets.append(copy.deepcopy(net))

    # fast init dataloader
    trainloader, testloader, decoder = get_fast_dataloader(dataset=args.dataset,
                                                  train_batch_size=args.batch_size,
                                                  test_batch_size=args.test_batch_size)

    # Load Plain Network
    if args.resume:
        checkpoint_name = './checkpoint/%s/%s_%s%s_best.t7' % (args.dataset, args.dataset, args.network, args.depth)
        checkpoint = torch.load(checkpoint_name, map_location=torch.device(torch.cuda.current_device()))
        for i in range(args.num_models):
            nets[i].load_state_dict(checkpoint['nets'][i])
        rprint(f'==> {checkpoint_name}', rank)
        rprint('==> Successfully Loaded Standard checkpoint..', rank)

    # Attack loader
    ensemble = Ensemble(nets)
    if args.dataset == 'imagenet' or args.dataset == 'tiny':
        rprint('Fast FGSM training', rank)
        attack = attack_loader(net=ensemble, attack='fgsm_train', eps=2/255 if args.dataset == 'imagenet' else 0.03, steps=args.steps)
    else:
        rprint('PGD training', rank)
        attack = attack_loader(net=ensemble, attack=args.attack, eps=args.eps, steps=args.steps)

    # init optimizer and lr scheduler
    parameters = []
    for net in nets:
        parameters += list(net.parameters())
    if args.SAM:
        base_optimizer = optim.SGD
        optimizer = SAM(parameters, base_optimizer, lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(parameters, lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0, max_lr=args.learning_rate,
    step_size_up=5 * len(trainloader) if args.dataset != 'imagenet' else 2 * len(trainloader),
    step_size_down=(args.epoch - 5) * len(trainloader) if args.dataset != 'imagenet' else (args.epoch - 2) * len(trainloader))

    # training and testing
    for epoch in range(args.epoch):
        rprint('\nEpoch: %d' % epoch, rank)
        if args.dataset == "imagenet":
            res = get_resolution(epoch=epoch, min_res=160, max_res=192, end_ramp=25, start_ramp=18)
            decoder.output_size = (res, res)
        train(nets, ema_nets, trainloader, optimizer, lr_scheduler, scaler, attack)
        if args.EMA:
            test(ema_nets, testloader, attack, rank)
        else:
            test(nets, testloader, attack, rank)

    # destroy process
    dist.destroy_process_group()

def run():
    torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, join=True)

if __name__ == '__main__':
    run()