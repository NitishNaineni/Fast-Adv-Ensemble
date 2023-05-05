# Import built-in modules
import argparse
import warnings
import copy
from contextlib import nullcontext
from typing import Optional
warnings.filterwarnings(action='ignore')

# Import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.autograd import Variable
from torch.func import functional_call, stack_module_state, replace_all_batch_norm_modules_
from torch import vmap, Tensor
from torch.nn import Parameter
from torch.nn.modules.module import _global_parameter_registration_hooks, _global_buffer_registration_hooks

# Import custom utils
from utils.fast_network_utils import get_network
from utils.fast_data_utils import get_fast_dataloader
from utils.utils import *

# Import attack loader
from attack.fastattack import attack_loader

# Accelerating forward and backward
from torch.cuda.amp import GradScaler, autocast

# Import loss for ensemble training
from utils.loss import trades_loss

# Import SAM optimizer
from utils.sam import SAM

# Import composer
import composer.functional as cf

# Set global parameters for performance optimization
torch.backends.cudnn.benchmark = True
torch.autograd.profiler.emit_nvtx(False)
torch.autograd.profiler.profile(False)
torch.autograd.set_detect_anomaly(False)
torch.set_flush_denormal(True)

# Fetch arguments
parser = argparse.ArgumentParser()

# Model parameters
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--network', default='resnet', type=str)
parser.add_argument('--depth', default=18, type=int)
parser.add_argument('--num_models', default=3, type=int)
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--port', default='12356', type=str)
parser.add_argument('--resume', default=False, type=bool)

# Learning parameters
parser.add_argument('--learning_rate', default=0.1, type=float)
parser.add_argument('--weight_decay', default=0.0002, type=float)
parser.add_argument('--batch_size', default=128, type=float)
parser.add_argument('--test_batch_size', default=256, type=float)
parser.add_argument('--epoch', default=10, type=int)

# Attack parameters only for CIFAR-10 and SVHN
parser.add_argument('--attack', default='pgd', type=str)
parser.add_argument('--eps', default=0.03, type=float)
parser.add_argument('--steps', default=10, type=int)

# Loss parameters
parser.add_argument('--beta', default=5.0, type=float)
parser.add_argument('--lamda', default=0.1, type=float)
parser.add_argument('--log_det_lamda', default=1.0, type=float)
parser.add_argument('--label_smoothing', default=0.1, type=float)
parser.add_argument('--num_classes', default=10, type=int)

# Composer addons
parser.add_argument('--blurpool', default=True, type=bool)
parser.add_argument('--EMA', default=True, type=bool)
parser.add_argument('--SAM', default=False, type=bool)

args = parser.parse_args()

# The number of GPUs for multi-process
gpu_list = list(map(int, args.gpu.split(',')))
ngpus_per_node = len(gpu_list)

# Set CUDA visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = args.port

# Set global best accuracy
best_acc = 0

# Mix Training
scaler = GradScaler()
sam_scaler = GradScaler()

class Ensemble(nn.Module):
    def __init__(self, models):
        """
        Initializes an ensemble model given a list of models.

        Args:
        - models (list): a list of individual models to be combined in the ensemble
        """
        super(Ensemble, self).__init__()
        params, buffers = stack_module_state(models)
        self.num_models = len(models)
        self.base_model = copy.deepcopy(models[0])

        # Rename the parameters and buffers of the base_model
        # param_names = []
        # for name, param in self.base_model.named_parameters():
        #     updated_name = name.replace('.', '_')
        #     param_names.append(name)
        #     self.base_model.register_parameter(updated_name, param)

        # for name in param_names:
        #     del self.base_model._parameters[name]

        # # Rename buffers
        # buffer_names = []
        # for name, buffer in self.base_model.named_buffers():
        #     updated_name = name.replace('.', '_')
        #     buffer_names.append(name)
        #     self.base_model.register_buffer(updated_name, buffer)

        # for name in buffer_names:
        #     del self.base_model._buffers[name]

        # Register stacked parameters and buffers
        self.params = {}
        self.buffers = {}
        for name, param in params.items():
            # updated_name = name.replace('.', '_')
            self.register_parameter(name, nn.Parameter(param))
            self.params[name] = param
        for name, buffer in buffers.items():
            # updated_name = name.replace('.', '_')
            self.register_buffer(name, buffer)
            self.buffers[name] = buffer

        # Add each model as a module with a unique name
        # for i, model in enumerate(models):
        #     self.add_module('model_{}'.format(i), model)
    
    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        if '_parameters' not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call")

        elif not isinstance(name, str):
            raise TypeError("parameter name should be a string. "
                            "Got {}".format(torch.typename(name)))
        elif name == '':
            raise KeyError("parameter name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError("attribute '{}' already exists".format(name))

        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError("cannot assign '{}' object to parameter '{}' "
                            "(torch.nn.Parameter or None required)"
                            .format(torch.typename(param), name))
        elif param.grad_fn:
            raise ValueError(
                "Cannot assign non-leaf Tensor to parameter '{0}'. Model "
                "parameters must be created explicitly. To express '{0}' "
                "as a function of another Tensor, compute the value in "
                "the forward() method.".format(name))
        else:
            for hook in _global_parameter_registration_hooks.values():
                output = hook(self, name, param)
                if output is not None:
                    param = output
            self._parameters[name] = param

    def register_buffer(self, name: str, tensor: Optional[Tensor], persistent: bool = True) -> None:
        if persistent is False and isinstance(self, torch.jit.ScriptModule):
            raise RuntimeError("ScriptModule does not support non-persistent buffers")

        if '_buffers' not in self.__dict__:
            raise AttributeError(
                "cannot assign buffer before Module.__init__() call")
        elif not isinstance(name, str):
            raise TypeError("buffer name should be a string. "
                            "Got {}".format(torch.typename(name)))
        elif name == '':
            raise KeyError("buffer name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._buffers:
            raise KeyError("attribute '{}' already exists".format(name))
        elif tensor is not None and not isinstance(tensor, torch.Tensor):
            raise TypeError("cannot assign '{}' object to buffer '{}' "
                            "(torch Tensor or None required)"
                            .format(torch.typename(tensor), name))
        else:
            for hook in _global_buffer_registration_hooks.values():
                output = hook(self, name, tensor)
                if output is not None:
                    tensor = output
            self._buffers[name] = tensor
            if persistent:
                self._non_persistent_buffers_set.discard(name)
            else:
                self._non_persistent_buffers_set.add(name)

    def forward(self, x, ensemble=True):
        """
        Forward pass for the ensemble model given an input.

        Args:
        - x (tensor): input data
        - ensemble (bool, optional): whether to use the ensemble mode or not; default is True

        Returns:
        - output (tensor): log softmax of the average prediction probabilities of the ensemble models if ensemble=True,
                           otherwise returns a tensor with outputs from each model stacked along the first dimension
        """
        def fmodel(params, buffers, minibatch):
            return functional_call(self.base_model, (params, buffers), (minibatch,))
        output = vmap(fmodel, in_dims=(0, 0, None))(self.params, self.buffers, x)
        if ensemble:
            output = F.softmax(output, dim=-1).mean(dim=0)
            # Clamp the output to prevent taking the log of zero
            output = torch.log(torch.clamp(output, min=1e-40))
        return output
         

def train(ensemble, ema_ensemble, trainloader, optimizer, lr_scheduler, scaler, attack):
    """
    Trains an ensemble of models on the given training data using the specified optimizer, learning rate scheduler,
    and gradient scaler.

    Args:
        ensemble : The ensemble of models to train.
        ema_ensemble : The ensemble of EMA models.
        trainloader : The data loader for the training data.
        optimizer (torch.optim.Optimizer): The optimizer.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        scaler (torch.cuda.amp.GradScaler): The gradient scaler.
        attack (callable): The function to generate adversarial examples.

    Returns:
        A tuple containing the average training loss (float), the number of correct predictions on the original inputs
        (int), and the number of correct predictions on the adversarial examples (int).
    """
    ensemble.train()
    train_loss = 0
    adv_correct = 0
    correct = 0
    total = 0

    desc = ('[Train/LR=%.3f] Loss: %.3f | Acc: %.3f%%' %
            (lr_scheduler.get_lr()[0], 0, 0))

    prog_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=desc, leave=True)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.cuda(), targets.cuda()
        with track_bn_stats(ensemble, False):
            adv_inputs = attack(inputs, targets)

        # Accerlating forward propagation
        optimizer.zero_grad(set_to_none=True)

        if args.SAM:
            loss, _, _ = trades_loss(
                inputs, 
                adv_inputs, 
                targets, 
                ensemble,
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
        with track_bn_stats(ensemble, False) if args.SAM else nullcontext():
            loss, nat_outputs, adv_outputs = trades_loss(
                inputs, 
                adv_inputs, 
                targets, 
                ensemble,
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

        if args.EMA:
            cf.compute_ema(ensemble, ema_ensemble, smoothing=0.99)

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



def test(ensemble, testloader, attack, rank):
    """
    Test the model on clean and adversarial examples

    Args:
    - ensemble: The ensemble of models to test.
    - testloader: DataLoader object
    - attack: attack function
    - rank: integer specifying the rank of the current process

    Returns:
    - None
    """
    global best_acc
    ensemble.eval()
    test_loss = 0
    correct = 0
    total = 0
    desc = ('[Test/Clean] Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(0+1), 0, correct, total))

    # Test on clean examples
    prog_bar = tqdm(enumerate(testloader), total=len(testloader), desc=desc, leave=False)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.cuda(), targets.cuda()

        # Accelerating forward propagation
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

    # Test on adversarial examples
    prog_bar = tqdm(enumerate(testloader), total=len(testloader), desc=desc, leave=False)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs = attack(inputs, targets)
        inputs, targets = inputs.cuda(), targets.cuda()

        # Accelerating forward propagation
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

    # Save checkpoint if this is the best accuracy achieved so far
    if acc > best_acc:
        state = {
            'ensemble': ensemble.state_dict(),
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
    
    rprint('Test Nat Acc: {:.2f} | Adv Acc: {:.2f}'.format(clean_acc, adv_acc), rank)


def main_worker(rank, ngpus_per_node=ngpus_per_node):
    """
    Initialize DDP environment, model, optimizer, and scheduler
    Perform the training and testing of the model

    Args:
    rank (int): process rank of current node
    ngpus_per_node (int): the number of GPUs per node

    Returns:
    None
    """

    # print configuration
    print_configuration(args, rank)

    # setting gpu id of this process
    torch.cuda.set_device(rank)

    # DDP environment settings
    print("Use GPU: {} for training".format(gpu_list[rank]))
    dist.init_process_group(backend='nccl', world_size=ngpus_per_node, rank=rank)

    # init model and Distributed Data Parallel
    def generate_net():
        net = get_network(network=args.network,depth=args.depth,dataset=args.dataset)
        if args.blurpool: cf.apply_blurpool(net, replace_convs=True, replace_maxpools=True, blur_first=True)
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = net.to(memory_format=torch.channels_last).cuda()
        return net

    nets = []

    for i in range(args.num_models):
        nets.append(generate_net())
    ensemble = Ensemble(nets)
    
    ensemble = torch.nn.parallel.DistributedDataParallel(ensemble, device_ids=[rank], output_device=[rank], broadcast_buffers=False)
    ensemble.num_models = args.num_models
    if args.EMA:
        ema_ensemble = copy.deepcopy(ensemble)

    # fast init dataloader
    trainloader, testloader = get_fast_dataloader(dataset=args.dataset,
                                                  train_batch_size=args.batch_size,
                                                  test_batch_size=args.test_batch_size)

    # Load Plain Network
    if args.resume:
        checkpoint_name = './checkpoint/%s/%s_%s%s_best.t7' % (args.dataset, args.dataset, args.network, args.depth)
        checkpoint = torch.load(checkpoint_name, map_location=torch.device(torch.cuda.current_device()))
        ensemble.load_state_dict(checkpoint['ensemble'])
        rprint(f'==> {checkpoint_name}', rank)
        rprint('==> Successfully Loaded Standard checkpoint..', rank)

    # Attack loader
    rprint('PGD training', rank)
    attack = attack_loader(net=ensemble, attack=args.attack, eps=args.eps, steps=args.steps)

    # init optimizer and lr scheduler
    if args.SAM:
        optimizer = SAM(ensemble.parameters(), optim.SGD, lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(ensemble.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, 
        base_lr=0, 
        max_lr=args.learning_rate,
        step_size_up=5 * len(trainloader),
        step_size_down=(args.epoch - 5) * len(trainloader)
    )

    # training and testing
    for epoch in range(args.epoch):
        rprint('\nEpoch: %d' % epoch, rank)
        train(ensemble, ema_ensemble, trainloader, optimizer, lr_scheduler, scaler, attack)
        if args.EMA:
            test(ema_ensemble, testloader, attack, rank)
        else:
            test(ensemble, testloader, attack, rank)

    # destroy process
    dist.destroy_process_group()

def run():
    torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, join=True)

if __name__ == '__main__':
    run()