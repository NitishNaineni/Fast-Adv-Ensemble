from typing import List

import torch

from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, \
    RandomTranslate, ToTensor, ToTorchImage, ToDevice
from ffcv.transforms.common import Squeeze

from typing import Callable, Optional, Tuple
from dataclasses import replace
from ffcv.pipeline.state import State
from ffcv.pipeline.allocation_query import AllocationQuery


class ToDevice_modified(ToDevice):
    def __init__(self, device, non_blocking=True):
        super(ToDevice_modified, self).__init__(device, non_blocking)

    def generate_code(self):
        def to_device(inp, dst):
            if len(inp.shape) == 4:
                if inp.is_contiguous(memory_format=torch.channels_last):
                    dst = dst.reshape(inp.shape[0], inp.shape[2], inp.shape[3], inp.shape[1])
                    dst = dst.permute(0, 3, 1, 2)

            if len(inp.shape) == 0:
                inp = inp.unsqueeze(0)

            dst = dst[:inp.shape[0]]
            dst.copy_(inp, non_blocking=self.non_blocking)
            return dst

        return to_device

class Normalize_and_Convert(Operation):
    def __init__(self, target_dtype, target_norm_bool):
        super().__init__()
        self.target_dtype = target_dtype
        self.target_norm_bool = target_norm_bool

    def generate_code(self) -> Callable:
        def convert(inp, dst):
            if self.target_norm_bool:
                inp = inp / 255.0
            return inp.type(self.target_dtype)

        convert.is_parallel = True

        return convert

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return replace(previous_state, dtype=self.target_dtype), None

def get_fast_dataloader(dataset, train_batch_size, test_batch_size, num_workers=16, dist=True):

    gpu = f'cuda:{torch.cuda.current_device()}'

    if dataset == 'cifar10':
        mean = torch.tensor([0.4914, 0.4822, 0.4465])*255
        img_size = 32
    elif dataset == 'cifar100':
        mean = torch.tensor([0.5071, 0.4867, 0.4408])*255
        img_size = 32
    else:
        raise Exception(f"{dataset} dataset is not supported")
   
    # for small dataset
    paths = {
        'train': f'./data/ffcv_data/{dataset}/train.beton',
        'test': f'./data/ffcv_data/{dataset}/test.beton'
    }

    loaders = {}
    for name in ['train', 'test']:
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]
        label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice_modified(torch.device(gpu)), Squeeze()]
        if name == 'train':
            image_pipeline.extend([
                RandomHorizontalFlip(),
                RandomTranslate(padding=int(img_size / 8.), fill=tuple(map(int, mean))),
            ])
        image_pipeline.extend([
            ToTensor(),
            ToDevice_modified(torch.device(gpu), non_blocking=True),
            ToTorchImage(),
            Normalize_and_Convert(torch.float16, True)
        ])

        order = OrderOption.RANDOM if name == 'train' else OrderOption.SEQUENTIAL

        loaders[name] = Loader(paths[name], batch_size=train_batch_size if name == 'train' else test_batch_size,
                            num_workers=num_workers, order=order, drop_last=(name == 'train'),
                                pipelines={'image': image_pipeline, 'label': label_pipeline})


    return loaders['train'], loaders['test']