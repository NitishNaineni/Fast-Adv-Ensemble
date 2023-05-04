import wget
import os
import torch
import torchvision
import numpy as np
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField


CIFAR10_URL = "https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar10/1m.npz"
CIFAR100_URL= "https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar100/1m.npz"

AUX_PATH = "./data/aux_data/"
BASE_PATH = "./data/base_data/"
FFCV_PATH = "./data/ffcv_data/"

class SemiSupervisedDataset(torch.utils.data.Dataset):
    """
    A dataset with auxiliary pseudo-labeled data.
    """
    def __init__(self, aux_data_filename=None, train=False, **kwargs):

        self.load_base_dataset(train, **kwargs)
        
        self.train = train

        if self.train:
            if aux_data_filename is not None:
                aux_path = aux_data_filename
                aux = np.load(aux_path)
                aux_data = aux['image']
                aux_targets = aux['label']
                orig_len = len(self.data)

                self.data = np.concatenate((self.data, aux_data), axis=0)
                self.targets.extend(aux_targets)
    
    def load_base_dataset(self, **kwargs):
        raise NotImplementedError()
    
    @property
    def data(self):
        return self.dataset.data

    @data.setter
    def data(self, value):
        self.dataset.data = value

    @property
    def targets(self):
        return self.dataset.targets

    @targets.setter
    def targets(self, value):
        self.dataset.targets = value

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        self.dataset.labels = self.targets
        return self.dataset[item]


class SemiSupervisedCIFAR10(SemiSupervisedDataset):
    """
    A dataset with auxiliary pseudo-labeled data for CIFAR10.
    """
    def load_base_dataset(self, train=False, **kwargs):
        self.dataset = torchvision.datasets.CIFAR10(train=train, **kwargs)
        self.dataset_size = len(self.dataset)


def load_cifar10(data_dir, cifar10_aux_path):
    download_data(cifar10_aux_path, CIFAR10_URL)
    train_dataset = SemiSupervisedCIFAR10(root=data_dir, train=True, download=True,aux_data_filename=cifar10_aux_path)
    test_dataset = SemiSupervisedCIFAR10(root=data_dir, train=False, download=True)
    
    return train_dataset, test_dataset

class SemiSupervisedCIFAR100(SemiSupervisedDataset):
    """
    A dataset with auxiliary pseudo-labeled data for CIFAR100.
    """
    def load_base_dataset(self, train=False, **kwargs):
        self.dataset = torchvision.datasets.CIFAR100(train=train, **kwargs)
        self.dataset_size = len(self.dataset)


def load_cifar100(data_dir, cifar100_aux_path):
    download_data(cifar100_aux_path, CIFAR100_URL)
    train_dataset = SemiSupervisedCIFAR100(root=data_dir, train=True, download=True,aux_data_filename=cifar100_aux_path)
    test_dataset = SemiSupervisedCIFAR100(root=data_dir, train=False, download=True)
    
    return train_dataset, test_dataset

def download_data(aux_path, url):
    """
    Download auxiliary data if it does not already exist.
    """
    if not os.path.isfile(aux_path):
        print("Downloading auxiliary data...")
        wget.download(url, aux_path)
        print(f"\nFile saved to: {aux_path}\n")
    else:
        print(f"File already exists at: {aux_path}")


def write(dataset, path):
    """
    Write the FFCV dataset to the specified path.
    """
    if os.path.isfile(path):
        print(f"File already exists at: {path}")
    else:
        print(f"Writing FFCV dataset to {path}")
        writer = DatasetWriter(path, {
            'image': RGBImageField(),
            'label': IntField()
        })
        writer.from_indexed_dataset(dataset)


def path_maker(root, dataset_name, filename):
    """
    Create a directory path for the given root, dataset_name, and filename.
    """
    path = os.path.join(root, dataset_name)
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError as e:
        print(f"Error creating directory: {e}")
        return None
    path = os.path.join(path, filename)
    return path


def process(dataset_name):
    """
    Process the specified dataset, downloading auxiliary data if needed,
    and writing the FFCV dataset to the appropriate paths.
    """
    aux_path = path_maker(AUX_PATH, dataset_name, "1m.npz")
    base_path = path_maker(BASE_PATH, dataset_name, "")
    ffcv_train_path = path_maker(FFCV_PATH, dataset_name, "train.beton")
    ffcv_test_path = path_maker(FFCV_PATH, dataset_name, "test.beton")

    if aux_path is None or base_path is None or ffcv_train_path is None or ffcv_test_path is None:
        print(f"Error processing {dataset_name}.")
        return

    if dataset_name == "cifar10":
        trainset, testset = load_cifar10(base_path, aux_path)
    elif dataset_name == "cifar100":
        trainset, testset = load_cifar100(base_path, aux_path)
    else:
        print("Only cifar10 and cifar100 datasets are supported.")
        return

    write(trainset, ffcv_train_path)
    write(testset, ffcv_test_path)


def main():
    print("Processing CIFAR-10 dataset...")
    process("cifar10")
    print("Processing CIFAR-100 dataset...")
    process("cifar100")


if __name__ == '__main__':
    main()