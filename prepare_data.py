import wget
import os
import torch
from torchvision.datasets import CIFAR10, CIFAR100
import numpy as np
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField
from torch.utils.data import Dataset


CIFAR10_URL = "https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar10/1m.npz"
CIFAR100_URL= "https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar100/1m.npz"

AUX_PATH = "./data/aux_data/"
BASE_PATH = "./data/base_data/"
FFCV_PATH = "./data/ffcv_data/"

class AuxDataset(Dataset):
    def __init__(self, aux_path):
        data = np.load(aux_path)
        self.images = data["image"]
        self.labels = data["label"]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        return image, label


def load_dataset(dataset_name, data_dir = None, aux_path = None, aux_url = None, dataset = None):
    if dataset_name == "train":
        dataset = dataset(root=data_dir, train=True, download=True)
    elif dataset_name == "test":
        dataset = dataset(root=data_dir, train=False, download=True)
    elif dataset_name == "aux":
        download_data(aux_path, aux_url)
        dataset = AuxDataset(aux_path)
    else:
        raise Exception(f"{dataset_name} is not supported")
    return dataset


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
            'image': RGBImageField(
                write_mode='raw'
            ),
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
    ffcv_aux_path = path_maker(FFCV_PATH, dataset_name, "aux.beton")

    if aux_path is None or base_path is None or ffcv_train_path is None or ffcv_test_path is None:
        print(f"Error processing {dataset_name}.")
        return


    if dataset_name == "cifar10":
        dataset = CIFAR10
        aux_url = CIFAR10_URL
        
    elif dataset_name == "cifar100":
        dataset = CIFAR100
        aux_url = CIFAR100_URL
    else:
        print("Only cifar10 and cifar100 datasets are supported.")
        return

    trainset = load_dataset("train", data_dir=base_path, dataset=dataset)
    write(trainset, ffcv_train_path)
    del trainset

    testset = load_dataset("test", data_dir=base_path, dataset=dataset)
    write(testset, ffcv_test_path)
    del testset

    auxset = load_dataset("aux", aux_path=aux_path, aux_url=aux_url)
    write(auxset, ffcv_aux_path)
    del auxset


def main():
    print("Processing CIFAR-10 dataset...")
    process("cifar10")
    print("Processing CIFAR-100 dataset...")
    process("cifar100")


if __name__ == '__main__':
    main()