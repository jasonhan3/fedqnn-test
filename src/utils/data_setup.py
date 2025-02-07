"""
Contains functionality for creating PyTorch DataLoaders for image classification data.
"""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset, Dataset
from .common import *

# Define a custom dataset wrapper that remaps the labels
class RelabelSubset(Dataset):
    def __init__(self, subset, label_mapping):
        """
        Args:
            subset (Dataset): The original dataset or Subset.
            label_mapping (dict): A dictionary mapping original labels to new labels.
        """
        self.subset = subset
        self.label_mapping = label_mapping

    def __getitem__(self, index):
        # Get the data from the underlying subset
        x, y = self.subset[index]
        # Remap the label using the provided dictionary
        new_y = self.label_mapping.get(y, y)  # Defaults to y if not found in mapping
        return x, new_y

    def __len__(self):
        return len(self.subset)

NUM_WORKERS = os.cpu_count()

# Normalization values for the different datasets
NORMALIZE_DICT = {    
    'cifar': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'MRI': dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    # TODO: this is just from online; verify it
    'MNIST': dict(mean=(0.1307, 0.1307, 0.137), std=(0.3081, 0.3081, 0.3081))   
    }

def split_data_client(dataset, num_clients, seed):
    """
    This function is used to split the dataset into train and test for each client.
    :param dataset: the dataset to split (type: torch.utils.data.Dataset)
    :param num_clients: the number of clients
    :param seed: the seed for the random split
    """
    # Split training set into `num_clients` partitions to simulate different local datasets
    partition_size = len(dataset) // num_clients
    lengths = [partition_size] * (num_clients - 1)
    lengths += [len(dataset) - sum(lengths)]
    print(f"split_data_client, len(lengths): {len(lengths)}")
    print(f"split_data_client, lengths: {lengths}")
    ds = random_split(dataset, lengths, torch.Generator().manual_seed(seed))
    return ds


# Define model, architecture and dataset
# The DataLoaders downloads the training and test data that are then normalized.
def load_datasets(num_clients: int, batch_size: int, resize: int, seed: int, num_workers: int, splitter=10,
                  dataset="cifar", data_path="./data/", data_path_val="", classes_of_interest=[4, 9]):
    """
    This function is used to load the dataset and split it into train and test for each client.
    :param num_clients: the number of clients
    :param batch_size: the batch size
    :param resize: the size of the image after resizing (if None, no resizing)
    :param seed: the seed for the random split
    :param num_workers: the number of workers
    :param splitter: percentage of the training data to use for validation. Example: 10 means 10% of the training data
    :param dataset: the name of the dataset in the data folder
    :param data_path: the path of the data folder
    :param data_path_val: the absolute path of the validation data (if None, no validation data)
    :return: the train and test data loaders
    """
    
    list_transforms = [transforms.ToTensor(), transforms.Normalize(**NORMALIZE_DICT[dataset])]
    print(dataset)

    if dataset == "cifar":
        # Download and transform CIFAR-10 (train and test)
        transformer = transforms.Compose(
            list_transforms
        )
        trainset = datasets.CIFAR10(data_path + dataset, train=True, download=True, transform=transformer)
        testset = datasets.CIFAR10(data_path + dataset, train=False, download=True, transform=transformer)
       
    elif dataset == "MRI":
        if resize is not None:
            list_transforms = [transforms.Resize((resize, resize))] + list_transforms

        transformer = transforms.Compose(list_transforms)
        supp_ds_store(data_path + dataset)
        supp_ds_store(data_path + dataset + "/Training")
        supp_ds_store(data_path + dataset + "/Testing")
        trainset = datasets.ImageFolder(data_path + dataset + "/Training", transform=transformer)
        testset = datasets.ImageFolder(data_path + dataset + "/Testing", transform=transformer)
    elif dataset == "MNIST":

        if resize is not None:
            list_transforms = [transforms.Resize((resize, resize))] + list_transforms
        
        list_transforms = [transforms.Lambda(lambda img: img.convert("RGB"))] + list_transforms

        transformer = transforms.Compose(
            list_transforms
        )


        
        full_train = datasets.MNIST(root=data_path + dataset, train=True, download=True, transform=transformer)

        full_test = datasets.MNIST(root=data_path + dataset, train=False, download=True, transform=transformer)

        # Create lists of indices for which the label is either 4 or 9
        train_indices = [i for i, (_, label) in enumerate(full_train) if label in classes_of_interest]
        test_indices = [i for i, (_, label) in enumerate(full_test) if label in classes_of_interest]

        print(f"load_datasets, train_indices: {train_indices}")
        print(f"load_datasets, test_indices: {test_indices}")

        # Create subsets using these indices
        trainset = Subset(full_train, train_indices)
        testset = Subset(full_test, test_indices)

        # Define the mapping from old labels to new labels
        mapping = {classes_of_interest[0]: 0, classes_of_interest[1]: 1}

        trainset = RelabelSubset(trainset, mapping)
        testset = RelabelSubset(testset, mapping)

    if (dataset == "cifar" or dataset == "MRI"):
        print(f"The training set is created for the classes : {trainset.classes}")
    elif (dataset == "MNIST"):
        print(f"The training set is created for the classes: {trainset.dataset.classes}")        

    # Split training set into `num_clients` partitions to simulate different local datasets
    datasets_train = split_data_client(trainset, num_clients, seed)
    print(f"load_datasets, type(datasets_train): {type(datasets_train)}")
    print(f"load_datasets, datasets_train: {datasets_train}")
    if data_path_val:
        valset = datasets.ImageFolder(data_path_val, transform=transformer)
        datasets_val = split_data_client(valset, num_clients, seed)    

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for i in range(num_clients):
        if data_path_val:
            # if we already have a validation dataset
            trainloaders.append(DataLoader(datasets_train[i], batch_size=batch_size, shuffle=True))
            valloaders.append(DataLoader(datasets_val[i], batch_size=batch_size))
        else:
            len_val = int(len(datasets_train[i]) * splitter / 100)  # splitter % validation set
            len_train = len(datasets_train[i]) - len_val
            lengths = [len_train, len_val]
            ds_train, ds_val = random_split(datasets_train[i], lengths, torch.Generator().manual_seed(seed))
            trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
            valloaders.append(DataLoader(ds_val, batch_size=batch_size))

    testloader = DataLoader(testset, batch_size=batch_size)
    return trainloaders, valloaders, testloader