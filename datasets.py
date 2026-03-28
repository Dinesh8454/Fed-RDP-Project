# datasets.py
import os
import gzip
import struct
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

# ======================================================
# Helpers
# ======================================================

def _make_loader(ds, batch_size: int, shuffle: bool, num_workers: int = 0):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False
    )

def _mnist_raw_available(root: str = "./data"):
    raw = os.path.join(root, "MNIST", "raw")
    required = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]
    return os.path.isdir(raw) and all(
        os.path.isfile(os.path.join(raw, f)) for f in required
    )

# ======================================================
# Transforms
# ======================================================

MNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

CIFAR_TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    ),
])

CIFAR_TEST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    ),
])

SVHN_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),
])

# ======================================================
# MNIST RAW LOADER
# ======================================================

def _read_idx_images(gz_path):
    with gzip.open(gz_path, 'rb') as f:
        _, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(n, rows, cols)

def _read_idx_labels(gz_path):
    with gzip.open(gz_path, 'rb') as f:
        _, n = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

def load_mnist_from_raw(root="./data"):
    raw = os.path.join(root, "MNIST", "raw")
    X_train = _read_idx_images(os.path.join(raw, "train-images-idx3-ubyte.gz")) / 255.0
    y_train = _read_idx_labels(os.path.join(raw, "train-labels-idx1-ubyte.gz"))
    X_test  = _read_idx_images(os.path.join(raw, "t10k-images-idx3-ubyte.gz")) / 255.0
    y_test  = _read_idx_labels(os.path.join(raw, "t10k-labels-idx1-ubyte.gz"))

    train_ds = TensorDataset(
        torch.tensor(X_train).unsqueeze(1).float(),
        torch.tensor(y_train).long()
    )
    test_ds = TensorDataset(
        torch.tensor(X_test).unsqueeze(1).float(),
        torch.tensor(y_test).long()
    )
    return train_ds, test_ds

# ======================================================
# STANDARD DATASETS
# ======================================================

def get_mnist(batch_size=64, root="./data"):
    if _mnist_raw_available(root):
        train_ds, test_ds = load_mnist_from_raw(root)
    else:
        train_ds = datasets.EMNIST(
            root, split="mnist", train=True,
            download=True, transform=MNIST_TRANSFORM
        )
        test_ds = datasets.EMNIST(
            root, split="mnist", train=False,
            download=True, transform=MNIST_TRANSFORM
        )

    return (
        _make_loader(train_ds, batch_size, True),
        _make_loader(test_ds, batch_size, False)
    )

def get_cifar10(batch_size=64, root="./data"):
    train_ds = datasets.CIFAR10(
        root, train=True, download=True,
        transform=CIFAR_TRAIN_TRANSFORM
    )
    test_ds = datasets.CIFAR10(
        root, train=False, download=True,
        transform=CIFAR_TEST_TRANSFORM
    )
    return (
        _make_loader(train_ds, batch_size, True),
        _make_loader(test_ds, batch_size, False)
    )

def get_svhn(batch_size: int = 64, root: str = "./data"):
    train_ds = datasets.SVHN(
        root=root, split="train",
        download=True, transform=SVHN_TRANSFORM
    )
    test_ds = datasets.SVHN(
        root=root, split="test",
        download=True, transform=SVHN_TRANSFORM
    )
    return (
        _make_loader(train_ds, batch_size, True),
        _make_loader(test_ds, batch_size, False)
    )

# ======================================================
# IID PARTITION (UNCHANGED)
# ======================================================

def _iid_partition(n, num_clients):
    sizes = [n // num_clients] * num_clients
    for i in range(n % num_clients):
        sizes[i] += 1
    idx = np.random.permutation(n)
    parts, start = [], 0
    for s in sizes:
        parts.append(idx[start:start+s].tolist())
        start += s
    return parts

# ======================================================
# NON-IID DIRICHLET LABEL SPLIT (NEW)
# ======================================================

def dirichlet_split(labels, num_clients, alpha=0.5):
    labels = np.array(labels)
    n_classes = len(np.unique(labels))

    class_indices = [np.where(labels == c)[0] for c in range(n_classes)]
    client_indices = [[] for _ in range(num_clients)]

    for c in range(n_classes):
        np.random.shuffle(class_indices[c])
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (np.cumsum(proportions) * len(class_indices[c])).astype(int)[:-1]
        splits = np.split(class_indices[c], proportions)

        for i, idx in enumerate(splits):
            client_indices[i].extend(idx.tolist())

    return client_indices

# ======================================================
# PARTITIONED MNIST
# ======================================================

def get_partitioned_mnist(cid, num_clients, batch_size=64,
                          root="./data", noniid=False, alpha=0.5):
    if _mnist_raw_available(root):
        train_ds, test_ds = load_mnist_from_raw(root)
        labels = train_ds.tensors[1].numpy()
    else:
        train_ds = datasets.EMNIST(
            root, split="mnist", train=True,
            download=True, transform=MNIST_TRANSFORM
        )
        test_ds = datasets.EMNIST(
            root, split="mnist", train=False,
            download=True, transform=MNIST_TRANSFORM
        )
        labels = np.array(train_ds.targets)

    if noniid:
        parts = dirichlet_split(labels, num_clients, alpha)
    else:
        parts = _iid_partition(len(train_ds), num_clients)

    sub_train = Subset(train_ds, parts[cid])

    return (
        _make_loader(sub_train, batch_size, True),
        _make_loader(test_ds, batch_size, False)
    )

# ======================================================
# PARTITIONED CIFAR-10
# ======================================================

def get_partitioned_cifar(cid, num_clients, batch_size=64,
                          root="./data", noniid=False, alpha=0.5):
    train_ds = datasets.CIFAR10(
        root, train=True, download=True,
        transform=CIFAR_TRAIN_TRANSFORM
    )
    test_ds = datasets.CIFAR10(
        root, train=False, download=True,
        transform=CIFAR_TEST_TRANSFORM
    )

    labels = np.array(train_ds.targets)

    if noniid:
        parts = dirichlet_split(labels, num_clients, alpha)
    else:
        parts = _iid_partition(len(train_ds), num_clients)

    sub_train = Subset(train_ds, parts[cid])

    return (
        _make_loader(sub_train, batch_size, True),
        _make_loader(test_ds, batch_size, False)
    )

# ======================================================
# PARTITIONED SVHN
# ======================================================

def get_partitioned_svhn(cid: int, num_clients: int, batch_size=64,
                         root="./data", noniid=False, alpha=0.5):
    train_ds = datasets.SVHN(
        root=root, split="train",
        download=True, transform=SVHN_TRANSFORM
    )
    test_ds = datasets.SVHN(
        root=root, split="test",
        download=True, transform=SVHN_TRANSFORM
    )

    labels = np.array(train_ds.labels)

    if noniid:
        parts = dirichlet_split(labels, num_clients, alpha)
    else:
        parts = _iid_partition(len(train_ds), num_clients)

    sub_train = Subset(train_ds, parts[cid])

    return (
        _make_loader(sub_train, batch_size, True),
        _make_loader(test_ds, batch_size, False)
    )

# ======================================================
# SMALL VALIDATION SPLIT
# ======================================================

def small_val_loader_from_test(test_loader, max_samples=2000):
    n = min(len(test_loader.dataset), max_samples)
    sub = Subset(test_loader.dataset, list(range(n)))
    return _make_loader(sub, test_loader.batch_size, False)
