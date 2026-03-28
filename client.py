# client.py
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ======================================================
# DATASETS
# ======================================================
from datasets import (
    get_partitioned_mnist,
    get_partitioned_svhn,
    get_partitioned_cifar
)

# ======================================================
# DP / UTILS
# ======================================================
from utils import (
    compute_sigma_gaussian,
    compute_sigma_adaptive,
    add_gaussian_noise_to_state,
    flatten_state_dict
)

# ======================================================
# MODEL
# ======================================================
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, input_size=28):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        s = input_size // 2
        self.fc1 = nn.Linear(64 * s * s, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ======================================================
# CLIENT
# ======================================================
@ray.remote
class Client:
    def __init__(
        self,
        cid,
        dataset="mnist",
        malicious_id=-1,
        num_clients=3,
        batch_size=64,
        noniid=False,
        dp_mode="gaussian"
    ):
        self.cid = int(cid)
        self.dataset = dataset.lower()
        self.malicious_id = int(malicious_id)
        self.num_clients = int(num_clients)
        self.batch_size = int(batch_size)
        self.noniid = noniid
        self.dp_mode = dp_mode

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ================= DATA =================
        if self.dataset == "mnist":
            self.train_loader, self.test_loader = get_partitioned_mnist(
                cid=self.cid,
                num_clients=self.num_clients,
                batch_size=self.batch_size,
                noniid=self.noniid
            )
            in_ch, size = 1, 28

        elif self.dataset == "cifar10":
            self.train_loader, self.test_loader = get_partitioned_cifar(
                cid=self.cid,
                num_clients=self.num_clients,
                batch_size=self.batch_size,
                noniid=self.noniid
            )
            in_ch, size = 3, 32

        elif self.dataset == "svhn":
            self.train_loader, self.test_loader = get_partitioned_svhn(
                cid=self.cid,
                num_clients=self.num_clients,
                batch_size=self.batch_size,
                noniid=self.noniid
            )
            in_ch, size = 3, 32

        else:
            raise ValueError("Unsupported dataset")

        # ================= MODEL =================
        self.model = SimpleCNN(
            in_channels=in_ch,
            num_classes=10,
            input_size=size
        ).to(self.device)

        self.prev_global = {
            k: v.detach().cpu().clone()
            for k, v in self.model.state_dict().items()
        }

        self.current_round = 0
        self.max_rounds = 10

    # ==================================================
    def set_global(self, state_dict):
        self.model.load_state_dict(state_dict)
        self.prev_global = {
            k: v.detach().cpu().clone()
            for k, v in state_dict.items()
        }

    # ==================================================
    def train(self, local_epochs=1, score=1.0, dp=False, lr=0.01, clip_norm=1.0):
        self.model.train()
        loss_fn = nn.CrossEntropyLoss()
        opt = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)

        total_loss, batches = 0.0, 0

        for _ in range(local_epochs):
            for x, y in self.train_loader:

                if self.cid == self.malicious_id:
                    y = (y + 1) % 10

                x, y = x.to(self.device), y.to(self.device)

                opt.zero_grad()
                loss = loss_fn(self.model(x), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
                opt.step()

                total_loss += loss.item()
                batches += 1

        # ================= DP =================
        current_state = {
            k: v.detach().cpu().clone()
            for k, v in self.model.state_dict().items()
        }

        delta = {
            k: current_state[k] - self.prev_global[k]
            for k in current_state
        }

        delta_vec = flatten_state_dict(delta)

        if dp:
            if self.dp_mode == "adaptive":
                sigma = compute_sigma_adaptive(
                    score, self.current_round, self.max_rounds
                )
            else:
                sigma = compute_sigma_gaussian(score)

            noisy_state = add_gaussian_noise_to_state(
                current_state, sigma, device="cpu"
            )
        else:
            noisy_state = current_state

        self.prev_global = current_state

        return noisy_state, delta_vec, {
            "loss": total_loss / batches if batches else None
        }

    # ==================================================
    def eval_global(self):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return correct / total if total else 0.0
