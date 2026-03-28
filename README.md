Conversation opened. 1 unread message.

Skip to content
Using Gmail with screen readers
1 of 1,189
(no subject)
Inbox
Summarize this email

dinesh swamy goruputi | AP22110010960
1:59 PM (0 minutes ago)
to me

Fed-RDP: Robust Federated Learning with Reputation & Differential Privacy
This repository contains an implementation of the Fed-RDP framework, a robust federated learning approach that combines contribution evaluation, reputation-based aggregation, and differential privacy (DP).
The implementation supports MNIST, CIFAR-10, and SVHN datasets and is fully runnable using Docker + Ray.
📌 Project Overview
Traditional federated learning (FL) (e.g., FedAvg) assumes all clients are honest and does not protect against:
Malicious or low-quality clients
Privacy leakage from model updates
Fed-RDP addresses this by:
Evaluating client contributions using similarity-based scores
Assigning reputation weights to clients
Applying Differential Privacy (DP) noise to model updates
Aggregating models using weighted sums instead of simple averaging
🧠 What We Implemented
Models & Methods
We implemented and compared:
Centralized / Base Model (DL) – Single-node training
FedAvg – Standard federated averaging
Fed-RDP – Federated learning with:
Contribution evaluation
Reputation-based aggregation
Differential Privacy (Gaussian noise)
Supported Datasets
MNIST
CIFAR-10
SVHN
🏗️ Repository Structure
Copy code

ray-fed-basic/
│
├── client.py # Client-side training & DP noise
├── run_experiment.py # Server orchestration (rounds, aggregation)
├── datasets.py # MNIST / CIFAR-10 / SVHN loaders
├── contribute_eval.py # Contribution & similarity scoring
├── reputation.py # Reputation calculation
├── ledger.py # Logging scores & weights
├── utils.py # DP noise & helper functions
├── Dockerfile # Docker environment
├── requirements.txt
├── outputs_* # Saved plots & logs
└── README.md
🐳 Running the Project (Docker)
1️⃣ Build Docker Image
Copy code
Bash
docker build -t team-python-env:local .
2️⃣ Start Docker Container
Copy code
Bash
docker run --rm -it --shm-size=5.0gb \
  -e RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 \
  -v "${PWD}:/app" \
  team-python-env:local bash
▶️ Running Experiments
MNIST – Fed-RDP
Copy code
Bash
python3 run_experiment.py \
  --dataset mnist \
  --clients 3 \
  --rounds 10 \
  --local-epochs 1 \
  --dp \
  --contrib similarity \
  --outdir outputs_rdp_mnist
CIFAR-10 – Fed-RDP
Copy code
Bash
python3 run_experiment.py \
  --dataset cifar10 \
  --clients 3 \
  --rounds 10 \
  --local-epochs 1 \
  --dp \
  --contrib similarity \
  --outdir outputs_rdp_cifar10
SVHN – Fed-RDP
Copy code
Bash
python3 run_experiment.py \
  --dataset svhn \
  --clients 3 \
  --rounds 10 \
  --local-epochs 1 \
  --dp \
  --contrib similarity \
  --outdir outputs_rdp_svhn
🧾 Command Arguments Explained
Argument
Meaning
--dataset
Dataset name (mnist / cifar10 / svhn)
--clients
Number of federated clients
--rounds
Number of communication rounds
--local-epochs
Epochs each client trains locally per round
--dp
Enables Differential Privacy
--contrib similarity
Contribution evaluation method
--outdir
Folder to save outputs & plots
🔄 What Happens During Training
Round starts
Server sends global model to all clients
Each client:
Trains locally (local-epochs)
Computes update (delta)
Adds DP noise (if enabled)
Server:
Computes contribution scores
Updates client reputation
Aggregates models using weighted sum
Accuracy & loss are logged
Next round begins
📊 Outputs Generated
Each run saves results in the specified outdir:
accuracy_vs_round.png
scores_vs_round.png
ledger.json (scores, weights, metadata)
These plots replicate the graphs shown in the reference Fed-RDP paper, such as:
Score fluctuations across rounds
Weighted sum comparisons
DP noise impact
🔐 Differential Privacy (DP)
Noise Type: Gaussian noise
Where added: Client-side model updates
Effect:
Higher privacy protection
Increased loss
Slight decrease in accuracy (expected & correct behavior)
This matches prior research and the original Fed-RDP paper results.
📈 Observations
MNIST converges smoothly
CIFAR-10 & SVHN show higher loss due to:
Complex data
DP noise accumulation
Increasing loss with DP is expected and acceptable
