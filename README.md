# 🔐 Federated Learning with Adaptive Differential Privacy (Fed-RDP)

## 📌 Project Overview
This project implements a **privacy-preserving Federated Learning system** using **Adaptive Differential Privacy (DP)** and **contribution-based noise scaling**.

The goal is to:
- Protect user data privacy
- Improve model robustness
- Handle malicious or low-quality clients
- Maintain model accuracy while ensuring privacy

---

## 🚀 Key Concepts Used

### 🔹 Federated Learning (FL)
- Training happens on **multiple clients locally**
- Only model updates are shared (not raw data)

### 🔹 Differential Privacy (DP)
- Adds **Gaussian noise** to model updates
- Protects individual data from being inferred

### 🔹 Adaptive Noise Scaling (Our Novelty)
- Noise level (σ) changes based on client contribution:
  - High contribution → Low noise
  - Low contribution → High noise

### 🔹 Contribution Score
- Measures how useful each client update is
- Used for:
  - Noise scaling
  - Weighted aggregation

---

## 🧠 System Architecture

1. Server initializes global model  
2. Sends model to clients  
3. Clients train locally  
4. Compute update (Δ = Local - Global)  
5. Add adaptive noise  
6. Send updates to server  
7. Server aggregates updates  
8. Repeat for multiple rounds  

---
## 📂 Project Structure
ray-fed-basic/
│
├── run_experiment.py # Main execution file
├── client.py # Client-side training
├── model.py # CNN model definition
├── datasets.py # Dataset loading
├── reputation.py # Contribution scoring
├── weight_sum.py # Aggregation logic
├── utils.py # Helper functions
├── ledger.py # Stores updates (blockchain-like)
├── plot_*.py # Graph generation
├── requirements.txt # Dependencies
├── README.md


---

## 📊 Datasets Used

- MNIST (Handwritten digits)
- CIFAR-10 (Image classification)
- SVHN (Street View House Numbers)

---

## 📈 Evaluation Metrics

- Accuracy vs Rounds
- Contribution Score
- Stability
- DP Error Impact
- Training Time

---

## ⚙️ Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/Dinesh8454/Fed-RDP-Project.git
cd Fed-RDP-Project
Step 2: Create Virtual Environment (Optional)
python -m venv venv
venv\Scripts\activate   # Windows
Step 3: Install Requirements
pip install -r requirements.txt
▶️ How to Run
🔹 Basic Run (MNIST)
python run_experiment.py --dataset mnist --clients 3 --rounds 10
🔹 With Differential Privacy
python run_experiment.py --dataset mnist --clients 3 --rounds 10 --dp
🔹 Adaptive DP (Our Proposed Model)
python run_experiment.py --dataset mnist --clients 3 --rounds 10 --dp --adaptive
📊 Output
After running, outputs are saved in:

outputs/
Includes:

Accuracy vs rounds graph

Contribution scores

Stability plots

JSON logs (ledger)

## 📂 Project Structure
