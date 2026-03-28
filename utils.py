import torch

# ======================================================
# Gaussian DP (Base paper)
# ======================================================
def compute_sigma_gaussian(score, base_sigma=1.0):
    """
    Static Gaussian DP noise (Base paper)
    """
    return base_sigma / (score + 1e-6)


# ======================================================
# Adaptive DP (YOUR NOVELTY)
# ======================================================
def compute_sigma_adaptive(score, round_id, max_rounds):
    """
    Adaptive DP:
    - Higher noise in early rounds
    - Lower noise in later rounds
    - Lower noise for high-contribution clients
    """

    base_sigma = 1.2

    # round-aware decay
    round_factor = max(0.3, 1 - (round_id / max_rounds))

    # contribution-aware
    contrib_factor = 1 / (score + 1e-6)

    return base_sigma * round_factor * contrib_factor


# ======================================================
# Add Gaussian noise
# ======================================================
def add_gaussian_noise_to_state(state_dict, sigma, device="cpu"):
    noisy = {}
    for k, v in state_dict.items():
        noise = torch.normal(0, sigma, size=v.shape, device=device)
        noisy[k] = (v.to(device) + noise).detach().cpu()
    return noisy


# ======================================================
# Flatten model updates
# ======================================================
def flatten_state_dict(state_dict):
    return torch.cat([p.detach().view(-1).cpu() for p in state_dict.values()])
