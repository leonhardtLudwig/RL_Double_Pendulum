import numpy as np
import matplotlib.pyplot as plt
import os
from config import RUN_NAME, LOG_DIR_BASE


log_dir = os.path.join(LOG_DIR_BASE, RUN_NAME)
data = np.load(os.path.join(log_dir, "evaluations.npz"))

# contenuto di evaluations.npz
timesteps = data["timesteps"]        # step a cui è stata fatta la valutazione
results   = data["results"]          # shape (n_eval, n_eval_episodes) — reward per episodio
ep_lengths = data["ep_lengths"]      # shape (n_eval, n_eval_episodes) — lunghezza episodi

mean_rewards = results.mean(axis=1)
std_rewards  = results.std(axis=1)
mean_lengths = ep_lengths.mean(axis=1)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# ── reward media di valutazione ────────────────────────────────────────────────
axes[0].plot(timesteps, mean_rewards, color="steelblue", linewidth=2, label="Mean eval reward")
axes[0].fill_between(timesteps,
                     mean_rewards - std_rewards,
                     mean_rewards + std_rewards,
                     alpha=0.2, color="steelblue", label="±1 std")
axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
axes[0].set_ylabel("Mean Episode Return")
axes[0].set_xlabel("Timesteps")
axes[0].set_title("Evaluation Reward during Training")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# ── lunghezza media degli episodi ──────────────────────────────────────────────
axes[1].plot(timesteps, mean_lengths, color="darkorange", linewidth=2)
axes[1].set_ylabel("Mean Episode Length (steps)")
axes[1].set_xlabel("Timesteps")
axes[1].set_title("Mean Episode Length during Training")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("eval_curve.png", dpi=150)
