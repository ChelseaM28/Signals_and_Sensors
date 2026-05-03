import numpy as np

# ── Background ────────────────────────────────────────────────────────────────
# We are tracking a target satellite. Our orbital model predicts what its
# signal should look like: expected signal strength (dBm) and Doppler shift (Hz).
#
# At each epoch we receive signals from 3 candidate satellites. We use
# Mahalanobis distance to determine which candidate is statistically closest
# to our model prediction — accounting for the different scales and correlation
# between signal strength and Doppler shift.
# ─────────────────────────────────────────────────────────────────────────────

# ── Tweak these to change the experiment ─────────────────────────────────────
N_EPOCHS = 20    # number of time steps / signal epochs
SEED     = 42

# Predicted model for our target satellite:
#   feature vector = [signal_strength (dBm), doppler_shift (Hz)]
MODEL_MEAN = np.array([-95.0, 2400.0])

# Model covariance — captures expected variance and correlation between features.
# Signal strength variance: 4 dBm^2 (std ~2 dBm)
# Doppler variance: 10000 Hz^2 (std ~100 Hz)
# Positive correlation: stronger signal tends to come with higher Doppler here
MODEL_COV = np.array([
    [4.0,   80.0],
    [80.0, 10000.0],
])

# True satellite matches the model. The other two are impostors with offsets.
SAT_OFFSETS = {
    "SAT-A (target)":  np.array([0.0,    0.0]),    # matches model
    "SAT-B (impostor)": np.array([-5.0, -800.0]),  # weaker, lower Doppler
    "SAT-C (impostor)": np.array([3.0,   300.0]),  # stronger, higher Doppler
}

# Per-satellite measurement noise (how noisy each satellite's signal is)
SAT_NOISE_COV = {
    "SAT-A (target)":   np.array([[1.0, 0.0], [0.0, 500.0]]),
    "SAT-B (impostor)": np.array([[2.0, 0.0], [0.0, 2000.0]]),
    "SAT-C (impostor)": np.array([[1.5, 0.0], [0.0, 800.0]]),
}
# ─────────────────────────────────────────────────────────────────────────────

rng = np.random.default_rng(SEED)
inv_model_cov = np.linalg.inv(MODEL_COV)

def mahalanobis(x, mean, inv_cov):
    d = x - mean
    return np.sqrt(d @ inv_cov @ d)

# Generate observations and compute distances epoch by epoch
print(f"{'Epoch':<7}", end="")
for name in SAT_OFFSETS:
    print(f"  {name:<22}", end="")
print(f"  {'Winner':<22}")
print("-" * 90)

win_counts = {name: 0 for name in SAT_OFFSETS}

for epoch in range(N_EPOCHS):
    distances = {}
    for name, offset in SAT_OFFSETS.items():
        true_signal = MODEL_MEAN + offset
        noise       = rng.multivariate_normal([0, 0], SAT_NOISE_COV[name])
        observation = true_signal + noise
        distances[name] = mahalanobis(observation, MODEL_MEAN, inv_model_cov)

    winner = min(distances, key=distances.get)
    win_counts[winner] += 1

    print(f"{epoch+1:<7}", end="")
    for name in SAT_OFFSETS:
        marker = " <--" if name == winner else "    "
        print(f"  {distances[name]:>6.2f}{marker:<18}", end="")
    print(f"  {winner}")

print("\n── Summary ──────────────────────────────────────────────────────────────")
for name, count in win_counts.items():
    print(f"  {name:<24}  identified as closest {count:>2}/{N_EPOCHS} epochs")