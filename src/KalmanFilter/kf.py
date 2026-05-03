import numpy as np
import matplotlib.pyplot as plt

# ── Tweak these to change the experiment ─────────────────────────────────────
N_STEPS       = 80      # number of time steps
TRUE_VELOCITY = 1.5     # true constant velocity of the object
SENSOR_NOISE  = 8.0     # std dev of measurement noise (higher = noisier sensor)
PROCESS_NOISE = 0.5     # how much the model trusts constant-velocity assumption
                        # (higher = filter adapts faster but is less smooth)
SEED          = 42
# ─────────────────────────────────────────────────────────────────────────────

rng = np.random.default_rng(SEED)

dt       = 1.0
true_pos = np.cumsum(np.full(N_STEPS, TRUE_VELOCITY * dt))
measurements = true_pos + rng.normal(0, SENSOR_NOISE, N_STEPS)

# ── Kalman matrices ───────────────────────────────────────────────────────────
# State: [position, velocity]
F = np.array([[1, dt], [0, 1]])   # constant-velocity transition
H = np.array([[1,  0]])           # we only observe position
Q = np.eye(2) * PROCESS_NOISE     # process noise covariance
R = np.array([[SENSOR_NOISE**2]]) # measurement noise covariance

x = np.array([[0.0], [0.0]])      # initial state: column vector (2,1)
P = np.eye(2) * 500               # high initial uncertainty

estimates = []
for z_val in measurements:
    z = np.array([[z_val]])       # measurement: column vector (1,1)

    # Predict
    x = F @ x
    P = F @ P @ F.T + Q

    # Update
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x = x + K @ (z - H @ x)
    P = (np.eye(2) - K @ H) @ P

    estimates.append(x[0, 0])     # x is (2,1), so index with [0, 0] not [0]

estimates = np.array(estimates)

# ── Results ───────────────────────────────────────────────────────────────────
meas_err = np.abs(measurements - true_pos).mean()
kalm_err = np.abs(estimates    - true_pos).mean()
print(f"Avg measurement error : {meas_err:.2f}")
print(f"Avg Kalman error      : {kalm_err:.2f}")
print(f"Improvement           : {(1 - kalm_err / meas_err) * 100:.1f}%")

plt.figure(figsize=(11, 4))
plt.plot(true_pos, label="True position", linewidth=2, color="steelblue")
plt.scatter(range(N_STEPS), measurements, label="Noisy measurements",
            s=12, alpha=0.5, color="orange")
plt.plot(estimates, label="Kalman estimate", linewidth=2, color="crimson")
plt.legend()
plt.title(f"1D Kalman filter  |  sensor noise={SENSOR_NOISE}  process noise={PROCESS_NOISE}")
plt.xlabel("Time step")
plt.ylabel("Position")
plt.tight_layout()
plt.savefig("kalman_output.png", dpi=120)
plt.show()
print("Plot saved to kalman_output.png")