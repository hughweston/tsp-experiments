import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# get gemini token output
gemini_points = []
for entry in os.listdir("tokens"):
    for file in os.listdir(f"tokens/{entry}"):
        name = file[:-4]
        if name[-3:] != "put":
            with open(f"tokens/{entry}/{file}", "r") as f:
                total_token = int(f.readlines()[2].strip().split(": ")[1])
                gemini_points.append((int(entry), total_token))
gx, gy = zip(*gemini_points)

qwen_points = []
for entry in os.listdir("qwen_tokens"):
    for file in os.listdir(f"qwen_tokens/{entry}"):
        name = file[:-4]
        if name[-3:] != "put":
            with open(f"qwen_tokens/{entry}/{file}", "r") as f:
                total_token = int(f.readlines()[0].strip())
                qwen_points.append((int(entry), total_token))
qx, qy = zip(*qwen_points)

gx, gy = np.array(gx), np.array(gy)
qx, qy = np.array(qx), np.array(qy)

gm, gb = np.polyfit(gx, gy, 1)
g_fit_line = gm * gx + gb

g_slope, g_intercept, g_r_value, g_p_value, g_std_err = stats.linregress(gx, gy)
g_r_squared = g_r_value**2

qm, qb = np.polyfit(qx, qy, 1)
q_fit_line = qm * qx + qb

q_slope, q_intercept, q_r_value, q_p_value, q_std_err = stats.linregress(qx, qy)
q_r_squared = q_r_value**2


plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(10, 7))

# 1. Scatter Plots with Labels
plt.scatter(gx, gy, c="blue", label="gemini-2.5-flash")
plt.scatter(qx, qy, c="orange", label="qwen3-vl")

# 2. Line of Best Fit Plots with Labels
# Use the same color as the scatter points
plt.plot(gx, g_fit_line, 
         color='blue', 
         linestyle='--', 
         linewidth=2, 
         label=f'Gemini Fit: y={gm:.2f}x + {gb:.0f}, ($R^2={g_r_squared:.3f}$)')

plt.plot(qx, q_fit_line, 
         color='orange', 
         linestyle='-', 
         linewidth=2, 
         label=f'Qwen Fit: y={qm:.2f}x + {qb:.0f}, ($R^2={q_r_squared:.3f}$)')


plt.xticks([10, 15, 20, 25, 30])
plt.xlabel("Number of Points (TSP Problem Size)")
plt.ylabel("Number of Tokens Used")
plt.title("Size of TSP Problem vs. Number of Tokens Used")

plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("tokens_used.svg")