import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# --- same setup as index.py ---
H, M, C = 10, 10, 10
epsilon = 0.001

goal      = random.uniform(-1, 1)
tolerance = 0.0075
t         = math.tanh(goal)

if t >= 0:
    y = epsilon + (1 - epsilon) * abs(t)
else:
    y = -epsilon - (1 - epsilon) * abs(t)

q_H, q_M, q_C = float(H), float(M), float(C)
x      = 0
target = y

explore_rate = 0.9
decay        = 0.95
min_explore  = 0.05
lr           = 0.15
max_steps    = 500

# --- run RL and record history ---
history = {
    'step':    [0],
    'x':       [0.0],
    'dist':    [abs(target)],
    'action':  ['START'],
    'q_H':     [q_H],
    'q_M':     [q_M],
    'q_C':     [q_C],
    'explore': [explore_rate],
}

step = 0
while abs(target - x) > tolerance and step < max_steps:
    distance = target - x
    subrange  = abs(distance)
    available = ['H', 'C'] if x == 0 else ['H', 'M', 'C']

    if random.random() < explore_rate:
        chosen = random.choice(available)
    else:
        scores = {'H': q_H, 'M': q_M, 'C': q_C}
        chosen = max(available, key=lambda a: scores[a])

    if chosen == 'H':
        step_val = -subrange * 0.5
    elif chosen == 'C':
        step_val =  subrange * 0.5
    else:
        step_val =  distance * 0.3

    prev_dist = abs(target - x)
    x         = max(-1.0, min(1.0, x + step_val))
    new_dist  = abs(target - x)
    reward    = prev_dist - new_dist

    if chosen == 'H':
        q_H += lr * (reward - q_H)
    elif chosen == 'C':
        q_C += lr * (reward - q_C)
    else:
        q_M += lr * (reward - q_M)

    explore_rate = max(min_explore, explore_rate * decay)
    step        += 1

    history['step'].append(step)
    history['x'].append(x)
    history['dist'].append(new_dist)
    history['action'].append(chosen)
    history['q_H'].append(q_H)
    history['q_M'].append(q_M)
    history['q_C'].append(q_C)
    history['explore'].append(explore_rate)

# --- unpack ---
steps   = history['step']
xs      = history['x']
actions = history['action']
qH      = history['q_H']
qM      = history['q_M']
qC      = history['q_C']

ACTION_COLOR = {'H': '#e63946', 'M': '#2a9d8f', 'C': '#457b9d', 'START': '#333333'}
colors = [ACTION_COLOR[a] for a in actions]

side = 'HOT side -> -1' if target < 0 else 'COLD side -> +1'
converged = abs(target - x) <= tolerance
status = f"Converged in {step} steps" if converged else f"Max steps ({step})"

fig = plt.figure(figsize=(15, 6))
fig.suptitle(
    f"RL Temperature Adjustment  |  target = {target:+.4f} ({side})  |  {status}",
    fontsize=12, fontweight='bold'
)

# ── 2D: convergence trajectory ──────────────────────────────────────────────
ax1 = fig.add_subplot(1, 2, 1)

for i in range(1, len(steps)):
    ax1.plot(steps[i-1:i+1], xs[i-1:i+1], color=colors[i], linewidth=1.8, alpha=0.85)

ax1.scatter(steps, xs, c=colors, s=35, zorder=5, edgecolors='white', linewidths=0.4)

# mark start and end
ax1.scatter([steps[0]],  [xs[0]],  c='black',  s=80, zorder=6, marker='o', label='start (x=0)')
ax1.scatter([steps[-1]], [xs[-1]], c='purple', s=80, zorder=6, marker='D', label='final x')

# target + tolerance band
ax1.axhline(target,              color='purple', linestyle='--', linewidth=1.6, alpha=0.9)
ax1.axhspan(target - tolerance, target + tolerance, color='purple', alpha=0.08)
ax1.axhline(0, color='gray', linestyle='-', linewidth=0.6, alpha=0.35)

ax1.set_xlabel('Step', fontsize=10)
ax1.set_ylabel('Temperature  x  (hot=-1 … cold=+1)', fontsize=10)
ax1.set_title('2D  —  Convergence Trajectory', fontsize=11)
ax1.set_ylim(-1.15, 1.15)
ax1.set_xlim(-0.5, max(steps) + 0.5)

legend_elems = [
    Patch(fc='#e63946', label='H  action (hot, toward -1)'),
    Patch(fc='#2a9d8f', label='M  action (main, fine-tune)'),
    Patch(fc='#457b9d', label='C  action (cold, toward +1)'),
    Line2D([0], [0], color='purple', ls='--', lw=1.5, label=f'target {target:+.4f}'),
    Patch(fc='purple',  alpha=0.15,  label=f'tolerance ±{tolerance}'),
    Line2D([0], [0], color='gray',   ls='-',  lw=0.8, label='zero (M origin)'),
]
ax1.legend(handles=legend_elems, fontsize=8, loc='best')
ax1.grid(True, alpha=0.2)

# ── 3D: Q-value vector space ─────────────────────────────────────────────────
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

for i in range(1, len(steps)):
    ax2.plot(
        qH[i-1:i+1], qC[i-1:i+1], qM[i-1:i+1],
        color=colors[i], linewidth=1.6, alpha=0.8
    )

ax2.scatter(qH, qC, qM, c=colors, s=20, zorder=5)
ax2.scatter([qH[0]],  [qC[0]],  [qM[0]],  c='black',  s=90, marker='*', zorder=10, label='start')
ax2.scatter([qH[-1]], [qC[-1]], [qM[-1]], c='purple', s=70, marker='D', zorder=10, label='end')

ax2.set_xlabel('Q_H  (hot)', fontsize=9)
ax2.set_ylabel('Q_C  (cold)', fontsize=9)
ax2.set_zlabel('Q_M  (main)', fontsize=9)
ax2.set_title('3D  —  Q-value Vector Space', fontsize=11)

legend_elems_3d = [
    Line2D([0], [0], color='#e63946', lw=2, label='H moves'),
    Line2D([0], [0], color='#2a9d8f', lw=2, label='M moves'),
    Line2D([0], [0], color='#457b9d', lw=2, label='C moves'),
    Line2D([0], [0], color='black',   lw=0, marker='*', markersize=9, label='start'),
    Line2D([0], [0], color='purple',  lw=0, marker='D', markersize=7, label='end'),
]
ax2.legend(handles=legend_elems_3d, fontsize=8, loc='upper left')

plt.tight_layout()
plt.show()