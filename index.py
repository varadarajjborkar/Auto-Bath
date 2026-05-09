import random
import math

H, M, C = 10, 10, 10
buck = [H, M, C]

epsilon = 0.001

# setting up initials
goal = random.uniform(-1, 1)
tolerance = 0.0075
t = math.tanh(goal)

# avoid zero part
if t >= 0:
    y = epsilon + (1 - epsilon)*abs(t)
else:
    y = -epsilon - (1 - epsilon)*abs(t)

# Q-values for each action, seeded from initial buck weights
q_H, q_M, q_C = float(H), float(M), float(C)

# start at M (neutral/main)
x = 0

# goal defined
target = y

# exploration schedule
explore_rate = 0.9
decay = 0.95
min_explore = 0.05
lr = 0.15

step = 0
max_steps = 500

print(f"Goal : {target:+.6f}  ({'HOT side: -1' if target < 0 else 'COLD side: +1'})")
print(f"Start: x=0, Tolerance: {tolerance}\n")

while abs(target - x) > tolerance and step < max_steps:
    distance = target - x
    subrange = abs(distance)

    # avoid zero: returning to 0 via M makes no sense when already at 0
    available = ['H', 'C'] if x == 0 else ['H', 'M', 'C']

    # epsilon-greedy: explore or exploit
    if random.random() < explore_rate:
        chosen = random.choice(available)
    else:
        scores = {'H': q_H, 'M': q_M, 'C': q_C}
        chosen = max(available, key=lambda a: scores[a])

    # range → subrange → vector step
    #   H: hot direction  → push toward -1  (half the remaining subrange)
    #   C: cold direction → push toward +1  (half the remaining subrange)
    #   M: main fine-tune → gentle nudge toward target (30 % of distance)
    if chosen == 'H':
        step_val = -subrange * 0.5
    elif chosen == 'C':
        step_val = subrange * 0.5
    else:
        step_val = distance * 0.3

    prev_dist = abs(target - x)
    x = max(-1.0, min(1.0, x + step_val))
    new_dist = abs(target - x)

    # reward: "how much closer"
    reward = prev_dist - new_dist

    if chosen == 'H':
        q_H += lr * (reward - q_H)
    elif chosen == 'C':
        q_C += lr * (reward - q_C)
    else:
        q_M += lr * (reward - q_M)

    explore_rate = max(min_explore, explore_rate * decay)

    print(
        f"step {step+1:3d} | {chosen} | x={x:+.6f} | dist={new_dist:.6f} "
        f"| explore={explore_rate:.3f} | Q[H:{q_H:.2f} M:{q_M:.2f} C:{q_C:.2f}]"
    )
    step += 1

print("")

if abs(target - x) <= tolerance:
    print(f"Converged -> x={x:+.6f}  target={target:+.6f}  ({step} steps)")
else:
    print(f"Max steps hit. x={x:+.6f}  target={target:+.6f}  dist={abs(target-x):.6f}")