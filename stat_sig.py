import numpy as np
import random
import json

simulation_file = './outputs/hiring-decisions_meta-llama_200/v11/simans_count.json'
task_file = './outputs/hiring-decisions_meta-llama_200/v11/taskans_count.json'

with open(simulation_file, "r") as f:
    simulation = json.load(f)

with open(task_file, "r") as f:
    task = json.load(f)

# Convert to flat lists
sim_values = np.array([v[0] for v in simulation.values()])
task_values = np.array([v[0] for v in task.values()])

# Define metric: agreement rate (same answer)
def agreement(a, b):
    return np.mean(a == b)

# Bootstrap setup
n = len(sim_values)
bootstrap_diffs = []

for _ in range(1000):
    # Sample 90% with replacement
    idx = np.random.choice(range(n), size=int(0.9 * n), replace=True)
    sim_s = sim_values[idx]
    task_s = task_values[idx]
    diff = agreement(sim_s, task_s)  # could be accuracy difference if you have ground truth
    bootstrap_diffs.append(diff)

# Convert to array
bootstrap_diffs = np.array(bootstrap_diffs)

# Compute 90% confidence interval
lower = np.percentile(bootstrap_diffs, 5)
upper = np.percentile(bootstrap_diffs, 95)

print(f"Bootstrap 90% CI for agreement difference: [{lower:.3f}, {upper:.3f}]")
