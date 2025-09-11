import matplotlib.pyplot as plt

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Bar chart data
categories = ['Non-toxic', 'Toxic', 'Detailed', 'Concise', 'COT']
values = [87.8, 90.4, 87.6, 88.3, 90.8]

# Horizontal bar chart
ax1.barh(categories, values, color='red')
ax1.set_title('Precision')
ax1.set_xlim(0, 100)

# Line chart data
shots = [0, 1, 3, 5]
performance = [63.6, 87.8, 90.8, 90.4]

# Line chart
ax2.plot(shots, performance, 'o-', color='red')
ax2.set_title('Precision')
ax2.set_xlabel('Shots')

plt.tight_layout()
plt.show()