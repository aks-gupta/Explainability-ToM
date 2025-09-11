import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Create DataFrame for Harmful Requests
harmful_requests_data = {
    'Category': ['WITH EXPL', 'NO EXPL', 'Concise', 'Detailed', 'Toxic', 'Non-Toxic'],
    'Score': [81.1, 72.2, 64.4, 75.6, 89.4, 77.2]
}

df_harmful = pd.DataFrame(harmful_requests_data)

# Print table in a format that can be copied to PowerPoint
print("Harmful Requests - Baseline Evaluations")
print(df_harmful.to_string(index=False))

# Save data to Excel for PowerPoint use
df_harmful.to_excel('harmful_requests_baseline.xlsx', index=False)

# Set a consistent style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000', '#5B9BD5', '#70AD47']  # PowerPoint-like colors

# GRAPH 1: WITH EXPL vs NO EXPL comparison
plt.figure(figsize=(8, 6))
# Filter data for just these two categories
expl_data = df_harmful[df_harmful['Category'].isin(['WITH EXPL', 'NO EXPL'])]

# Create bar chart
ax1 = sns.barplot(x='Category', y='Score', data=expl_data, palette=colors[:2])

# Add data labels on top of bars
for i, v in enumerate(expl_data['Score']):
    ax1.text(i, v + 1, f"{v}%", ha='center', fontweight='bold')

# Style the chart
plt.title('Explanation vs No Explanation Comparison', fontsize=16, fontweight='bold')
plt.xlabel('Condition', fontsize=14)
plt.ylabel('Score (%)', fontsize=14)
plt.ylim(0, 100)  # Set y-axis from 0 to 100 for percentage
plt.xticks(fontsize=12)
plt.tight_layout()

# Add a text annotation showing the difference
diff = expl_data['Score'].iloc[0] - expl_data['Score'].iloc[1]
plt.annotate(f'Difference: {diff:.1f}%', 
             xy=(0.5, 40), 
             xytext=(0.5, 40),
             textcoords='axes fraction',
             ha='center', 
             va='center',
             bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3),
             fontsize=14)

# Save the figure
plt.savefig('harmful_requests_expl_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# GRAPH 2: Comparing the other four categories (Toxic, Non-Toxic, Concise, Detailed)
plt.figure(figsize=(10, 6))
# Filter data for just these categories
other_data = df_harmful[df_harmful['Category'].isin(['Concise', 'Detailed', 'Toxic', 'Non-Toxic'])]

# Create bar chart
ax2 = sns.barplot(x='Category', y='Score', data=other_data, palette=colors[2:])

# Add data labels on top of bars
for i, v in enumerate(other_data['Score']):
    ax2.text(i, v + 1, f"{v}%", ha='center', fontweight='bold')

# Style the chart
plt.title('Response Type Comparison', fontsize=16, fontweight='bold')
plt.xlabel('Response Type', fontsize=14)
plt.ylabel('Score (%)', fontsize=14)
plt.ylim(0, 100)  # Set y-axis from 0 to 100 for percentage
plt.tight_layout()

# Save the figure
plt.savefig('harmful_requests_types_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Optional: Create a single figure with both graphs for easy comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# First subplot - EXPL comparison
sns.barplot(x='Category', y='Score', data=expl_data, palette=colors[:2], ax=ax1)
for i, v in enumerate(expl_data['Score']):
    ax1.text(i, v + 1, f"{v}%", ha='center', fontweight='bold')
ax1.set_title('Explanation vs No Explanation', fontsize=14, fontweight='bold')
ax1.set_xlabel('Condition', fontsize=12)
ax1.set_ylabel('Score (%)', fontsize=12)
ax1.set_ylim(0, 100)

# Second subplot - Other categories
sns.barplot(x='Category', y='Score', data=other_data, palette=colors[2:], ax=ax2)
for i, v in enumerate(other_data['Score']):
    ax2.text(i, v + 1, f"{v}%", ha='center', fontweight='bold')
ax2.set_title('Response Type Comparison', fontsize=14, fontweight='bold')
ax2.set_xlabel('Response Type', fontsize=12)
ax2.set_ylabel('Score (%)', fontsize=12)
ax2.set_ylim(0, 100)

plt.tight_layout()
plt.savefig('harmful_requests_combined_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

"""
Instructions for use:
1. Run this code to generate three visualization files:
   - harmful_requests_expl_comparison.png (EXPL vs NO EXPL)
   - harmful_requests_types_comparison.png (Toxic, Non-Toxic, Concise, Detailed)
   - harmful_requests_combined_comparison.png (Both graphs side by side)

2. The tables are printed in a format that can be copied directly to PowerPoint

3. An Excel file is also created that can be imported into PowerPoint

To add BBQ and Hiring Decisions datasets, modify this code by creating similar DataFrames.
"""