import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json

# Try Qt5Agg, fallback to Agg if Qt binding is unavailable
try:
    matplotlib.use('Qt5Agg')
except ImportError:
    print("Qt5Agg backend unavailable, falling back to Agg (non-interactive).")
    matplotlib.use('Agg')

plt.rc('font', family='Times New Roman', size=18)
plt.rcParams['axes.unicode_minus'] = False

# Load data from results.json
with open('results.json', 'r') as f:
    data = json.load(f)

# Define methods and models
methods = ["IKE", "GRACE", "WISE", "LoRA"]
models = ["llavaov", "qwen2vl", "huatuo"]
labels = methods + [methods[0]]  # Close the radar chart loop

# Extract data for each model and metric
time_data = {model: [] for model in models}
memory_data = {model: [] for model in models}

for method in methods:
    for entry in data:
        method_name = entry['method_name']
        if method_name.startswith(method):
            model = method_name.split('_')[1]
            time_data[model].append(entry['avg_time_per_sample'])
            memory_data[model].append(entry['peak_gpu_memory'])

# Normalize data to 0-1 for radar chart
def normalize(arr):
    arr = np.array(arr)
    return (arr - arr.min()) / (arr.max() - arr.min()) if arr.max() != arr.min() else arr / arr.max()

for model in models:
    time_data[model] = normalize(time_data[model]).tolist()
    memory_data[model] = normalize(memory_data[model]).tolist()
    time_data[model].append(time_data[model][0])  # Close the loop
    memory_data[model].append(memory_data[model][0])  # Close the loop

# Set up radar chart
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=True)
yticks = [0, 0.25, 0.5, 0.75, 1]

fig, axs = plt.subplots(1, 2, subplot_kw=dict(polar=True), figsize=(10, 5))

for idx, ax in enumerate(axs):
    ax.set_theta_offset(np.pi / 4)
    ax.set_theta_direction(-1)

    if idx == 0:
        title = "Avg Time per Sample (Normalized)"
        plot_data = [time_data[model] for model in models]
    else:
        title = "Peak GPU Memory (Normalized)"
        plot_data = [memory_data[model] for model in models]

    ax.set_title(title, fontsize=18, fontweight='bold', y=1.1)

    # Plot data lines and fill
    colors = ['orangered', 'mediumslateblue', 'skyblue']
    for data, color, model in zip(plot_data, colors, models):
        ax.plot(angles, data, color=color, linewidth=2, zorder=1, label=model)
        ax.fill(angles, data, color=color, alpha=0.25, zorder=1)

    # Remove default angle labels
    ax.set_thetagrids(angles * 180 / np.pi, [''] * len(labels))

    # Manually draw radar chart labels
    for angle, lab in zip(angles, labels):
        ax.text(angle, 1.25, lab, ha='center', va='center',
                fontsize=18, family='Times New Roman', zorder=5)

    # Set polar radius range and ticks
    ax.set_ylim(0, 1.05)
    ax.set_yticks(yticks)
    ax.set_yticklabels([''] * len(yticks))  # No default labels
    ax.set_rlabel_position(90)

    # Manually add r labels
    for y in yticks:
        ax.text(np.pi / 2, y, f"{y:.2f}", ha='center', va='center',
                fontsize=12, zorder=10, color='black')

# Legend
fig.legend(['LLaVA-Onevision', 'QWen2-VL', 'HuaTuoGPT-7B'],
           loc='lower center', bbox_to_anchor=(0.5, -0.12),
           ncol=3, frameon=False, fontsize=18)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("time_memory_radar.pdf", format='pdf', bbox_inches='tight')
plt.show()