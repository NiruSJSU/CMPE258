from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

root = Path(__file__).resolve().parent
fig_dir = root / 'figures'
fig_dir.mkdir(exist_ok=True)

arch = pd.read_csv(root / 'nanolm_architecture_plot_data.csv')
context = pd.read_csv(root / 'nanolm_context_plot_data.csv')

# Architecture bar chart (lower is better)
bar_data = arch.sort_values('val_bpb').copy()
fig, ax = plt.subplots(figsize=(10, 6))
colors = []
for model in bar_data['run_name']:
    if 'gpt' in model:
        colors.append('#2E86AB')
    elif 'lstm' in model:
        colors.append('#F18F01')
    elif 'gru' in model:
        colors.append('#C73E1D')
    else:
        colors.append('#6C757D')

ax.barh(bar_data['run_name'], bar_data['val_bpb'], color=colors)
ax.invert_yaxis()
ax.set_xlabel('Validation bits-per-byte (bpb)')
ax.set_title('Architecture sweep: validation bpb (lower is better)')
ax.grid(axis='x', linestyle='--', alpha=0.3)
for i, v in enumerate(bar_data['val_bpb']):
    ax.text(v + 0.003, i, f'{v:.4f}', va='center', fontsize=8)
plt.tight_layout()
fig.savefig(fig_dir / 'architecture_sweep.png', dpi=200)
plt.close(fig)

# Context study line chart
fig, ax = plt.subplots(figsize=(8, 6))
for model, group in context.groupby('model_family'):
    color = '#2E86AB' if model == 'lstm' else '#C73E1D'
    marker = 'o' if model == 'lstm' else 's'
    xs = group['seq_len'].astype(int).tolist()
    ys = group['val_bpb'].tolist()
    ax.plot(xs, ys, label=model.upper(), marker=marker, linewidth=2.5, markersize=7, color=color)
    for x, y in zip(xs, ys):
        ax.annotate(f'{y:.4f}', (x, y), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=8)
ax.set_xlabel('Context length (tokens)')
ax.set_ylabel('Validation bpb')
ax.set_title('Context-length study: LSTM vs modern GPT')
ax.grid(True, linestyle='--', alpha=0.3)
ax.legend()
plt.tight_layout()
fig.savefig(fig_dir / 'context_study.png', dpi=200)
plt.close(fig)

# Attention summary chart
fig, ax = plt.subplots(figsize=(7, 5))
heads = ['Previous-token', 'Word-start']
vals = [0.5761, 0.2932]
colors = ['#2E86AB', '#F18F01']
ax.bar(heads, vals, color=colors)
ax.set_ylabel('Average attention weight')
ax.set_title('Attention probe summary')
for i, v in enumerate(vals):
    ax.text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom', fontsize=10)
ax.set_ylim(0, 0.7)
ax.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
fig.savefig(fig_dir / 'attention_summary.png', dpi=200)
plt.close(fig)

print('Generated figures in', fig_dir)
for name in ['architecture_sweep.png', 'context_study.png', 'attention_summary.png']:
    print((fig_dir / name).resolve())
