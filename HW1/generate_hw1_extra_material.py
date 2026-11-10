import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
LLM_DIR = ROOT / 'DeepDataMiningLearning' / 'DeepDataMiningLearning' / 'llm'
if str(LLM_DIR) not in sys.path:
    sys.path.insert(0, str(LLM_DIR))

from nanolm.viz import attention_maps, load_run

FIG_DIR = ROOT / 'figures'
FIG_DIR.mkdir(exist_ok=True)


def load_eval_curve(run_dir: Path):
    path = run_dir / 'eval_log.jsonl'
    rows = []
    if not path.exists():
        return rows
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def generate_arch_learning_curves():
    sweep_dir = LLM_DIR / 'outputs' / 'nanolm' / 'sweeps' / 'char_models'
    selected = [
        'rnn_s42',
        'gru_s42',
        'lstm_s42',
        'lstm_512_s42',
        'gpt_textbook_s42',
        'gpt_modern_s42',
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in selected:
        run_dir = sweep_dir / name
        rows = load_eval_curve(run_dir)
        if not rows:
            continue
        xs = [r['train_tokens'] / 1e6 for r in rows]
        ys = [r['bpb'] for r in rows]
        label = name
        if 'rnn' in name:
            color = '#6c757d'
        elif 'gru' in name:
            color = '#C73E1D'
        elif 'lstm_512' in name:
            color = '#F18F01'
        elif 'lstm' in name:
            color = '#2E86AB'
        elif 'gpt_textbook' in name:
            color = '#7F5AF0'
        else:
            color = '#0F766E'
        ax.plot(xs, ys, label=label, color=color, linewidth=2.0)
    ax.set_xlabel('Tokens seen (M)')
    ax.set_ylabel('Validation bpb')
    ax.set_title('Architecture learning curves (same token budget, two seeds)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'architecture_learning_curves.png', dpi=200)
    plt.close(fig)
    print('Saved architecture learning curve:', FIG_DIR / 'architecture_learning_curves.png')


def prompt_attention_probe(text: str, prompt_name: str, run_dir: Path):
    cfg, model, tok, dinfo, _ = load_run(str(run_dir))
    idx = torch.tensor([tok.encode(text)])
    maps = attention_maps(model, idx)
    labels = [repr(c)[1:-1].replace('\\n', '⏎') for c in text]
    L, H = len(maps), maps[0].shape[0]
    fig, axes = plt.subplots(L, H, figsize=(2.2 * H, 2.0 * L), squeeze=False)
    for l in range(L):
        for h in range(H):
            ax = axes[l, h]
            ax.imshow(maps[l][h], cmap='Blues', vmin=0, vmax=1)
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=5, rotation=45)
            ax.set_yticklabels(labels, fontsize=5)
            ax.tick_params(length=0)
            ax.set_title(f'layer {l} head {h}', fontsize=7)
    fig.suptitle(f'Attention probe: {prompt_name}', fontsize=10)
    fig.tight_layout()
    out = FIG_DIR / f'attention_probe_{prompt_name}.png'
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print('Saved attention probe:', out)

    # summary metrics for the head that most strongly attends to previous token and the head that most strongly attends to word starts
    prev_scores = []
    word_scores = []
    for layer in range(L):
        for head in range(H):
            arr = maps[layer][head]
            prev_vals = []
            word_vals = []
            for t in range(1, len(text)):
                prev_vals.append(float(arr[t, t - 1]))
                if text[t - 1] in {' ', '\n', '\t'}:
                    word_vals.append(float(arr[t, t - 1]))
                else:
                    j = t - 1
                    while j > 0 and text[j - 1] not in {' ', '\n', '\t'}:
                        j -= 1
                    word_vals.append(float(arr[t, j]))
            if prev_vals:
                prev_scores.append((sum(prev_vals)/len(prev_vals), layer, head))
            if word_vals:
                word_scores.append((sum(word_vals)/len(word_vals), layer, head))
    best_prev = max(prev_scores, key=lambda x: x[0]) if prev_scores else None
    best_word = max(word_scores, key=lambda x: x[0]) if word_scores else None
    return {
        'prompt': prompt_name,
        'best_prev_head': best_prev,
        'best_word_head': best_word,
    }


def main():
    generate_arch_learning_curves()
    run_dir = LLM_DIR / 'outputs' / 'nanolm' / 'sweeps' / 'char_models' / 'gpt_modern_s1337'
    prompts = [
        ('shakespeare_prev', 'ROMEO:\nBut soft! what light through yonder window breaks?'),
        ('word_boundary', 'The quick brown fox jumps over the lazy dog and runs away.'),
        ('punctuation', 'Alice, Bob, and Carol went home; then they stayed inside all night.'),
    ]
    for slug, text in prompts:
        prompt_attention_probe(text, slug, run_dir)


if __name__ == '__main__':
    main()
