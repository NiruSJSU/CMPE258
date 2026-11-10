import csv
import json
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

ROOT = Path(r"C:\Users\malad\Desktop\SJSU_Work\CMPE-258\DeepDataMiningLearning\DeepDataMiningLearning\llm\outputs\nanolm")
OUT_DIR = Path(r"C:\Users\malad\Desktop\SJSU_Work\CMPE-258")


def read_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def infer_run_info(run_dir: Path):
    info = {
        "run_name": run_dir.name,
        "model": "unknown",
        "seq_len": "",
        "seed": "",
        "dataset": "",
        "tokenizer": "",
        "final_val_bpb": "",
        "final_val_loss": "",
        "final_val_ppl": "",
    }

    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        info["model"] = m.get("arch", info["model"])
        info["seed"] = m.get("seed", info["seed"])
        info["dataset"] = m.get("dataset", info["dataset"])
        info["tokenizer"] = m.get("tokenizer", info["tokenizer"])
        info["final_val_bpb"] = m.get("val_bpb", info["final_val_bpb"])
        info["final_val_loss"] = m.get("val_loss", info["final_val_loss"])
        info["final_val_ppl"] = m.get("val_ppl", info["final_val_ppl"])

    config_path = run_dir / "config.yaml"
    if config_path.exists() and yaml is not None:
        try:
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            if "train" in cfg and isinstance(cfg["train"], dict):
                seq_len = cfg["train"].get("seq_len")
                if seq_len is not None:
                    info["seq_len"] = seq_len
            if "model" in cfg and isinstance(cfg["model"], dict):
                arch = cfg["model"].get("arch")
                if arch is not None and info["model"] == "unknown":
                    info["model"] = arch
        except Exception:
            pass

    if not info["seq_len"]:
        for pattern in ("lstm_ctx_(\\d+)", "gpt_ctx_(\\d+)"):
            import re
            m = re.search(pattern, run_dir.name)
            if m:
                info["seq_len"] = m.group(1)
                break

    if not info["seed"]:
        import re
        m = re.search(r"s(\d+)$", run_dir.name)
        if m:
            info["seed"] = m.group(1)

    return info


def write_csv(path: Path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


train_rows = []
eval_rows = []
summary_rows = []

for metrics_file in sorted(ROOT.rglob("metrics.json")):
    run_dir = metrics_file.parent
    info = infer_run_info(run_dir)
    summary_rows.append({
        "run_name": info["run_name"],
        "model": info["model"],
        "seq_len": info["seq_len"],
        "seed": info["seed"],
        "dataset": info["dataset"],
        "tokenizer": info["tokenizer"],
        "final_val_bpb": info["final_val_bpb"],
        "final_val_loss": info["final_val_loss"],
        "final_val_ppl": info["final_val_ppl"],
    })

    train_file = run_dir / "train_log.jsonl"
    for record in read_jsonl(train_file):
        row = {"run_name": info["run_name"], "model": info["model"], "seq_len": info["seq_len"], "seed": info["seed"], "dataset": info["dataset"], "tokenizer": info["tokenizer"]}
        row.update(record)
        train_rows.append(row)

    eval_file = run_dir / "eval_log.jsonl"
    for record in read_jsonl(eval_file):
        row = {"run_name": info["run_name"], "model": info["model"], "seq_len": info["seq_len"], "seed": info["seed"], "dataset": info["dataset"], "tokenizer": info["tokenizer"]}
        row.update(record)
        eval_rows.append(row)

train_fields = [
    "run_name", "model", "seq_len", "seed", "dataset", "tokenizer",
    "step", "loss", "lr", "time", "tok_per_s"
]

eval_fields = [
    "run_name", "model", "seq_len", "seed", "dataset", "tokenizer",
    "step", "train_tokens", "time", "loss", "bpb", "ppl", "acc", "tokens", "bytes", "bytes_per_token"
]

summary_fields = [
    "run_name", "model", "seq_len", "seed", "dataset", "tokenizer",
    "final_val_bpb", "final_val_loss", "final_val_ppl"
]

write_csv(OUT_DIR / "nanolm_train_curve_data.csv", train_fields, train_rows)
write_csv(OUT_DIR / "nanolm_eval_curve_data.csv", eval_fields, eval_rows)
write_csv(OUT_DIR / "nanolm_run_summary.csv", summary_fields, summary_rows)

print(f"Wrote {len(train_rows)} train rows to {OUT_DIR / 'nanolm_train_curve_data.csv'}")
print(f"Wrote {len(eval_rows)} eval rows to {OUT_DIR / 'nanolm_eval_curve_data.csv'}")
print(f"Wrote {len(summary_rows)} run summaries to {OUT_DIR / 'nanolm_run_summary.csv'}")
