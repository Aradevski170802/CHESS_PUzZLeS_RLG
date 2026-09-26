"""
Rewrite the ablation table in sections/puzzlenet_results2.tex from the measured
numbers in eval/neural/puzzlenet_evaluation.json, so the table cannot drift from
the experiments.

    python dissertation/build_ablation_table.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVAL = json.loads((ROOT.parent / "eval" / "neural" / "puzzlenet_evaluation.json")
                  .read_text(encoding="utf-8"))
TRAIN = ROOT.parent / "eval" / "neural" / "training"

ROWS = [
    ("abl_full_1m", "Reference: 512--256 trunk, all features, all heads", True),
    (None, None, None),
    ("abl_linear", "No hidden layers (logistic / linear regression)", False),
    ("abl_raw", "Raw boards and moves only", False),
    ("abl_engineered", "Hand-built relations only (no boards, no moves)", False),
    ("abl_wide", "Wider trunk: 1024--512", False),
    (None, None, None),
    ("abl_cat_only", "Category head alone (no multi-task)", False),
    ("abl_rating_only", "Difficulty head alone (no multi-task)", False),
    ("abl_beta0", "$\\beta = 0$ (plain noise-aware likelihood)", False),
    ("abl_beta1", "$\\beta = 1$ (squared-error gradient for the mean)", False),
    (None, None, None),
    ("lc_30k", "30{,}000 training puzzles", False),
    ("lc_100k", "100{,}000", False),
    ("lc_300k", "300{,}000", False),
    ("abl_full_1m", "1{,}000{,}000", False),
    ("puzzlenet", "\\textbf{5{,}526{,}626 (the production model)}", "bold"),
]


def cell(name: str, key: str) -> str:
    if name == "puzzlenet":
        block = EVAL["test"]["puzzlenet"]
        heads = ("cat", "theme", "rating")
    else:
        block = EVAL["models"][name]
        heads = block["setup"]["heads"]
    if key in ("cohens_kappa", "macro_f1"):
        if "cat" not in heads:
            return "---"
        return f"{block['cat'][key]:.3f}"
    if key == "rmse":
        if "rating" not in heads:
            return "---"
        return f"{block['rating']['rmse']:.0f}"
    if key == "seconds":
        meta = json.loads((TRAIN / f"{name}.json").read_text(encoding="utf-8"))["meta"]
        return f"{meta['train_seconds']}"
    raise KeyError(key)


lines = []
for name, label, bold in ROWS:
    if name is None:
        lines.append("\\midrule")
        continue
    values = [cell(name, k) for k in ("cohens_kappa", "macro_f1", "rmse", "seconds")]
    if bold == "bold":
        values = [f"\\textbf{{{v}}}" if v != "---" else v for v in values]
    lines.append(f"{label} & " + " & ".join(values) + " \\\\")
body = "\n".join(lines)

p = ROOT / "sections" / "puzzlenet_results2.tex"
s = p.read_text(encoding="utf-8")
pattern = re.compile(r"(\\textbf\{Variant\} & \\textbf\{\$\\kappa\$\} & \\textbf\{Macro-\$F_1\$\} & "
                     r"\\textbf\{RMSE\} & \\textbf\{Train \(s\)\} \\\\\n\\midrule\n)(.*?)(\n\\bottomrule)",
                     re.S)
assert pattern.search(s), "ablation table not found"
s = pattern.sub(lambda m: m.group(1) + body + m.group(3), s)
p.write_text(s, encoding="utf-8")
print("ablation table rebuilt from eval/neural/puzzlenet_evaluation.json")
print(body)
