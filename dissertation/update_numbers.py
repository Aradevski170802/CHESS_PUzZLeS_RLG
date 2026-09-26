"""
Set the headline PuzzleNet numbers in main.tex from the evaluation output, so that
the chapters cannot drift from what was measured.

    python dissertation/update_numbers.py

Reads eval/neural/puzzlenet_evaluation.json and rewrites the \\newcommand lines for
\\KAPPANN and \\AGREENN.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVAL = ROOT.parent / "eval" / "neural" / "puzzlenet_evaluation.json"

ev = json.loads(EVAL.read_text(encoding="utf-8"))
nn = ev["harness"]["puzzlenet"]
values = {
    "KAPPANN": f"{nn['cohens_kappa']:.2f}",
    "AGREENN": f"{nn['strict_agreement'] * 100:.1f}\\%",
}

main = ROOT / "main.tex"
text = main.read_text(encoding="utf-8")
for name, value in values.items():
    pattern = re.compile(r"\\newcommand\{\\" + name + r"\}\{[^}]*\}")
    assert pattern.search(text), f"macro {name} not declared in main.tex"
    text = pattern.sub(r"\\newcommand{\\" + name + "}{" + value + "}", text)
main.write_text(text, encoding="utf-8")
print("main.tex updated:", values)
