"""
Build Dissertation_Overleaf.zip with forward-slash paths.

(Windows PowerShell 5.1's Compress-Archive writes backslash separators,
which Linux unzippers — Overleaf's included — can treat as literal filename
characters, breaking every \\input{chapters/...}.)

Usage: python dissertation/make_overleaf_zip.py
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Dissertation_Overleaf.zip"
INCLUDE = ["main.tex", "references.bib", "frontmatter", "chapters", "sections", "appendices", "figures"]

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for item in INCLUDE:
        p = ROOT / item
        paths = [p] if p.is_file() else sorted(q for q in p.rglob("*") if q.is_file())
        for q in paths:
            z.write(q, q.relative_to(ROOT).as_posix())

with zipfile.ZipFile(OUT) as z:
    names = z.namelist()
bad = [n for n in names if "\\" in n]
print(f"{OUT.name}: {len(names)} entries, {OUT.stat().st_size / 1024:.0f} KB, "
      f"backslash paths: {len(bad)}")
