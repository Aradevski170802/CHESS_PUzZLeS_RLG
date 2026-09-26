"""
Static checks for the dissertation LaTeX project (no TeX installation needed).

Catches the errors that most often break an Overleaf build:
  * \\input files that do not exist
  * unbalanced braces and mis-nested \\begin/\\end environments
  * \\cite keys missing from references.bib; \\ref/\\eqref labels never defined
  * \\includegraphics files that do not exist
  * characters pdfLaTeX's utf8 inputenc cannot typeset (e.g. θ, ≈, −)
  * unescaped % (would silently comment out the rest of a line), _ and #
    outside math mode

Usage: python dissertation/check_latex.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Latin-1 letters and typographic punctuation that T1 + utf8 inputenc handle.
ALLOWED_NON_ASCII = set("éèêëàâäáíìîïóòôöúùûüçñÉÈÀÖÜßœæ–—‘’“”…")


def strip_comments(line: str) -> str:
    return re.sub(r"(?<!\\)%.*$", "", line)


def files() -> list[Path]:
    """main.tex and every file it pulls in, following \\input recursively
    (chapters include their own sections)."""
    out, seen = [], set()
    queue = [ROOT / "main.tex"]
    while queue:
        f = queue.pop(0)
        if f in seen:
            continue
        seen.add(f)
        out.append(f)
        if not f.exists():
            continue
        for m in re.finditer(r"\\input\{([^}]+)\}", f.read_text("utf-8")):
            name = m.group(1)
            queue.append(ROOT / (name + ("" if name.endswith(".tex") else ".tex")))
    return out


def check() -> int:
    problems: list[str] = []
    labels, refs, cites = set(), [], []
    for f in files():
        if not f.exists():
            problems.append(f"missing \\input file: {f.relative_to(ROOT)}")
            continue
        text = f.read_text("utf-8")
        rel = f.relative_to(ROOT)
        # --- braces and environments (comments stripped, verbatim skipped)
        depth, env_stack, in_verb = 0, [], False
        for n, raw in enumerate(text.splitlines(), 1):
            if "\\begin{verbatim}" in raw:
                in_verb = True
                continue
            if "\\end{verbatim}" in raw:
                in_verb = False
                continue
            if in_verb:
                continue
            line = strip_comments(raw)
            clean = re.sub(r"\\[{}]", "", line)
            depth += clean.count("{") - clean.count("}")
            for kind, env in re.findall(r"\\(begin|end)\{([^}]+)\}", line):
                if kind == "begin":
                    env_stack.append((env, n))
                elif not env_stack or env_stack[-1][0] != env:
                    problems.append(f"{rel}:{n}: \\end{{{env}}} does not match "
                                    f"{env_stack[-1] if env_stack else 'nothing open'}")
                else:
                    env_stack.pop()
            # --- characters pdfLaTeX cannot typeset
            for ch in set(raw):
                if ord(ch) > 127 and ch not in ALLOWED_NON_ASCII:
                    problems.append(f"{rel}:{n}: unsupported character {ch!r} (U+{ord(ch):04X})")
            if re.search(r"(?<!\\)\d%", raw):
                problems.append(f"{rel}:{n}: unescaped % after a number (comments out the rest of the line)")
        # --- bare _ and # outside math, tracked across lines. Removed spans are
        # replaced by their own newlines so line numbers stay correct.
        def blank(m: re.Match) -> str:
            return "\n" * m.group(0).count("\n")
        prose = "\n".join(strip_comments(l) for l in text.splitlines())
        for pat in (r"\\begin\{verbatim\}.*?\\end\{verbatim\}",
                    r"\\begin\{(equation|align|tikzpicture)\*?\}.*?\\end\{\1\*?\}",
                    r"\\\[.*?\\\]", r"\$\$.*?\$\$", r"(?<!\\)\$.*?(?<!\\)\$",
                    r"\\(code|texttt|url|label|ref|eqref|cite|input|includegraphics|href)(\[[^\]]*\])?\{[^}]*\}",
                    r"#\d"):
            prose = re.sub(pat, blank, prose, flags=re.S)
        for n, line in enumerate(prose.splitlines(), 1):
            if re.search(r"(?<!\\)_", line):
                problems.append(f"{rel}:{n}: bare _ outside math: {line.strip()[:80]}")
            if re.search(r"(?<!\\)#", line):
                problems.append(f"{rel}:{n}: bare # outside a macro definition: {line.strip()[:80]}")
        if depth != 0:
            problems.append(f"{rel}: braces unbalanced by {depth:+d}")
        for env, n in env_stack:
            problems.append(f"{rel}:{n}: \\begin{{{env}}} never closed")
        body = "\n".join(strip_comments(l) for l in text.splitlines())
        labels |= set(re.findall(r"\\label\{([^}]+)\}", body))
        refs += re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", body)
        for grp in re.findall(r"\\cite\{([^}]+)\}", body):
            cites += [c.strip() for c in grp.split(",")]
        for img in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body):
            if not (ROOT / img).exists():
                problems.append(f"{rel}: missing figure {img}")

    bib = (ROOT / "references.bib").read_text("utf-8")
    bib_keys = set(re.findall(r"@\w+\{([^,\s]+),", bib))
    for c in sorted(set(cites) - bib_keys):
        problems.append(f"\\cite{{{c}}} not in references.bib")
    for r in sorted(set(refs) - labels):
        problems.append(f"\\ref{{{r}}} has no \\label")
    unused = sorted(bib_keys - set(cites))

    print(f"files checked: {len(files())} | labels: {len(labels)} | refs: {len(refs)} | "
          f"citations: {len(set(cites))} unique | bib entries: {len(bib_keys)}")
    if unused:
        print("bib entries never cited (harmless, BibTeX omits them):", ", ".join(unused))
    for p in problems:
        print("PROBLEM:", p)
    print("OK — no problems found" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(check())
