"""
Figure: the first rows of the Lichess puzzle database, with all of its columns.

Reads the raw CSV (DataSets/lichess_db_puzzle.csv) and draws the first N_ROWS
puzzles as a table, the way pandas shows a DataFrame. Long values (FEN, moves,
themes, URLs) are wrapped, not cut, so every value is shown in full; only the
"https://" prefix of the game URL is left out.

Usage: python -m scripts.report_figures.make_dataset_sample
Output: report/figures/fig4_0_lichess_sample.png (and .svg)
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "DataSets" / "lichess_db_puzzle.csv"
OUT = ROOT / "report" / "figures" / "fig4_0_lichess_sample"
N_ROWS = 5

# column: (header shown, wrap width in characters)
COLUMNS = {
    "PuzzleId":        ("PuzzleId", 8),
    "FEN":             ("FEN", 18),
    "Moves":           ("Moves", 10),
    "Rating":          ("Rating", 6),
    "RatingDeviation": ("Rating\nDeviation", 9),
    "Popularity":      ("Popularity", 10),
    "NbPlays":         ("NbPlays", 7),
    "Themes":          ("Themes", 12),
    "GameUrl":         ("GameUrl", 14),
    "OpeningTags":     ("OpeningTags", 13),
}


def wrap(value, width: int) -> str:
    """Wrap at spaces, and after '/' or '_' inside long tokens (FEN ranks, URL parts,
    opening names), so no line is cut in the middle of a word."""
    if pd.isna(value):
        return "NaN"
    text = str(value).replace("https://", "")
    lines, cur = [], ""
    for piece in re.split(r"(?<=[/_ ])", text):
        if cur and len((cur + piece).rstrip()) > width:
            lines.append(cur.rstrip())
            cur = ""
        cur += piece
    if cur.strip():
        lines.append(cur.rstrip())
    return "\n".join(lines)


def main() -> None:
    df = pd.read_csv(CSV, nrows=N_ROWS)
    assert list(df.columns) == list(COLUMNS), df.columns
    cells = [[wrap(row[c], w) for c, (_, w) in COLUMNS.items()] for _, row in df.iterrows()]
    headers = [h for h, _ in COLUMNS.values()]

    plt.rcParams["font.family"] = "Arial"
    font_pt, pad_pt, line_pt = 8.0, 4.0, 10.5

    # Measure every line of text at the final font size, so each column is exactly as
    # wide as its widest line (plus padding) and the figure is drawn at true size.
    probe = plt.figure()
    renderer = probe.canvas.get_renderer()

    def text_width(s: str, bold: bool = False) -> float:
        t = probe.text(0, 0, s, fontsize=font_pt, fontweight="bold" if bold else "normal")
        w = t.get_window_extent(renderer).width * 72.0 / probe.dpi
        t.remove()
        return w

    col_pt = []
    for j, h in enumerate(headers):
        lines = [(ln, True) for ln in h.split("\n")]
        lines += [(ln, False) for row in cells for ln in row[j].split("\n")]
        col_pt.append(max(text_width(ln, b) for ln, b in lines) + 2 * pad_pt)
    plt.close(probe)

    n_lines = [max(c.count("\n") + 1 for c in row) for row in cells]
    head_lines = max(h.count("\n") + 1 for h in headers)
    row_pt = [head_lines * line_pt + 2 * pad_pt] + [n * line_pt + 2 * pad_pt for n in n_lines]
    width_pt, height_pt = sum(col_pt), sum(row_pt)

    fig = plt.figure(figsize=(width_pt / 72.0, height_pt / 72.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    table = ax.table(cellText=cells, colLabels=headers, loc="upper left", cellLoc="left",
                     colLoc="left", colWidths=[w / width_pt for w in col_pt])
    table.auto_set_font_size(False)
    table.set_fontsize(font_pt)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#bdbdbd")
        cell.set_linewidth(0.6)
        cell.PAD = pad_pt / col_pt[c]
        cell.set_height(row_pt[r] / height_pt)
        if r == 0:
            cell.set_facecolor("#e8e8e8")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#ffffff" if r % 2 else "#f5f5f5")
        cell.get_text().set_verticalalignment("center")
        cell.get_text().set_linespacing(1.15)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUT.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.05)
    print("saved", OUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
