"""Extract verbatim sample records from each raw dataset for the manuscript (Table 'sample rows').

    python scripts/sample_rows.py  ->  manuscript/tables/sample_rows.tex
"""
from __future__ import annotations

from pathlib import Path

from vigil.config import data_root, load_config, repo_path


def grab(path: Path, starts: str, n: int = 2, header_prefix: str | None = None, sep: str | None = None):
    lines = Path(path).read_text(errors="ignore").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    header = None
    if header_prefix is not None:
        header = next(l for l in lines if l.startswith(header_prefix))
    else:
        header = lines[0]
    i = next(k for k, l in enumerate(lines) if starts in l)
    return header, lines[i:i + n]


def tex_escape(s: str) -> str:
    return s.replace("\\", "/").replace("_", r"\_").replace("%", r"\%").replace("#", r"\#").replace("&", r"\&")


def fmt_tab(s: str, keep: int) -> str:
    parts = [p.strip().strip('"') for p in s.replace(";\t", "\t").replace(",", "\t").split("\t")]
    parts = [p for p in parts if p != ""]
    out = [tex_escape(p) for p in parts[:keep]]
    if len(parts) > keep:
        out.append(r"\ldots")
    return ", ".join(out)


def main():
    cfg = load_config("data")
    root = data_root()
    caf = root / cfg["paths"]["caf_dir"] / "caf_sensors" / "Hourly" / "CAF003.txt"
    h_caf, r_caf = grab(caf, "03/15/2015\t6:00")
    power = root / cfg["paths"]["power_csv"]
    h_pow, r_pow = grab(power, "2015,3,15,6,", header_prefix="YEAR,")
    scan = root / cfg["paths"]["scan_csv"]
    h_scan, r_scan = grab(scan, "2015-03-15 06:00", header_prefix="Date,")
    bb = root / cfg["paths"]["bitbrains_dir"] / "1.csv"
    h_bb, r_bb = grab(bb, "1376314846")
    rows = [
        ("CAF (hourly, CAF003.txt)", h_caf, r_caf, 9),
        ("NASA POWER (hourly, LST)", h_pow, r_pow, 9),
        ("SCAN 2198 (hourly, US units)", h_scan, r_scan, 5),
        ("Bitbrains GWA-T-12 (VM 1)", h_bb, r_bb, 6),
    ]
    lines = [r"\begin{table*}[t]", r"\centering", r"\scriptsize",
             r"\caption{Verbatim records from the four raw datasets (first fields of the header line and of two "
             r"consecutive records; `\ldots' marks omitted fields). VWC: volumetric water content "
             r"(m$^3$\,m$^{-3}$); T: soil temperature ($^\circ$C). The Bitbrains time stamp is in Unix seconds despite its header label.}",
             r"\label{tab:samples}", r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.19\textwidth}>{\raggedright\arraybackslash}p{0.77\textwidth}@{}}", r"\toprule",
             r"Dataset & Header / records \\", r"\midrule"]
    for name, h, rs, keep in rows:
        lines.append(rf"{tex_escape(name)} & \texttt{{{fmt_tab(h, keep)}}} \\")
        for r in rs:
            lines.append(rf" & \texttt{{{fmt_tab(r, keep)}}} \\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    out = repo_path("manuscript", "tables", "sample_rows.tex")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
