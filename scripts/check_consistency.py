"""Consistency checks between manuscript, generated results and configuration.

    python scripts/check_consistency.py

Checks that every macro used in the manuscript is generated from the result files, every input table and
figure exists, every citation key is in references.bib (and every entry is cited), and that the settings
quoted in the hand-written text agree with the configuration files.
"""
from __future__ import annotations

import re
import sys

from vigil.config import load_config, repo_path

M = repo_path("manuscript")


def main() -> int:
    errors = []
    tex = {f.name: f.read_text() for f in sorted((M / "sections").glob("*.tex"))}
    tex["main.tex"] = (M / "main.tex").read_text()
    for f in sorted((M / "tables").glob("*.tex")):
        tex["tables/" + f.name] = f.read_text()
    alltex = "\n".join(tex.values())

    # 1. macros used in the text are defined by the generated macro files (or in main.tex)
    defined = set()
    for f in (M / "macros").glob("*.tex"):
        defined |= set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", f.read_text()))
    defined |= set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", tex["main.tex"]))
    used = set(re.findall(r"\\([A-Z][A-Za-z]+)", alltex))
    latex_builtin = {"Var", "E", "R", "Delta", "Phi", "Gamma", "Big", "Bigl", "Bigr", "Huge", "Large", "LaTeX",
                     "Require", "State", "For", "EndFor", "If", "EndIf", "Comment", "Return", "TeX",
                     "DeclareMathOperator", "IfFileExists", "RequirePackage", "Psi", "Theta", "Lambda", "Omega", "Sigma", "Pi", "Xi"}
    missing = sorted(m for m in used - defined - latex_builtin if not m.startswith(("Big", "Large")))
    if missing:
        errors.append(f"undefined macros: {missing}")

    # 2. inputs and figures exist
    for path in re.findall(r"\\input\{([^}]+)\}", alltex):
        p = M / (path if path.endswith(".tex") else path + ".tex")
        if not p.exists() and "macros/" not in path:
            errors.append(f"missing input: {path}")
    for path in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", alltex):
        if not (M / path).exists():
            errors.append(f"missing figure: {path}")

    # 3. citations
    bib = (M / "references.bib").read_text()
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    cited = set()
    for group in re.findall(r"\\cite[a-z]*\{([^}]+)\}", alltex):
        cited |= {k.strip() for k in group.split(",")}
    if cited - keys:
        errors.append(f"cited but not in bib: {sorted(cited - keys)}")
    if keys - cited:
        errors.append(f"in bib but never cited: {sorted(keys - cited)}")

    # 3b. the pre-built reference list (read by main.tex instead of running BibTeX) is up to date
    rl = M / "references_list.tex"
    if rl.exists():
        items = set(re.findall(r"\\bibitem\[\{.*?\}\]\s*\{([^}]+)\}", rl.read_text(), flags=re.S))
        if items != cited:
            errors.append("references_list.tex is out of date (run scripts/refresh_reference_list.sh): "
                          f"missing {sorted(cited - items)}, extra {sorted(items - cited)}")
        bib_mtime = (M / "references.bib").stat().st_mtime
        if bib_mtime > rl.stat().st_mtime + 1:
            errors.append("references.bib is newer than references_list.tex (run scripts/refresh_reference_list.sh)")

    # 4. settings quoted in hand-written text
    exp, tr, data = load_config("experiments"), load_config("train"), load_config("data")
    setup = tex["06_setup.tex"]
    budgets = ",".join(str(k) for k in exp["budgets"])
    if f"K\\in\\{{{budgets}\\}}" not in setup:
        errors.append(f"budgets in setup do not match configs/experiments.yaml ({budgets})")
    if f"A_{{\\max}}={tr['train']['max_age']}" not in setup:
        errors.append("freshness guarantee in setup does not match configs/train.yaml")
    if f"\\eta={exp['eta']}" not in setup:
        errors.append("eta in setup does not match configs/experiments.yaml")
    if f"{len(exp['seeds'])} seeds" not in setup and "five seeds" not in setup:
        errors.append("number of seeds in setup")
    if f"k={data['graph']['knn']}" not in alltex.replace(" ", "") and f"k={data['graph']['knn']}" not in alltex:
        errors.append("kNN graph size not quoted consistently")

    for e in errors:
        print("ERROR:", e)
    print("consistency check:", "OK" if not errors else f"{len(errors)} problem(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
