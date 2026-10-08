#!/usr/bin/env bash
# Regenerate manuscript/references_list.tex (the formatted reference list read by main.tex) from
# manuscript/references.bib. Run it after adding, removing or editing references.
set -euo pipefail
cd "$(dirname "$0")/../manuscript"
rm -f references_list.tex
latexmk -pdf -g -interaction=nonstopmode main > /dev/null 2>&1   # BibTeX path: writes main.bbl
{
  echo "% Reference list generated with BibTeX (style elsarticle-harv) from references.bib."
  echo "% Regenerate after editing references.bib:  bash scripts/refresh_reference_list.sh  (or delete this file)."
  cat main.bbl
} > references_list.tex
latexmk -pdf -g -interaction=nonstopmode main > /dev/null 2>&1   # rebuild with the pre-built list
echo "manuscript/references_list.tex updated ($(grep -c '\\bibitem' references_list.tex) references)"
