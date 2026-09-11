#!/usr/bin/env bash
# Rebuild the code-architecture diagram in docs/assets from its model.
#
# Requirements: a TeX distribution providing pdflatex with the standalone, tikz,
# helvet and sansmath packages; poppler's pdftocairo; Python with Pillow.
#
# Usage: bash tools/figures/build_architecture.sh
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
assets="$here/../../docs/assets"

# MacTeX installs outside the default PATH; add it only if it is there.
if [ -d /Library/TeX/texbin ]; then
  export PATH="/Library/TeX/texbin:$PATH"
fi

for tool in pdflatex pdftocairo python; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "error: $tool not found on PATH" >&2
    exit 1
  }
done

work="$(mktemp -d)"
keep=0
trap '[ "$keep" = 1 ] && echo "build files kept in $work" >&2 || rm -rf "$work"' EXIT

cp -r "$here/img" "$work/"

for mode in light dark; do
  python "$here/gen_tikz.py" "$here/arch_final.json" "$work/arch_$mode.tex" "$mode"
  if ! (cd "$work" && pdflatex -interaction=nonstopmode -halt-on-error "arch_$mode.tex" \
        > "arch_$mode.build.log" 2>&1); then
    keep=1
    echo "error: pdflatex failed for the $mode variant; see $work/arch_$mode.build.log" >&2
    tail -20 "$work/arch_$mode.build.log" >&2
    exit 1
  fi
  pdftocairo -svg "$work/arch_$mode.pdf" "$work/arch_$mode.svg"
  python "$here/add_svg_links.py" \
    "$work/arch_$mode.svg" "$work/arch_$mode.links.json" "$work/final_$mode.svg"
done

cp "$work/final_light.svg" "$assets/proteus_architecture.svg"
cp "$work/final_dark.svg" "$assets/proteus_architecture_darkmode.svg"
echo "updated $assets/proteus_architecture{,_darkmode}.svg"
