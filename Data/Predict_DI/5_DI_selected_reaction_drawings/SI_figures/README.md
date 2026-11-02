# SI reaction figures

84 reactions in HTML gallery order, grouped into 14 figures with up to 6 reactions each. Figure width: 180 mm. PNG resolution: 600 dpi.

Each figure is supplied as vector PDF, editable SVG, and a white-background PNG. The combined PDF contains one figure per page. Use index.html to browse the figures, captions.md for caption text, and figure_manifest.csv to trace every panel to its source reaction, AAM, and mapping provenance. Original reaction SVGs are unchanged.

Rebuild from the parent directory with Python packages reportlab, svglib, pypdf and Poppler installed:

```powershell
python build_si_figures.py --pdftoppm <path-to-pdftoppm>
```
