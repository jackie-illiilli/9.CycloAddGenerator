# Editable ChemDraw reaction schemes

Open `ChemDraw_reactions_all.cdxml` in ChemDraw for all 30 reactions on five vertical A4 sheets.
`ChemDraw_reactions_01.cdxml` through `ChemDraw_reactions_05.cdxml` contain six
reactions each as individual A4 documents. Reaction order follows the parent
`manifest.csv`.

The combined document has one logical CDXML drawing space (`page`), with
`WidthPages=1` and `HeightPages=5`. Each sheet's contents have a different vertical
coordinate offset. With 36 pt margins, each A4 printable tile is
523.275591 x 769.889764 pt. This avoids overlaying multiple page objects at the
same coordinates. A4 print metadata replaces the Letter metadata in the template.

The files contain native atom, bond, text, graphic, and reaction-step objects,
including heavy-atom reaction mappings. Molecular drawings are editable chemistry,
not embedded images. Drawing settings were extracted from the repository's
`Exa.cds` stylesheet: Arial 10 pt atom labels, 14.4 pt bond length,
1.303925 pt bond width, 2.296051 pt bold bond width, 1.587387 pt margin width,
and 2.494476 pt hash spacing. Atom and bond
stereo labels are hidden while wedge/hash bonds preserve the supplied stereochemistry.
Standalone explicit H atoms are omitted; heteroatom hydrogen counts are retained.

No colored circles are used. The template's purple/magenta (`#FF00FF`, index 9)
colors the four atoms and three bonds of the reactive 4pi conjugated path. The
template's blue (`#0000FF`, index 8) colors the two reactive 2pi atoms and every
original dienophile bond incident to either atom, including substituent connections.
The product inherits these same atom colors and original bond colors through AAM.
The two newly formed cross-fragment bonds remain black; all other unselected
atoms and bonds are black. Heteroatom labels use their atom's assigned color.

For aromatic dienes, the selected four-atom path must have the displayed Kekule
bond-order pattern 2-1-2 and change to 1-2-1 in the mapped product. This identifies
one consistent conjugated path in each of the 30 drawings without coloring an
entire aromatic ring. Reaction arrows, plus signs, captions, atoms, and bonds
remain editable. The mapped structures come from the existing reaction manifest.

All 90 molecular fragments were read back from the completed CDXML and compared
against the source graphs with stereochemistry included. Color inheritance,
four-atom/three-bond purple paths, blue incident bonds, and distinct per-sheet
coordinate ranges were checked in the actual files. In the final reaction's
product, two implicit-H wedge directions were corrected to preserve the source
SMILES stereochemistry. The source SVG files were not modified.

The installed ChemScript probe did not return and the desktop automation channel
was unavailable, so native ChemDraw rendering could not be inspected automatically.
The files were generated directly as CDXML objects, rather than exported through
the ChemDraw application. Open them in ChemDraw for final typography and print review.

Rebuild using the parent `build_chemdraw_cdxml.py` script with RDKit and NumPy.
The chemical object layout follows the published
[CDX/CDXML specification](https://iupac.github.io/IUPAC-FAIRSpec/cdx_sdk/Document.htm)
and its [multi-page drawing-space convention](https://iupac.github.io/IUPAC-FAIRSpec/cdx_sdk/Page.htm).
