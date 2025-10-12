# Codegen Brief

## Progress – C
- Implemented PDF drawing extraction with adaptive curve flattening via PyMuPDF.
- Added TLS-based straightness reduction for subpaths without cross-element stitching.
- Enforced configurable straightness and angle spread gates plus H/V filtering logic.
- Improved SVG scene sizing by parsing width/height and viewBox units.
- Added regression tests and config defaults for curve tolerance and straightness thresholds.
