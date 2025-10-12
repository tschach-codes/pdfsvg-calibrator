# pdfsvg-calibrator

Axis-aligned PDF→SVG calibration & verification.

### Quickstart
```bash
pip install -e .[dev]
pdfsvg-calibrate run examples/sample.pdf --page 0
```

Outputs in `out/`:
- `*_overlay_lines.pdf`
- `*_overlay_lines.svg`
- `*_check.csv`
