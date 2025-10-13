# Code Generation Brief

## Tool Overview & Must-Haves
- The utility aligns a single PDF page with an exported SVG by estimating an axis-aligned affine transform `X = sx * x + tx`, `Y = sy * y + ty`, optionally considering 0°/180° (and hypothesis branches for 90°/270° flips) to cope with orientation differences.
- It must extract exactly horizontal or vertical line segments from both sources—regardless of element type (`line`, `rect`, `path`, `polyline`, `polygon`, stroked vs. filled) and treat each standalone element independently without stitching across objects.
- The algorithm needs to handle diverse coordinate systems (different origins, viewBox scales, nested transforms, symbol/use indirection, axis flips) and work even when the SVG applies raster fallbacks.
- RANSAC seeds with H/V hypotheses, tests flip combinations for `sx`/`sy`, refines translation using segment midpoints, and performs local optimization.
- Five long horizontal/vertical line pairs must be matched and validated to ±0.01 relative error, producing overlays (`*_overlay_lines.pdf`, `*_overlay_lines.svg`) with consistent IDs and a `*_check.csv` summary.
- Matching requires neighborhood signatures (relative positions/orientations/length ratios of nearby segments) that contribute to the cost and tie-breaking before running a Hungarian assignment (own implementation, no SciPy).

## Primary Components & Rationale
1. **Geometry Extraction Layer** – parses PDF via PyMuPDF and SVG via lxml, normalizes all primitives into candidate H/V segments, approximating curves through adaptive polyline sampling followed by a total-least-squares line fit to reduce them to segments. This ensures every relevant element is captured despite format differences.
2. **Transform Hypothesis & Estimation** – enumerates orientation flips (0°/180° core, optional 90°/270°) and uses RANSAC with chamfer-like scores to robustly estimate scale/translation parameters, accommodating outliers.
3. **Neighborhood Signature Builder** – for each segment, computes descriptors of adjacent segments (positions, orientation parity, length ratios) so that matching leverages spatial context, reducing false correspondences.
4. **Global Matcher** – combines transform and signature scores to assemble a cost matrix and solves for the top five matches using an in-house Hungarian algorithm to guarantee one-to-one assignments.
5. **Verification & Reporting** – revalidates the five matched pairs with precise geometry under the refined transform, generates overlays (PDF/SVG) with line IDs, and outputs a CSV containing per-line metrics and notes. Also emits status flags/confidences when data is insufficient or validation partly fails.
6. **Configuration & CLI** – exposes tolerances, iteration limits, and search radii via YAML config, accessible through `pdfsvg-calibrate run <pdf> --page N --config ... --outdir ...` using Typer/Rich for UX.

## Edge Cases & Coverage
- Coordinate system variations: differing origins, viewBox scaling, nested `transform` chains, `use`/`symbol` indirections, and axis flips (mirroring).
- Orientation differences: 0°/180° required, 90°/270° hypotheses considered but optional; sign flips on scales must be explored.
- Mixed geometry types: lines embedded in `path`, `rect` outlines, polygons, polylines, strokes vs. fills.
- Curves: approximated adaptively to detect nearly straight H/V segments and filtered via TLS fit.
- Rasterized SVGs or missing vector data: must detect and abort cleanly with diagnostic output.
- Sparse data (<5 long segments): tool should relax tolerances but still attempt matching, reporting lowered confidence.

## Fallback Strategies
- **Raster SVG Detection** – if no vector segments are extracted, raise a controlled error explaining the limitation.
- **Insufficient Segments** – when fewer than five viable lines exist, report partial matches, relax thresholds per config, and mark results with degraded confidence.
- **Matching Failures** – still emit overlay/CSV with available data, annotating unmatched slots and reasons.
- **Hypothesis Exhaustion** – if all orientation hypotheses fail, provide detailed logs and suggestions (e.g., check for 90° rotation) before aborting.

## Quick Test Strategy
- Unit-test geometry extraction on synthetic SVG/PDF fixtures to ensure H/V lines survive transforms and conversions.
- Test RANSAC and matching with mocked segments to verify flip handling and Hungarian implementation.
- Integration-style smoke test using a minimal PDF/SVG pair to confirm CLI, config loading, outputs, and tolerance enforcement.

## Documented Defaults
- Default to attempting 0°/180° orientations first, only exploring 90°/270° when explicitly enabled via config to control runtime.
- Treat "long" lines via a configurable minimum length threshold, initially 10% of max dimension unless overridden.
- Neighborhood signature considers the four nearest H/V neighbors within a configurable radius.

## Progress – F
- Built deterministic neighborhood signatures with `(t, Δθ, ρ)` tuples.
- Added composite candidate cost combining direction, midpoint/endpoints, and neighbor penalties.
- Integrated custom Hungarian solver for five-to-many assignments.
- Produced match metrics (ratio, relative error, PASS/FAIL, confidence) across synthetic tests.

## Progress – G
- Implemented SVG overlays with transformed PDF baselines and labelled matches.
- Added PDF overlays that mirror original geometry and embed readable ID badges.
- Generated CSV reports combining model parameters with per-line metrics and notes.
- Created smoke tests that assert overlay structure and CSV content for matched/unmatched cases.

## Progress – H
- Added Typer CLI `pdfsvg-calibrate run` that wires config loading, geometry extraction, calibration, and reporting.
- Integrated Rich-powered status logging with plain-text fallback and deterministic RNG overrides.
- Hardened error paths for missing vectors, calibration failures, and unreadable configs with exit codes 2/1.
- Summarised results via model/score panel plus five-line table and emitted overlay/CSV paths in the console.
- Added subprocess-based end-to-end tests for happy path, raster SVG rejection, and calibration failure diagnostics.

## Progress – R
- Dokumentationseinheit ergänzt: laienfreundliche README mit Quickstart, Outputs, Troubleshooting und FAQ.
- Technische Referenz `docs/HOW_IT_WORKS.md` verfasst (Pipeline, Datenflüsse, Parametertabelle mit Code-Fundstellen, Debug-/Performance-Hinweise).
