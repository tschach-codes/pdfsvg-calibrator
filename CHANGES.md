# Changelog

## Unreleased
- Added transform-aware SVG parser with curve flattening and line fitting.
- Introduced SVG parsing tests and configuration options for curve tolerance and straightness thresholds.
- Implemented neighborhood-aware PDF↔SVG line matching with Hungarian assignment and confidence metrics.
- Added synthetic matching tests covering clean, ambiguous, and missing-target scenarios.
- Produced overlay visualizations (SVG/PDF) with labelled IDs and emitted a per-line CSV report.
- Wired the Typer-based CLI for end-to-end calibration with rich logging, error handling, and integration tests covering success and failure paths.
- Made the segment-length prefilter configurable for width/height/diagonal references and log the resulting pixel thresholds.
- **Docs:** Replaced the README with a user-friendly quickstart and added `docs/HOW_IT_WORKS.md` as deep-dive reference.
