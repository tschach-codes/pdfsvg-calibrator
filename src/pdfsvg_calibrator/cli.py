from __future__ import annotations

import copy
import importlib.util
import logging
import shutil
import math
import os
import tempfile
import sys
import textwrap
import traceback
from dataclasses import replace
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)

import typer
import yaml

from pprint import pformat

from pdfsvg_calibrator.dimscale_extractor import estimate_dimline_scale
from pdfsvg_calibrator.pagebox_align import (
    compute_pagebox_alignment,
    confidence_label,
    explain_alignment_reason,
)
from .raster_align import coarse_raster_align as estimate_raster_alignment

from .config import load_config
from .calibrate import calibrate
from .calibrate_pipeline import calibrate_pdf_svg_preprocess
from .io_svg_pdf import (
    convert_pdf_to_svg_if_needed,
    load_pdf_segments,
    load_svg_segments,
)
from .metrics import MetricsTracker, Timer, use_tracker
from .match_verify import match_lines, select_lines
from .overlays import write_pdf_overlay, write_report_csv, write_svg_overlay
from .types import Match, Model, Segment
from .svg_autogen import ensure_svg_for_pdf
from .utils.timer import timer
from .debug.pdf_probe import probe_page


_RICH_AVAILABLE = importlib.util.find_spec("rich") is not None
if _RICH_AVAILABLE:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.theme import Theme
    import rich  # type: ignore[import-not-found]
else:  # pragma: no cover - executed only when Rich is missing
    Console = None  # type: ignore[assignment]
    Panel = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]
    Theme = None  # type: ignore[assignment]

    class _RichStub:
        @staticmethod
        def print(*objects: object, **kwargs: object) -> None:
            print(*objects, **kwargs)

    rich = _RichStub()  # type: ignore[assignment]


log = logging.getLogger(__name__)


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "configs" / "default.yaml"


HARDCODED_DEFAULTS: Dict[str, Any] = {
    "orientation": {
        "enabled": True,
        "raster_size": 512,
        "use_phase_correlation": True,
        "sample_topk_rel": 0.2,
    },
    "coarse": {
        "enabled": True,
        "topk_rel_by_length": 0.1,
        "hv_angle_tol_deg": 6.0,
        "bins": 4096,
        "blur_sigma_bins": 1.5,
        "orientations": [0, 90, 180, 270],
        "flips": ["none", "x", "y"],
        "scale_quantiles": [0.05, 0.95],
        "fallback_use_heatmap": True,
        "heatmap": {
            "raster": 1024,
            "blur_sigma_px": 2.0,
            "iters_phase": 1,
        },
        "kdtree_match": {
            "angle_tol_deg": 4.0,
            "len_tol_rel": 0.15,
            "dist_tol_px": 8.0,
            "max_pairs": 3000,
        },
        "score_weights": {
            "corr_x": 0.4,
            "corr_y": 0.4,
            "inliers": 0.2,
        },
        "adapt": {
            "min_long_segments": 300,
            "relax_topk_rel": [0.1, 0.2, 0.4],
            "relax_hv_angle": [6.0, 8.0, 10.0],
        },
        "refine_windows": {
            "scale_rel": 0.02,
            "shift_px": 8.0,
            "angle_deg": 3.0,
        },
        "debug": False,
    },
    "rot_degrees": [0, 180],
    "angle_tol_deg": 6.0,
    "curve_tol_rel": 0.001,
    "straight_max_dev_rel": 0.0015,
    "straight_max_angle_spread_deg": 3.0,
    "min_seg_length_rel": 0.1,
    "min_len_raw": 6.0,
    "prefilter": {"len_rel_ref": "diagonal"},
    "merge": {
        "enable": True,
        "collinear_angle_tol_deg": 3.0,
        "gap_max_rel": 0.003,
        "offset_tol_rel": 0.002,
        "off_tol_rel": 0.002,
        "thresholds": {
            "collinear_angle_tol_deg": 3.0,
            "gap_max_rel": 0.003,
            "offset_tol_rel": 0.002,
        },
    },
    "grid": {"initial_cell_rel": 0.05, "final_cell_rel": 0.02},
    "chamfer": {"sigma_rel": 0.004, "hard_mul": 3.0},
    "refine": {
        "max_iters": 60,
        "max_samples": 1500,
        "scale_max_dev_rel": 0.02,
        "trans_max_dev_px": 8.0,
    },
    "ransac": {
        "iters": 60,
        "refine_scale_step": 0.004,
        "refine_trans_px": 3.0,
        "patience": 60,
    },
    "verify": {
        "mode": "lines",
        "pick_k": 5,
        "diversity_rel": 0.10,
        "radius_px": 80.0,
        "dir_tol_deg": 6.0,
        "tol_rel": 0.01,
        "angle_tol_deg": 4.0,
    },
    "neighbors": {
        "use": True,
        "radius_rel": 0.06,
        "dt": 0.03,
        "dtheta_deg": 8.0,
        "rho_soft": 0.25,
        "penalty_miss": 1.5,
        "penalty_empty": 5.0,
    },
    "cost_weights": {
        "endpoint": 0.5,
        "midpoint": 0.3,
        "direction": 0.2,
        "neighbors": 0.1,
    },
    "sampling": {"step_rel": 0.03, "max_points": 1500},
}


def _normalize_opts_arguments(argv: List[str]) -> None:
    if not argv:
        return
    normalized: List[str] = [argv[0]]
    idx = 1
    argc = len(argv)
    while idx < argc:
        token = argv[idx]
        if token == "--opts" and idx + 1 < argc:
            normalized.append(token)
            next_token = argv[idx + 1]
            if "=" not in next_token and idx + 2 < argc:
                value_token = argv[idx + 2]
                is_short_option = (
                    value_token.startswith("-")
                    and not value_token.startswith("--")
                    and len(value_token) == 2
                    and value_token[1].isalpha()
                )
                if not value_token.startswith("--") and not is_short_option:
                    normalized.append(f"{next_token}={value_token}")
                    idx += 3
                    continue
            normalized.append(next_token)
            idx += 2
            continue
        normalized.append(token)
        idx += 1
    argv[:] = normalized


# Backwards compatibility: allow calling ``python -m pdfsvg_calibrator.cli <pdf>``
# by implicitly inserting the ``run`` command while keeping explicit subcommand
# invocation (``python -m pdfsvg_calibrator.cli run <pdf>``) functional.
if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
    first = sys.argv[1]
    known_commands = {"run", "preprocess"}
    if first not in known_commands:
        sys.argv.insert(1, "run")

_normalize_opts_arguments(sys.argv)


class _PlainStatus:
    def __init__(self, logger: "Logger", message: str) -> None:
        self._logger = logger
        self._message = message

    def __enter__(self) -> "_PlainStatus":
        self._logger.info(f"{self._message}…")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        return None


class Logger:
    def __init__(self, verbose: bool) -> None:
        self.verbose = verbose
        if _RICH_AVAILABLE:
            theme = Theme({
                "info": "cyan",
                "step": "bold cyan",
                "warning": "bold yellow",
                "error": "bold red",
            })
            self.console = Console(theme=theme, highlight=False, record=False)
            self.err_console = Console(theme=theme, highlight=False, record=False, stderr=True)
        else:
            self.console = None
            self.err_console = None

    def _print(self, message: str, style: str | None = None, *, err: bool = False) -> None:
        if self.console is not None:
            target = self.err_console if err and self.err_console is not None else self.console
            if style:
                target.print(f"[{style}]{message}[/{style}]")
            else:
                target.print(message)
        else:
            typer.echo(message, err=err)

    def info(self, message: str) -> None:
        self._print(message, style="info")

    def step(self, message: str) -> None:
        if self.console is not None:
            self.console.print(f"[step]▶ {message}")
        else:
            typer.echo(f"▶ {message}")

    def warn(self, message: str) -> None:
        self._print(message, style="warning")

    def error(self, message: str) -> None:
        self._print(message, style="error", err=True)

    def debug(self, message: str) -> None:
        if self.verbose:
            if self.console is not None:
                self.console.print(f"[dim]{message}[/dim]")
            else:
                typer.echo(message)

    def status(self, message: str):  # type: ignore[override]
        if self.console is not None:
            return self.console.status(message)
        return _PlainStatus(self, message)


class _LoggingBridge(logging.Handler):
    def __init__(self, cli_logger: Logger, level: int) -> None:
        super().__init__(level)
        self._cli_logger = cli_logger

    def emit(self, record: logging.LogRecord) -> None:  # type: ignore[override]
        msg = self.format(record)
        if record.levelno >= logging.ERROR:
            self._cli_logger.error(msg)
        elif record.levelno >= logging.WARNING:
            self._cli_logger.warn(msg)
        elif record.levelno >= logging.INFO:
            self._cli_logger.info(msg)
        else:
            self._cli_logger.debug(msg)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key in base:
        value = base[key]
        if isinstance(value, Mapping):
            result[key] = copy.deepcopy(value)
        elif isinstance(value, list):
            result[key] = list(value)
        else:
            result[key] = value
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)  # type: ignore[arg-type]
        else:
            result[key] = value
    return result


def _set_nested(config: MutableMapping[str, Any], path: Sequence[str], value: Any) -> None:
    if not path:
        return
    cursor: MutableMapping[str, Any] = config
    for key in path[:-1]:
        next_value = cursor.get(key)
        if not isinstance(next_value, MutableMapping):
            next_value = {}
            cursor[key] = next_value
        cursor = next_value
    cursor[path[-1]] = value


def _parse_cli_override(entry: str) -> Tuple[Tuple[str, ...], Any]:
    if "=" not in entry:
        raise ValueError("--opts erwartet 'pfad wert' oder 'pfad=wert'")
    raw_path, raw_value = entry.split("=", 1)
    path = tuple(part.strip() for part in raw_path.split(".") if part.strip())
    if not path:
        raise ValueError("--opts benötigt einen Schlüsselpfad, z. B. orientation.sample_topk_rel")
    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError as exc:
        raise ValueError(f"--opts {raw_path}: Wert konnte nicht geparst werden ({exc})") from exc
    return path, value


def _parse_allow_flips(value: str) -> List[str]:
    tokens = [token.strip().lower() for token in value.split(",")]
    valid = {"none", "x", "y"}
    result: List[str] = []
    seen = set()
    for token in tokens:
        if not token:
            continue
        if token not in valid:
            raise ValueError(
                "Ungültiger Flip-Wert – erlaubt sind none, x, y (Komma-separiert)"
            )
        if token not in seen:
            result.append(token)
            seen.add(token)
    if not result:
        raise ValueError("--allow-flips erwartet eine Liste aus none,x,y")
    return result


def _adapt_config_for_raster(cfg: MutableMapping[str, object]) -> None:
    """Limit expensive refinement loops when PDF segments come from raster edges."""

    refine_cfg = cfg.setdefault("refine", {})
    if isinstance(refine_cfg, MutableMapping):
        try:
            max_iters = int(refine_cfg.get("max_iters", 60))
        except (TypeError, ValueError):
            max_iters = 60
        refine_cfg["max_iters"] = min(max_iters, 2)

        try:
            max_samples = int(refine_cfg.get("max_samples", 1500))
        except (TypeError, ValueError):
            max_samples = 1500
        refine_cfg["max_samples"] = min(max_samples, 80)

    ransac_cfg = cfg.setdefault("ransac", {})
    if isinstance(ransac_cfg, MutableMapping):
        try:
            iters = int(ransac_cfg.get("iters", 60))
        except (TypeError, ValueError):
            iters = 60
        ransac_cfg["iters"] = min(iters, 3)

    coarse_cfg = cfg.setdefault("coarse", {})
    if isinstance(coarse_cfg, MutableMapping):
        coarse_cfg.setdefault("enabled", False)
        try:
            bins = int(coarse_cfg.get("bins", 1024))
        except (TypeError, ValueError):
            bins = 1024
        coarse_cfg["bins"] = min(bins, 256)

        try:
            topk_rel = float(coarse_cfg.get("topk_rel_by_length", 0.1))
        except (TypeError, ValueError):
            topk_rel = 0.1
        coarse_cfg["topk_rel_by_length"] = min(topk_rel, 0.05)

        heatmap_cfg = coarse_cfg.setdefault("heatmap", {})
        if isinstance(heatmap_cfg, MutableMapping):
            try:
                raster = int(heatmap_cfg.get("raster", 1024))
            except (TypeError, ValueError):
                raster = 1024
            heatmap_cfg["raster"] = min(raster, 256)
        try:
            bins = int(coarse_cfg.get("bins", 1024))
        except (TypeError, ValueError):
            bins = 1024
        coarse_cfg["bins"] = min(bins, 256)

        try:
            topk_rel = float(coarse_cfg.get("topk_rel_by_length", 0.1))
        except (TypeError, ValueError):
            topk_rel = 0.1
        coarse_cfg["topk_rel_by_length"] = min(topk_rel, 0.05)

        heatmap_cfg = coarse_cfg.setdefault("heatmap", {})
        if isinstance(heatmap_cfg, MutableMapping):
            try:
                raster = int(heatmap_cfg.get("raster", 1024))
            except (TypeError, ValueError):
                raster = 1024
            heatmap_cfg["raster"] = min(raster, 256)

    sampling_cfg = cfg.setdefault("sampling", {})
    if isinstance(sampling_cfg, MutableMapping):
        try:
            max_points = int(sampling_cfg.get("max_points", 1500))
        except (TypeError, ValueError):
            max_points = 1500
        sampling_cfg["max_points"] = min(max_points, 200)

        try:
            step_rel = float(sampling_cfg.get("step_rel", 0.03))
        except (TypeError, ValueError):
            step_rel = 0.03
        sampling_cfg["step_rel"] = max(step_rel, 0.08)

    neighbors_cfg = cfg.setdefault("neighbors", {})
    if isinstance(neighbors_cfg, MutableMapping):
        neighbors_cfg["use"] = False

    verify_cfg = cfg.setdefault("verify", {})
    if isinstance(verify_cfg, MutableMapping):
        try:
            pick_k = int(verify_cfg.get("pick_k", 5))
        except (TypeError, ValueError):
            pick_k = 5
        verify_cfg["pick_k"] = min(pick_k, 1)


def _load_config_with_defaults(path: Path, rng_seed: Optional[int]) -> Dict[str, Any]:
    raw_cfg: Dict[str, Any] = {}
    loaded = load_config(str(path))
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, MutableMapping):
        raise ValueError("Config root must be a mapping")
    raw_cfg = dict(loaded)
    config = _deep_merge(HARDCODED_DEFAULTS, raw_cfg)
    if rng_seed is not None:
        config["rng_seed"] = int(rng_seed)
    neighbors = config.setdefault("neighbors", {})
    neighbors["use"] = True
    return config


def _ensure_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} nicht gefunden: {path}")
    if path.is_dir():
        raise FileNotFoundError(f"{description} darf keine Ordner sein: {path}")


def _load_pdf_page_bytes(pdf_path: Path, page_index: int) -> bytes:
    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - dependency issue
        raise RuntimeError("PyMuPDF (fitz) ist erforderlich, wurde aber nicht gefunden") from exc

    doc = fitz.open(str(pdf_path))
    try:
        page_count = doc.page_count
        if page_index < 0 or page_index >= page_count:
            max_index = max(page_count - 1, 0)
            raise ValueError(
                f"Seitenindex {page_index} außerhalb des gültigen Bereichs (0-{max_index})"
            )
        single = fitz.open()
        try:
            single.insert_pdf(doc, from_page=page_index, to_page=page_index)
            pdf_bytes = single.write()
        finally:
            single.close()
    finally:
        doc.close()

    return bytes(pdf_bytes)


def _renumber_matches(matches: Sequence[Match]) -> List[Match]:
    return [replace(match, id=index) for index, match in enumerate(matches, start=1)]


def _segment_axis(seg: Segment, angle_tol_deg: float) -> str:
    dx = seg.x2 - seg.x1
    dy = seg.y2 - seg.y1
    if dx == 0.0 and dy == 0.0:
        return "H"
    angle = math.degrees(math.atan2(dy, dx)) % 180.0
    delta_h = min(angle, 180.0 - angle)
    delta_v = abs(angle - 90.0)
    tol = max(angle_tol_deg, 0.0)
    if delta_h <= tol and delta_h <= delta_v:
        return "H"
    if delta_v <= tol:
        return "V"
    return "H" if abs(dx) >= abs(dy) else "V"


def _distmap_placeholder_matches(
    pdf_lines: Sequence[Segment], verify_cfg: Mapping[str, Any]
) -> List[Match]:
    pick_k = int(verify_cfg.get("pick_k", 5)) if verify_cfg else 5
    pick_k = max(0, pick_k)
    angle_tol = float(verify_cfg.get("angle_tol_deg", 4.0)) if verify_cfg else 4.0
    matches: List[Match] = []
    for seg in list(pdf_lines)[:pick_k]:
        axis = _segment_axis(seg, angle_tol)
        length = math.hypot(seg.x2 - seg.x1, seg.y2 - seg.y1)
        matches.append(
            Match(
                id=len(matches) + 1,
                axis=axis,
                pdf_seg=seg,
                svg_seg=None,
                cost=0.0,
                confidence=0.0,
                pdf_len=length,
                svg_len=None,
                rel_error=None,
                pass01=None,
            )
        )
    return matches


def _model_flip_label(model: Model) -> str:
    fx = int(model.flip_x)
    fy = int(model.flip_y)
    if fx < 0 and fy < 0:
        return "XY"
    if fx < 0:
        return "X"
    if fy < 0:
        return "Y"
    return "none"


def _alignment_diagnostics(
    model: Model, svg_size: Tuple[float, float], config: Mapping[str, Any]
) -> Dict[str, Any]:
    diag = math.hypot(*svg_size)
    tol_auto = max(diag * 0.002, 0.5)
    refine_cfg = config.get("refine", {}) if isinstance(config, Mapping) else {}
    trans_tol_cfg = abs(float(refine_cfg.get("trans_max_dev_px", 0.0) or 0.0))
    scale_tol_cfg = abs(float(refine_cfg.get("scale_max_dev_rel", 0.0) or 0.0))
    if trans_tol_cfg > 0.0:
        tol_used = trans_tol_cfg
        tol_source = "refine.trans_max_dev_px"
    else:
        tol_used = tol_auto
        tol_source = "auto (0.2% diag, ≥0.5px)"

    shift = math.hypot(model.tx, model.ty)
    scale_avg = 0.5 * (model.sx + model.sy)
    flip_x = model.flip_x < 0
    flip_y = model.flip_y < 0
    rot_norm = model.rot_deg % 360
    det_negative = (model.flip_x * model.flip_y) < 0

    flip_label = _model_flip_label(model)
    flip_desc = f"(x={model.flip_x:+.0f}, y={model.flip_y:+.0f})"

    bounds_parts: List[str] = []
    if scale_tol_cfg > 0.0:
        bounds_parts.append(f"scale±{scale_tol_cfg:.4f} rel (refine.scale_max_dev_rel)")
    if trans_tol_cfg > 0.0:
        bounds_parts.append(f"trans±{trans_tol_cfg:.3f}px (refine.trans_max_dev_px)")
    else:
        bounds_parts.append(f"trans≤{tol_auto:.3f}px (auto)")

    header_lines = [
        (
            f"rot={rot_norm}° | flips={flip_desc} | "
            f"sx={model.sx:.6f} | sy={model.sy:.6f}"
        ),
        (
            f"tx={model.tx:.3f}px | ty={model.ty:.3f}px | "
            f"|t|={shift:.3f}px (tol {tol_used:.3f}px via {tol_source})"
        ),
        "Bounds: " + (", ".join(bounds_parts) if bounds_parts else "n/a"),
    ]

    shift_exceeds = shift > (tol_used + 1e-6)
    if shift_exceeds:
        header_lines.append("⚠ Shift exceeds tolerance – verify SVG export origin.")

    return {
        "diag": diag,
        "tol": tol_used,
        "tol_auto": tol_auto,
        "tol_config": trans_tol_cfg,
        "tol_source": tol_source,
        "scale_tol_rel": scale_tol_cfg if scale_tol_cfg > 0.0 else None,
        "shift": shift,
        "tx": model.tx,
        "ty": model.ty,
        "scale_avg": scale_avg,
        "flip_x": flip_x,
        "flip_y": flip_y,
        "flip_label": flip_label,
        "rot_norm": rot_norm,
        "det_negative": det_negative,
        "shift_exceeds": shift_exceeds,
        "header_lines": header_lines,
        "bounds": header_lines[2] if len(header_lines) >= 3 else None,
    }


def _format_float(value: Optional[float], digits: int) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "-"
    return f"{value:.{digits}f}"


def _gather_row_notes(match: Optional[Match]) -> List[str]:
    notes: List[str] = []
    if match is None:
        notes.append("missing match entry")
        return notes
    if match.svg_seg is None:
        notes.append("no match")
    if match.pass01 is not None and match.pass01 <= 0:
        notes.append("FAIL rel tol")
    if match.confidence is not None and match.confidence < 0.3:
        notes.append("low confidence")
    return notes


def _build_summary_table(matches: Sequence[Match]) -> Tuple[Sequence[str], List[List[str]]]:
    matches_by_id = {match.id: match for match in matches}
    headers = [
        "ID",
        "Axis",
        "PDF-Len",
        "SVG-Len",
        "Ratio",
        "RelErr",
        "Pass01",
        "Conf",
        "Notes",
    ]
    rows: List[List[str]] = []
    for idx in range(1, 6):
        match = matches_by_id.get(idx)
        pdf_len = match.pdf_len if match is not None else None
        svg_len = match.svg_len if match is not None else None
        ratio = None
        if pdf_len and svg_len and pdf_len != 0.0:
            ratio = svg_len / pdf_len
        rel_err = match.rel_error if match is not None else None
        notes = ", ".join(_gather_row_notes(match))
        rows.append(
            [
                str(idx),
                match.axis if match is not None else "-",
                _format_float(pdf_len, 3),
                _format_float(svg_len, 3),
                _format_float(ratio, 4),
                _format_float(rel_err, 4),
                "" if match is None or match.pass01 is None else str(int(match.pass01)),
                _format_float(match.confidence if match is not None else None, 3),
                notes if notes else "",
            ]
        )
    return headers, rows


def _needs_mirror_notice(model: Model) -> bool:
    rot_norm = int(model.rot_deg) % 360
    has_flip = (model.flip_x < 0) or (model.flip_y < 0)
    return has_flip and 0 < rot_norm < 180


def _write_coarse_segments_temp(segments: Sequence[Segment]) -> Path:
    tmp = tempfile.NamedTemporaryFile(
        prefix="coarse_segments_", suffix=".csv", delete=False, mode="w", encoding="utf-8"
    )
    with tmp as handle:
        handle.write("x1,y1,x2,y2\n")
        for seg in segments:
            handle.write(
                f"{seg.x1:.6f},{seg.y1:.6f},{seg.x2:.6f},{seg.y2:.6f}\n"
            )
    return Path(tmp.name)


def _write_placeholder_overlays(
    pdf_path: Path, svg_path: Path, outdir: Path, page_index: int
) -> Mapping[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    base = os.path.splitext(os.path.basename(pdf_path))
    base_name = base[0]
    overlay_svg = outdir / f"{base_name}_p{page_index:03d}_overlay_lines.svg"
    overlay_pdf = outdir / f"{base_name}_p{page_index:03d}_overlay_lines.pdf"
    report_csv = outdir / f"{base_name}_p{page_index:03d}_check.csv"

    overlay_svg.write_text(
        "<svg xmlns=\"http://www.w3.org/2000/svg\"><!-- coarse-only placeholder --></svg>\n",
        encoding="utf-8",
    )
    try:
        shutil.copyfile(pdf_path, overlay_pdf)
    except Exception:
        overlay_pdf.write_text("coarse-only placeholder", encoding="utf-8")
    report_csv.write_text("line,status,notes\n", encoding="utf-8")
    return {"SVG": overlay_svg, "PDF": overlay_pdf, "CSV": report_csv}


def _report_coarse_only(
    logger: Logger,
    model: Model,
    coarse_info: Mapping[str, Any],
    *,
    pdf_path: Path,
    svg_path: Path,
    outdir: Path,
    page_index: int,
) -> None:
    rot_norm = int(model.rot_deg) % 360
    flip_desc = f"(x={model.flip_x:+.0f}, y={model.flip_y:+.0f})"
    logger.info("Grobausrichtung abgeschlossen.")
    logger.info(
        f"rot={rot_norm}° | flips={flip_desc} | "
        f"sx={model.sx:.6f} | sy={model.sy:.6f} | tx={model.tx:.3f}px | ty={model.ty:.3f}px"
    )
    logger.info(f"Score≈{model.score:.4f} (Grobausrichtung)")
    if _needs_mirror_notice(model):
        logger.warn("Spiegelung erkannt")

    matrix = coarse_info.get("transform_matrix")
    translation = coarse_info.get("translation")
    if (
        isinstance(matrix, tuple)
        and len(matrix) == 2
        and all(isinstance(row, tuple) and len(row) == 2 for row in matrix)
        and isinstance(translation, tuple)
        and len(translation) == 2
    ):
        logger.info("Transformationsmatrix (SVG→PDF):")
        logger.info(f"[{matrix[0][0]: .6f} {matrix[0][1]: .6f}]")
        logger.info(f"[{matrix[1][0]: .6f} {matrix[1][1]: .6f}]")
        logger.info(
            f"Translation: ({translation[0]:.3f}px, {translation[1]:.3f}px)"
        )
    else:
        logger.warn("Transformationsmatrix nicht verfügbar.")

    segments = coarse_info.get("svg_segments_coarse")
    if isinstance(segments, Sequence) and segments and isinstance(segments[0], Segment):
        temp_path = _write_coarse_segments_temp(segments)
        logger.info(f"Transformierte SVG-Segmente gespeichert: {temp_path}")
    else:
        logger.warn("Keine transformierten Segmente für Ausgabe erhalten.")

    placeholder_outputs = _write_placeholder_overlays(pdf_path, svg_path, outdir, page_index)
    logger.info("Outputs:")
    for label, path in placeholder_outputs.items():
        logger.info(f"  - {label}: {path}")


def _summarize(
    logger: Logger,
    model: Model,
    matches: Sequence[Match],
    outputs: Mapping[str, Path],
    line_info: Mapping[str, Any],
    extra_warnings: Iterable[str],
    alignment: Mapping[str, Any] | None,
) -> None:
    headers, rows = _build_summary_table(matches)
    rot_norm = int(model.rot_deg) % 360
    flip_desc = f"(x={model.flip_x:+.0f}, y={model.flip_y:+.0f})"
    summary_lines = [
        (
            f"rot={rot_norm}° | flips={flip_desc} | "
            f"sx={model.sx:.6f} | sy={model.sy:.6f} | tx={model.tx:.3f}px | ty={model.ty:.3f}px"
        ),
        (
            "DT QA: RMSE={rmse:.4f}px | P95={p95:.4f}px | Median={median:.4f}px | Score={score:.4f}".format(
                rmse=model.rmse,
                p95=model.p95,
                median=model.median,
                score=model.score,
            )
        ),
    ]
    if _needs_mirror_notice(model):
        summary_lines.append("Spiegelung erkannt")
    if alignment is not None:
        tol_source = alignment.get("tol_source", "config")
        summary_lines.append(
            "Shift |t|={shift:.3f}px (tol {tol:.3f}px via {tol_source}) | avg scale={scale_avg:.6f}".format(
                **alignment
            )
        )
        header_lines = alignment.get("header_lines")
        if header_lines and len(header_lines) >= 3:
            summary_lines.append(header_lines[2])
        if alignment.get("shift_exceeds"):
            summary_lines.append("⚠ Shift exceeds tolerance – verify SVG export origin.")

    warnings: List[str] = list(line_info.get("notes", []))
    warnings.extend(extra_warnings)
    warnings.extend(model.quality_notes)

    if logger.console is not None:
        logger.console.rule("Calibration Summary")
        panel = Panel("\n".join(summary_lines), title="Model", expand=False)
        logger.console.print(panel)
        table = Table(show_header=True, header_style="bold")
        for header in headers:
            table.add_column(header)
        for row in rows:
            table.add_row(*row)
        logger.console.print(table)
        logger.console.print("Outputs:")
        for label, path in outputs.items():
            logger.console.print(f"  • {label}: {path}")
        if warnings:
            logger.console.print("Warnings:", style="warning")
            for item in warnings:
                logger.console.print(f"  - {item}", style="warning")
    else:
        logger.info("Calibration Summary")
        for line in summary_lines:
            logger.info(line)
        logger.info("Outputs:")
        for label, path in outputs.items():
            logger.info(f"  - {label}: {path}")
        if warnings:
            logger.warn("Warnings:")
            for item in warnings:
                logger.warn(f"  - {item}")


def _log_timing_summary(tracker: MetricsTracker, stats: Mapping[str, float]) -> None:
    total = sum(stats.values())
    orientation = stats.get("Orientation", 0.0)
    seed = stats.get("Seed", 0.0)
    refine = stats.get("Refine", 0.0)
    verify = stats.get("Verify", 0.0)
    io_overlays = stats.get("IO/Overlays", 0.0)
    summary = (
        f"Total={total:.3f}s | Orientation={orientation:.2f}s | "
        f"Seed={seed:.2f}s | Refine={refine:.2f}s | "
        f"Verify={verify:.2f}s | IO/Overlays={io_overlays:.2f}s"
    )
    log.info("[timing] " + summary)
    chamfer_calls = int(tracker.get_count("chamfer.calls"))
    samples = int(tracker.get_count("sampling.points"))
    log.info("[timing] Chamfer calls=%d | Samples=%d", chamfer_calls, samples)


def _handle_known_exception(logger: Logger, exc: Exception, *, prefix: str | None = None) -> None:
    message = str(exc) if str(exc) else exc.__class__.__name__
    if prefix:
        message = f"{prefix}: {message}"
    logger.error(message)


app = typer.Typer(help="PDF→SVG calibration & verification")


@app.command("preprocess")
def preprocess(
    pdf: Path = typer.Argument(
        ..., exists=True, readable=True, resolve_path=True, help="Pfad zur PDF-Datei"
    ),
    svg: Optional[Path] = typer.Option(
        None,
        "--svg",
        exists=True,
        readable=True,
        resolve_path=True,
        help="Optional: vorhandene SVG verwenden; andernfalls automatisch aus der PDF erzeugen.",
    ),
    svg_from: Literal["auto", "inkscape", "pdftocairo", "pdf2svg", "mutool", "pymupdf"] = typer.Option(
        "auto",
        "--svg-from",
        show_default=True,
        help="Backend für die SVG-Erzeugung aus der PDF",
    ),
    keep_text: bool = typer.Option(
        True,
        "--keep-text/--no-keep-text",
        show_default=True,
        help="Textobjekte beim SVG-Export beibehalten",
    ),
    keep_layers: bool = typer.Option(
        True,
        "--keep-layers/--no-keep-layers",
        show_default=True,
        help="Layer-/Gruppenstruktur beim SVG-Export beibehalten",
    ),
    page: int = typer.Option(0, "--page", min=0, show_default=True, help="0-basierter Seitenindex"),
    config: Path = typer.Option(
        DEFAULT_CONFIG_PATH,
        "--config",
        resolve_path=True,
        show_default=True,
        help="Pfad zur YAML-Konfiguration",
    ),
    outdir: Path = typer.Option(
        Path("out"),
        "--outdir",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        show_default=True,
        help="Ausgabeverzeichnis für generierte Dateien",
    ),
    save_debug_rasters: bool = typer.Option(
        False,
        "--save-debug-rasters",
        help="Debug-Raster (DPI Debug) exportieren",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        is_flag=True,
        help="Ausführliche Ausgaben aktivieren",
    ),
    dpi_coarse_opt: Optional[int] = typer.Option(
        None,
        "--dpi-coarse",
        min=1,
        help="DPI für die Grobrenderings überschreiben",
    ),
    dpi_coarse_fallback_opt: Optional[int] = typer.Option(
        None,
        "--dpi-coarse-fallback",
        min=1,
        help="Fallback-DPI für die Grobphase",
    ),
    dpi_debug_opt: Optional[int] = typer.Option(
        None,
        "--dpi-debug",
        min=1,
        help="DPI für Debug-Exporte",
    ),
    svg_ppu_coarse_opt: Optional[float] = typer.Option(
        None,
        "--svg-ppu-coarse",
        min=0.0001,
        help="Pixels-per-unit für SVG (Grobphase)",
    ),
    svg_ppu_fallback_opt: Optional[float] = typer.Option(
        None,
        "--svg-ppu-fallback",
        min=0.0001,
        help="Fallback-Pixels-per-unit für SVG",
    ),
    svg_ppu_debug_opt: Optional[float] = typer.Option(
        None,
        "--svg-ppu-debug",
        min=0.0001,
        help="Pixels-per-unit für SVG Debug-Raster",
    ),
) -> None:
    logger = Logger(verbose=verbose)
    try:
        config_data = load_config(str(config))
        if config_data is None:
            config_dict: Dict[str, Any] = {}
        elif isinstance(config_data, MutableMapping):
            config_dict = dict(config_data)
        else:
            raise ValueError("Config root must be a Mapping")

        svg_export_cfg = config_dict.get("svg_export")
        if not isinstance(svg_export_cfg, MutableMapping):
            svg_export_cfg = {}
        else:
            svg_export_cfg = dict(svg_export_cfg)
        svg_export_cfg["method"] = str(svg_from)
        svg_export_cfg["keep_text"] = bool(keep_text)
        svg_export_cfg["keep_layers"] = bool(keep_layers)
        svg_export_cfg["page"] = int(page)
        config_dict["svg_export"] = svg_export_cfg

        raster_cfg = config_dict.get("raster")
        if not isinstance(raster_cfg, MutableMapping):
            raster_cfg = {}
        else:
            raster_cfg = dict(raster_cfg)
        if dpi_coarse_opt is not None:
            raster_cfg["dpi_coarse"] = int(dpi_coarse_opt)
        if dpi_coarse_fallback_opt is not None:
            raster_cfg["dpi_coarse_fallback"] = int(dpi_coarse_fallback_opt)
        if dpi_debug_opt is not None:
            raster_cfg["dpi_debug"] = int(dpi_debug_opt)
        if svg_ppu_coarse_opt is not None:
            raster_cfg["svg_ppu_coarse"] = float(svg_ppu_coarse_opt)
        if svg_ppu_fallback_opt is not None:
            raster_cfg["svg_ppu_fallback"] = float(svg_ppu_fallback_opt)
        if svg_ppu_debug_opt is not None:
            raster_cfg["svg_ppu_debug"] = float(svg_ppu_debug_opt)
        config_dict["raster"] = raster_cfg

        svg_path = ensure_svg_for_pdf(pdf, svg, outdir, config_dict)
        svg_bytes = svg_path.read_bytes()
        pdf_bytes = _load_pdf_page_bytes(pdf, page)

        debug_dir: Optional[Path] = None
        if save_debug_rasters:
            debug_dir = outdir / "debug_rasters" / pdf.stem
            debug_dir.mkdir(parents=True, exist_ok=True)

        result = calibrate_pdf_svg_preprocess(
            pdf_bytes,
            svg_bytes,
            config_dict,
            save_debug_rasters=save_debug_rasters,
            debug_outdir=str(debug_dir) if debug_dir else None,
            debug_prefix=f"pre_{pdf.stem}",
        )
        if not isinstance(result, Mapping):
            raise ValueError("Preprocess-Ergebnis hat unerwartetes Format")

        coarse = result.get("coarse_alignment")
        metric = result.get("metric_scale")
        coarse_map = coarse if isinstance(coarse, Mapping) else {}
        metric_map = metric if isinstance(metric, Mapping) else {}
        scale_summary = result.get("scale_summary")
        scale_map = scale_summary if isinstance(scale_summary, Mapping) else {}

        def _fmt(value: Any) -> str:
            if isinstance(value, float):
                return f"{value:.6f}"
            return str(value)

        typer.echo("=== Coarse alignment ===")
        typer.echo(f"rotation_deg: {_fmt(coarse_map.get('rotation_deg'))}")
        typer.echo(f"flip_horizontal: {_fmt(coarse_map.get('flip_horizontal'))}")
        typer.echo(f"tx_px: {_fmt(coarse_map.get('tx_px'))}")
        typer.echo(f"ty_px: {_fmt(coarse_map.get('ty_px'))}")
        typer.echo(f"scale_hint: {_fmt(coarse_map.get('scale_hint'))}")
        typer.echo(f"score: {_fmt(coarse_map.get('score'))}")
        if coarse_map.get("used_fallback"):
            typer.echo("used_fallback: True")

        typer.echo("=== Metric scale ===")
        typer.echo(f"pixels_per_meter: {_fmt(metric_map.get('pixels_per_meter'))}")
        typer.echo(f"ok: {_fmt(metric_map.get('ok'))}")

        typer.echo("=== Scale comparison ===")
        typer.echo(f"scale_hint: {_fmt(scale_map.get('scale_hint'))}")
        typer.echo(f"scale_vbox: {_fmt(scale_map.get('scale_vbox'))}")
        typer.echo(f"scale_dim: {_fmt(scale_map.get('scale_dim'))}")
        typer.echo(
            f"diff_hint_vs_vbox_pct: {_fmt(scale_map.get('diff_hint_vs_vbox_pct'))}"
        )
        typer.echo(f"diff_hint_vs_dim_pct: {_fmt(scale_map.get('diff_hint_vs_dim_pct'))}")
        typer.echo(f"diff_vbox_vs_dim_pct: {_fmt(scale_map.get('diff_vbox_vs_dim_pct'))}")

        timing_summary = result.get("timing_summary")
        if isinstance(timing_summary, str) and timing_summary:
            typer.echo(timing_summary)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        _handle_known_exception(logger, exc, prefix="Fehler")
        raise typer.Exit(code=2) from exc
    except Exception as exc:  # pragma: no cover - fallback path
        _handle_known_exception(logger, exc, prefix="Unerwarteter Fehler")
        raise typer.Exit(code=1) from exc


@app.command("run")
def run(
    pdf: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True, help="Pfad zur PDF-Datei"),
    page: int = typer.Option(..., "--page", min=0, help="0-basierter Seitenindex"),
    no_pdfium: bool = typer.Option(
        False,
        "--no-pdfium",
        help="Legacy-Flag ohne Wirkung (Vektorextraktion entfernt)",
    ),
    config: Path = typer.Option(
        DEFAULT_CONFIG_PATH,
        "--config",
        resolve_path=True,
        help="Pfad zur YAML-Konfiguration",
    ),
    outdir: Path = typer.Option(Path("out"), "--outdir", resolve_path=True, help="Ausgabeverzeichnis"),
    svg: Optional[Path] = typer.Option(
        None,
        "--svg",
        resolve_path=True,
        help="Pfad zu einer bereits exportierten SVG-Datei",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Ausführliche Ausgaben"),
    rng_seed: Optional[int] = typer.Option(None, "--rng-seed", help="Zufalls-Seed überschreiben"),
    orientation_enabled: Optional[bool] = typer.Option(
        None,
        "--orientation-enabled/--no-orientation-enabled",
        help="Automatische Orientierungserkennung aktivieren/deaktivieren",
    ),
    orientation_raster_size: Optional[int] = typer.Option(
        None,
        "--orientation-raster-size",
        min=64,
        help="Rasterauflösung für die Orientierungserkennung überschreiben",
    ),
    refine_max_iters: Optional[int] = typer.Option(
        None,
        "--refine-max-iters",
        min=1,
        help="Maximale Iterationen der Feinabstimmung",
    ),
    refine_max_samples: Optional[int] = typer.Option(
        None,
        "--refine-max-samples",
        min=1,
        help="Maximale Samples für die Feinabstimmung",
    ),
    refine_scale_max_dev_rel: Optional[float] = typer.Option(
        None,
        "--refine-scale-max-dev-rel",
        min=0.0,
        help="Relative Skalenabweichung für die Feinabstimmung",
    ),
    refine_trans_max_dev_px: Optional[float] = typer.Option(
        None,
        "--refine-trans-max-dev-px",
        min=0.0,
        help="Maximale Translation in Pixeln für die Feinabstimmung",
    ),
    grid_initial_cell_rel: Optional[float] = typer.Option(
        None,
        "--grid-initial-cell-rel",
        min=0.0,
        help="Relativer Anfangswert der Gitterzelle",
    ),
    grid_final_cell_rel: Optional[float] = typer.Option(
        None,
        "--grid-final-cell-rel",
        min=0.0,
        help="Relativer Endwert der Gitterzelle",
    ),
    prefilter_len_rel_ref: Optional[str] = typer.Option(
        None,
        "--prefilter-len-ref",
        help="Referenzmaß für Längenfilterung (width/height/diagonal)",
    ),
    merge_enabled: Optional[bool] = typer.Option(
        None,
        "--merge-enabled/--merge-disabled",
        help="Kolliniare SVG-Segmente zusammenführen",
    ),
    merge_collinear_angle_tol_deg: Optional[float] = typer.Option(
        None,
        "--merge-collinear-angle-deg",
        min=0.0,
        help="Winkel-Toleranz für Kolliniare-Merges",
    ),
    merge_gap_max_rel: Optional[float] = typer.Option(
        None,
        "--merge-gap-rel",
        min=0.0,
        help="Maximaler Abstand (relativ zur Diagonale) beim Mergen",
    ),
    merge_offset_tol_rel: Optional[float] = typer.Option(
        None,
        "--merge-offset-rel",
        min=0.0,
        help="Maximale Versatz-Toleranz (relativ zur Diagonale) beim Mergen",
    ),
    verify_angle_tol_deg: Optional[float] = typer.Option(
        None,
        "--verify-angle-tol-deg",
        min=0.0,
        help="Winkel-Toleranz für die Verifikation",
    ),
    verify_radius_px: Optional[float] = typer.Option(
        None,
        "--verify-radius-px",
        min=0.0,
        help="Suchradius in Pixel für die Verifikation",
    ),
    coarse_only: bool = typer.Option(
        False,
        "--coarse-only",
        help="Nur Grobausrichtung ausführen und anschließend Bericht ausgeben",
    ),
    coarse_debug: bool = typer.Option(
        False,
        "--coarse-debug",
        help="Debug-Ausgaben der Grobausrichtung (Profile, Heatmaps, Scores) speichern",
    ),
    allow_flips: Optional[str] = typer.Option(
        None,
        "--allow-flips",
        help="Grobausrichtungs-Flips überschreiben (none,x,y Komma-separiert)",
    ),
    opts: List[str] = typer.Option(
        [],
        "--opts",
        help=(
            "Konfigurationswerte überschreiben; Pfad und Wert angeben, z. B."
            " --opts verify.mode distmap oder --opts orientation.sample_topk_rel=0.4"
        ),
        show_default=False,
        metavar="PATH=VALUE",
    ),
) -> None:
    logger = Logger(verbose=verbose)

    log_level = logging.DEBUG if verbose else logging.INFO
    package_logger = logging.getLogger("pdfsvg_calibrator")
    package_logger.setLevel(log_level)
    package_logger.propagate = False
    for handler in list(package_logger.handlers):
        if isinstance(handler, _LoggingBridge):
            package_logger.removeHandler(handler)
    bridge = _LoggingBridge(logger, log_level)
    bridge.setFormatter(logging.Formatter("%(message)s"))
    package_logger.addHandler(bridge)

    tracker = MetricsTracker()
    stats: Dict[str, float] = {}
    coarse_debug_dir_path: Optional[Path] = None

    try:
        with use_tracker(tracker):
            with Timer("total.run"):
                logger.step("Konfiguration laden")
                try:
                    cfg = _load_config_with_defaults(config, rng_seed)
                except FileNotFoundError as exc:
                    raise
                except yaml.YAMLError as exc:
                    raise ValueError(f"Konfiguration konnte nicht gelesen werden: {exc}") from exc

                neighbors = cfg.setdefault("neighbors", {})
                neighbors["use"] = True

                allow_flips_values: Optional[List[str]] = None
                if allow_flips is not None:
                    allow_flips_values = _parse_allow_flips(allow_flips)

                overrides: List[Tuple[Tuple[str, ...], Any]] = [
                    (("orientation", "enabled"), orientation_enabled),
                    (("orientation", "raster_size"), orientation_raster_size),
                    (("refine", "max_iters"), refine_max_iters),
                    (("refine", "max_samples"), refine_max_samples),
                    (("refine", "scale_max_dev_rel"), refine_scale_max_dev_rel),
                    (("refine", "trans_max_dev_px"), refine_trans_max_dev_px),
                    (("grid", "initial_cell_rel"), grid_initial_cell_rel),
                    (("grid", "final_cell_rel"), grid_final_cell_rel),
                    (("prefilter", "len_rel_ref"),
                     prefilter_len_rel_ref.lower()
                     if isinstance(prefilter_len_rel_ref, str)
                     else prefilter_len_rel_ref),
                    (("merge", "enable"), merge_enabled),
                    (("merge", "collinear_angle_tol_deg"), merge_collinear_angle_tol_deg),
                    (("merge", "gap_max_rel"), merge_gap_max_rel),
                    (("merge", "offset_tol_rel"), merge_offset_tol_rel),
                    (("merge", "thresholds", "collinear_angle_tol_deg"), merge_collinear_angle_tol_deg),
                    (("merge", "thresholds", "gap_max_rel"), merge_gap_max_rel),
                    (("merge", "thresholds", "offset_tol_rel"), merge_offset_tol_rel),
                    (("verify", "angle_tol_deg"), verify_angle_tol_deg),
                    (("verify", "radius_px"), verify_radius_px),
                ]

                for entry in opts:
                    path, value = _parse_cli_override(entry)
                    overrides.append((path, value))

                for path, value in overrides:
                    if value is None:
                        continue
                    _set_nested(cfg, path, value)

                coarse_cfg_raw = cfg.setdefault("coarse", {})
                if not isinstance(coarse_cfg_raw, MutableMapping):
                    raise ValueError("coarse config must be a mapping")
                coarse_cfg_map = coarse_cfg_raw
                if allow_flips_values is not None:
                    coarse_cfg_map["flips"] = allow_flips_values
                coarse_debug_enabled = coarse_debug or bool(coarse_cfg_map.get("debug"))
                if coarse_debug_enabled:
                    coarse_cfg_map["debug"] = True

                config_yaml = yaml.safe_dump(
                    cfg,
                    sort_keys=False,
                    default_flow_style=False,
                ).strip()
                logger.info("Aktive Konfiguration:\n" + textwrap.indent(config_yaml, "  "))

                if svg is not None:
                    svg_path = svg.resolve()
                    _ensure_exists(svg_path, "SVG")
                else:
                    with logger.status("SVG ermitteln"):
                        svg_path = Path(
                            convert_pdf_to_svg_if_needed(str(pdf), page, str(outdir))
                        )

                outdir.mkdir(parents=True, exist_ok=True)
                if coarse_debug_enabled:
                    coarse_debug_dir_path = outdir / "coarse-debug"
                    coarse_debug_dir_path.mkdir(parents=True, exist_ok=True)
                else:
                    coarse_debug_dir_path = None

                logger.step("PDF-Segmente laden")
                with logger.status("PDF analysieren"):
                    try:
                        pdf_segs, pdf_size = load_pdf_segments(
                            str(pdf), page, cfg, use_pdfium=not no_pdfium
                        )
                    except RuntimeError as exc:
                        if (
                            svg is None
                            and "keine vektoriellen Segmente" in str(exc)
                        ):
                            raise ValueError("SVG enthält keine Vektoren") from exc
                        raise
                if isinstance(cfg, MutableMapping) and cfg.get("_pdf_segment_source") == "raster":
                    _adapt_config_for_raster(cfg)
                    if not coarse_only:
                        coarse_only = True
                logger.debug(
                    f"PDF-Segmente: {len(pdf_segs)} (Seite {page}, Größe {pdf_size})"
                )
                if verbose:
                    try:
                        probe_stats = probe_page(str(pdf), page)
                        logger.debug("PDF-Probe:\n" + pformat(probe_stats, width=100))
                    except Exception as exc:  # pragma: no cover - diagnostic fallback
                        logger.debug(f"PDF-Probe fehlgeschlagen: {exc}")

                logger.step("SVG-Segmente laden")
                with logger.status("SVG analysieren"):
                    svg_segs, svg_size = load_svg_segments(str(svg_path), cfg)
                logger.debug(f"SVG-Segmente: {len(svg_segs)} (Größe {svg_size})")

                try:
                    pagebox_align = compute_pagebox_alignment(
                        pdf_path=str(pdf),
                        page_index=page,
                        svg_path=str(svg_path),
                    )
                except Exception as exc:
                    pagebox_align = None
                    logger.warn(f"PageBox-Alignment: Fehler bei Berechnung: {exc}")

                if pagebox_align is not None:
                    logger.info("PageBox/ViewBox-Heuristik:")
                    logger.info(
                        f"  pdf_size={pagebox_align.pdf_size} "
                        f"svg_size={pagebox_align.svg_size}"
                    )
                    logger.info(
                        f"  rotation_deg={pagebox_align.rotation_deg:.1f} "
                        f"flip_x={pagebox_align.flip_x} "
                        f"flip_y={pagebox_align.flip_y}"
                    )
                    logger.info(
                        f"  scale_x={pagebox_align.scale_x:.6f} "
                        f"scale_y={pagebox_align.scale_y:.6f}"
                    )
                    logger.info(
                        f"  confidence={confidence_label(pagebox_align)} "
                        f"reason='{explain_alignment_reason(pagebox_align)}'"
                    )
                else:
                    logger.warn(
                        "PageBox/ViewBox-Heuristik nicht verfügbar (keine viewBox gefunden)"
                    )

                # ---- Rasterbasierter Fallback (PDF vs SVG als Bitmap) ----
                try:
                    rast_align = estimate_raster_alignment(
                        pdf_path=str(pdf),
                        page_index=page,
                        svg_path=str(svg_path),
                    )
                except Exception as exc:  # pragma: no cover - diagnostic only
                    rast_align = None
                    rich.print(
                        f"[yellow]Raster-Fallback: Fehler bei Berechnung: {exc}[/yellow]"
                    )

                if rast_align is not None:
                    rich.print("[bold magenta]Raster-Korrelation (PDF↔SVG-Bitmap):[/bold magenta]")
                    rich.print(
                        f"  rotation_deg={rast_align.rotation_deg:.1f} "
                        f"flip_y={rast_align.flip_y} "
                        f"scale≈{rast_align.scale:.4f}"
                    )
                    rich.print(
                        f"  shift_xy(px)={rast_align.shift_xy} "
                        f"score={rast_align.score:.4f} "
                        f"confidence={rast_align.confidence}"
                    )
                else:
                    rich.print("[yellow]Raster-Fallback nicht verfügbar[/yellow]")
                # ---- Ende Raster ----

                # ---- Maßlinien-basierte Skalierung (Dimension Lines) ----
                try:
                    dimscale_res = estimate_dimline_scale(
                        svg_path=svg_path,
                        enable_ocr_paths=False,  # set True if you want OCR-path fallback
                    )
                except Exception as e:  # pragma: no cover - diagnostic only
                    dimscale_res = None
                    rich.print(f"[yellow]Maßlinien-Scale: Fehler bei Berechnung: {e}[/yellow]")

                if dimscale_res is None:
                    rich.print("[yellow]Maßlinien-Scale nicht verfügbar[/yellow]")
                else:
                    rich.print("[bold green]Maßlinien-Scale (Dimension Lines):[/bold green]")
                    rich.print(
                        f"  ok={dimscale_res.ok} reason='{dimscale_res.reason}' "
                        f"candidates={dimscale_res.candidates_used} "
                        f"inliers={dimscale_res.inlier_cluster_size}"
                    )
                    rich.print(
                        "  scale_x: "
                        f"{dimscale_res.scale_x_svg_per_unit} svg/unit  |  "
                        f"{dimscale_res.scale_x_unit_per_svg} unit/svg"
                    )
                    rich.print(
                        "  scale_y: "
                        f"{dimscale_res.scale_y_svg_per_unit} svg/unit  |  "
                        f"{dimscale_res.scale_y_unit_per_svg} unit/svg"
                    )

                    # Debug-Preview-Crops (horizontal / vertical)
                    # We save them to outdir for visual QA if available.
                    # We'll name them dimscale_preview_h.png / dimscale_preview_v.png
                    try:
                        if dimscale_res.preview_horizontal is not None:
                            out_h_path = outdir / "dimscale_preview_h.png"
                            dimscale_res.preview_horizontal.save(out_h_path)
                            rich.print(f"  preview_horizontal: {out_h_path}")
                        if dimscale_res.preview_vertical is not None:
                            out_v_path = outdir / "dimscale_preview_v.png"
                            dimscale_res.preview_vertical.save(out_v_path)
                            rich.print(f"  preview_vertical: {out_v_path}")
                    except Exception as e:  # pragma: no cover - diagnostic only
                        rich.print(
                            f"[yellow]Warnung: konnte Maßlinien-Preview nicht speichern: {e}[/yellow]"
                        )

                # ---- Ende Maßlinien-Scale ----

                if not svg_segs:
                    raise ValueError("SVG enthält keine Vektoren")

                logger.step("Modell kalibrieren")
                with logger.status("Kalibrierung läuft"):
                    coarse_outputs: Dict[str, Any] = {}
                    coarse_outputs_ref: Optional[MutableMapping[str, Any]] = (
                        coarse_outputs if coarse_only else None
                    )
                    model = calibrate(
                        pdf_segs,
                        svg_segs,
                        pdf_size,
                        svg_size,
                        cfg,
                        stats=stats,
                        coarse_only=coarse_only,
                        coarse_outputs=coarse_outputs_ref,
                        coarse_debug_dir=coarse_debug_dir_path,
                    )

                if coarse_only:
                    _report_coarse_only(
                        logger,
                        model,
                        coarse_outputs,
                        pdf_path=pdf,
                        svg_path=svg_path,
                        outdir=outdir,
                        page_index=page,
                    )
                else:
                    alignment = _alignment_diagnostics(model, svg_size, cfg)

                    alignment_warnings: List[str] = []
                    rot_norm = alignment.get("rot_norm", model.rot_deg % 360)
                    if rot_norm != 0 and not alignment.get("det_negative", False):
                        alignment_warnings.append(
                            f"Rotation {rot_norm}° erkannt – automatische Exporte erwarten 0°."
                        )
                    if alignment.get("flip_x") or alignment.get("flip_y"):
                        axes: List[str] = []
                        if alignment.get("flip_x"):
                            axes.append("horizontal (sx < 0)")
                        if alignment.get("flip_y"):
                            axes.append("vertical (sy < 0)")
                        axis_desc = " & ".join(axes) if axes else "unbekannt"
                        if alignment.get("det_negative", False):
                            alignment_warnings.append(
                                "Automatisch erzeugte SVG wirkt gespiegelt: "
                                f"{axis_desc}."
                            )
                        else:
                            alignment_warnings.append(
                                "SVG-Export enthält Flip(s): "
                                f"{axis_desc}."
                            )
                    if alignment.get("shift", 0.0) > alignment.get("tol", 0.0):
                        tol_source = alignment.get("tol_source", "config")
                        alignment_warnings.append(
                            "Automatischer SVG-Export wirkt verschoben: "
                            f"|t|={alignment['shift']:.3f}px > {alignment['tol']:.3f}px (Grenze {tol_source}). "
                            "Bitte Ursprung der SVG-Exportdatei prüfen."
                        )

                    logger.step("Top-Linien auswählen")
                    with logger.status("Linienauswahl"):
                        with Timer("seed.select"):
                            pdf_lines, line_info = select_lines(pdf_segs, model, svg_segs, cfg)
                    logger.debug(f"Ausgewählte Linien: {len(pdf_lines)}")

                    verify_cfg = cfg.get("verify", {}) if isinstance(cfg, Mapping) else {}
                    verify_mode = ""
                    if isinstance(verify_cfg, Mapping):
                        verify_mode = str(verify_cfg.get("mode", "")).strip().lower()
                    step_label = "Distanz-Map QA" if verify_mode == "distmap" else "Linien abgleichen"
                    status_label = "Distanz-Map" if verify_mode == "distmap" else "Matching"
                    logger.step(step_label)
                    with logger.status(status_label):
                        with timer(stats, "Verify"):
                            if verify_mode == "distmap":
                                verify_cfg_map = verify_cfg if isinstance(verify_cfg, Mapping) else {}
                                raw_matches = _distmap_placeholder_matches(pdf_lines, verify_cfg_map)
                                if isinstance(line_info, MutableMapping):
                                    notes = line_info.setdefault("notes", [])
                                    if isinstance(notes, list):
                                        notes.append("verify.mode=distmap – pairwise matching skipped.")
                            else:
                                raw_matches = match_lines(pdf_lines, svg_segs, model, cfg)
                    matches = _renumber_matches(raw_matches)

                    extra_warnings: List[str] = alignment_warnings
                    if len(matches) < 5:
                        extra_warnings.append(
                            f"Nur {len(matches)} Linien verfügbar – verbleibende Slots bleiben leer."
                        )
                    if any(match.svg_seg is None for match in matches):
                        extra_warnings.append(
                            "Mindestens eine Linie konnte keinem SVG-Segment zugeordnet werden."
                        )

                    logger.step("Overlays & Bericht schreiben")
                    with timer(stats, "IO/Overlays"):
                        with logger.status("SVG-Overlay"):
                            overlay_svg = Path(
                                write_svg_overlay(
                                    str(svg_path),
                                    str(outdir),
                                    str(pdf),
                                    page,
                                    model,
                                    matches,
                                    alignment,
                                )
                            )
                        with logger.status("PDF-Overlay"):
                            overlay_pdf = Path(
                                write_pdf_overlay(
                                    str(pdf),
                                    page,
                                    str(outdir),
                                    model,
                                    matches,
                                    alignment,
                                )
                            )
                        with logger.status("CSV-Bericht"):
                            report_csv = Path(
                                write_report_csv(
                                    str(outdir),
                                    str(pdf),
                                    page,
                                    model,
                                    matches,
                                    alignment,
                                )
                            )

                    outputs = {
                        "SVG": overlay_svg,
                        "PDF": overlay_pdf,
                        "CSV": report_csv,
                    }

                    _summarize(
                        logger,
                        model,
                        matches,
                        outputs,
                        line_info,
                        extra_warnings,
                        alignment,
                    )

            _log_timing_summary(tracker, stats)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        _handle_known_exception(logger, exc, prefix="Fehler")
        raise typer.Exit(code=2) from exc
    except Exception as exc:  # pragma: no cover - fallback path
        _handle_known_exception(logger, exc, prefix="Unerwarteter Fehler")
        if verbose:
            traceback.print_exc()
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
