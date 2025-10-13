from __future__ import annotations

import copy
import importlib.util
import math
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import typer
import yaml

from .config import load_config
from .fit_model import calibrate
from .io_svg_pdf import (
    convert_pdf_to_svg_if_needed,
    load_pdf_segments,
    load_svg_segments,
)
from .match_verify import match_lines, select_lines
from .overlays import write_pdf_overlay, write_report_csv, write_svg_overlay
from .types import Match, Model


_RICH_AVAILABLE = importlib.util.find_spec("rich") is not None
if _RICH_AVAILABLE:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.theme import Theme
else:  # pragma: no cover - executed only when Rich is missing
    Console = None  # type: ignore[assignment]
    Panel = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]
    Theme = None  # type: ignore[assignment]


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "configs" / "default.yaml"


HARDCODED_DEFAULTS: Dict[str, Any] = {
    "rot_degrees": [0, 180],
    "angle_tol_deg": 6.0,
    "curve_tol_rel": 0.001,
    "straight_max_dev_rel": 0.0015,
    "straight_max_angle_spread_deg": 3.0,
    "min_len_raw": 6.0,
    "merge": {"gap_max_rel": 0.01, "off_tol_rel": 0.003},
    "grid_cell_rel": 0.02,
    "chamfer": {"sigma_rel": 0.004, "hard_mul": 3.0},
    "ransac": {"iters": 900, "refine_scale_step": 0.004, "refine_trans_px": 3.0},
    "verify": {
        "pick_k": 5,
        "diversity_rel": 0.10,
        "radius_px": 80.0,
        "dir_tol_deg": 6.0,
        "tol_rel": 0.01,
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
    "sampling": {"step_rel": 0.02, "max_points": 5000},
}


if len(sys.argv) > 1 and sys.argv[1] == "run":  # pragma: no cover - CLI convenience
    del sys.argv[1]


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


def _renumber_matches(matches: Sequence[Match]) -> List[Match]:
    return [replace(match, id=index) for index, match in enumerate(matches, start=1)]


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


def _summarize(
    logger: Logger,
    model: Model,
    matches: Sequence[Match],
    outputs: Mapping[str, Path],
    line_info: Mapping[str, Any],
    extra_warnings: Iterable[str],
) -> None:
    headers, rows = _build_summary_table(matches)
    flip_notes: List[str] = []
    if model.sx < 0:
        flip_notes.append("horizontal flip (sx < 0)")
    if model.sy < 0:
        flip_notes.append("vertical flip (sy < 0)")
    flip_text = ", ".join(flip_notes) if flip_notes else "none"
    summary_lines = [
        f"Model rot={model.rot_deg}° | sx={model.sx:.6f} sy={model.sy:.6f} tx={model.tx:.3f} ty={model.ty:.3f}",
        f"Flips: {flip_text}",
        f"Score={model.score:.4f} | RMSE={model.rmse:.4f} | P95={model.p95:.4f} | Median={model.median:.4f}",
    ]

    warnings: List[str] = list(line_info.get("notes", []))
    warnings.extend(extra_warnings)

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


def _handle_known_exception(logger: Logger, exc: Exception, *, prefix: str | None = None) -> None:
    message = str(exc) if str(exc) else exc.__class__.__name__
    if prefix:
        message = f"{prefix}: {message}"
    logger.error(message)


app = typer.Typer(help="PDF→SVG calibration & verification")


@app.command("run")
def run(
    pdf: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True, help="Pfad zur PDF-Datei"),
    page: int = typer.Option(..., "--page", min=0, help="0-basierter Seitenindex"),
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
) -> None:
    logger = Logger(verbose=verbose)

    try:
        logger.step("Konfiguration laden")
        try:
            cfg = _load_config_with_defaults(config, rng_seed)
        except FileNotFoundError as exc:
            raise
        except yaml.YAMLError as exc:
            raise ValueError(f"Konfiguration konnte nicht gelesen werden: {exc}") from exc

        neighbors = cfg.setdefault("neighbors", {})
        neighbors["use"] = True

        if svg is not None:
            svg_path = svg.resolve()
            _ensure_exists(svg_path, "SVG")
        else:
            with logger.status("SVG ermitteln"):
                svg_path = Path(convert_pdf_to_svg_if_needed(str(pdf), page, str(outdir)))

        outdir.mkdir(parents=True, exist_ok=True)

        logger.step("PDF-Segmente laden")
        with logger.status("PDF analysieren"):
            pdf_segs, pdf_size = load_pdf_segments(str(pdf), page, cfg)
        logger.debug(f"PDF-Segmente: {len(pdf_segs)} (Seite {page}, Größe {pdf_size})")

        logger.step("SVG-Segmente laden")
        with logger.status("SVG analysieren"):
            svg_segs, svg_size = load_svg_segments(str(svg_path), cfg)
        logger.debug(f"SVG-Segmente: {len(svg_segs)} (Größe {svg_size})")

        if not svg_segs:
            raise ValueError("SVG enthält keine Vektoren")

        logger.step("Modell kalibrieren")
        with logger.status("Kalibrierung läuft"):
            model = calibrate(pdf_segs, svg_segs, pdf_size, svg_size, cfg)

        logger.step("Top-Linien auswählen")
        with logger.status("Linienauswahl"):
            pdf_lines, line_info = select_lines(pdf_segs, model, svg_segs, cfg)
        logger.debug(f"Ausgewählte Linien: {len(pdf_lines)}")

        logger.step("Linien abgleichen")
        with logger.status("Matching"):
            raw_matches = match_lines(pdf_lines, svg_segs, model, cfg)
        matches = _renumber_matches(raw_matches)

        extra_warnings: List[str] = []
        if len(matches) < 5:
            extra_warnings.append(f"Nur {len(matches)} Linien verfügbar – verbleibende Slots bleiben leer.")
        if any(match.svg_seg is None for match in matches):
            extra_warnings.append("Mindestens eine Linie konnte keinem SVG-Segment zugeordnet werden.")

        logger.step("Overlays & Bericht schreiben")
        with logger.status("SVG-Overlay"):
            overlay_svg = Path(write_svg_overlay(str(svg_path), str(outdir), str(pdf), page, model, matches))
        with logger.status("PDF-Overlay"):
            overlay_pdf = Path(write_pdf_overlay(str(pdf), page, str(outdir), matches))
        with logger.status("CSV-Bericht"):
            report_csv = Path(write_report_csv(str(outdir), str(pdf), page, model, matches))

        outputs = {
            "SVG": overlay_svg,
            "PDF": overlay_pdf,
            "CSV": report_csv,
        }

        _summarize(logger, model, matches, outputs, line_info, extra_warnings)
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
