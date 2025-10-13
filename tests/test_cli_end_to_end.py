from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import fitz
import pytest


def _write_pdf(path: Path, segments: list[tuple[float, float, float, float]]) -> None:
    doc = fitz.open()
    try:
        page = doc.new_page(width=200, height=200)
        thickness = 1.5
        for x1, y1, x2, y2 in segments:
            if abs(y1 - y2) < 1e-6:
                top = min(y1, y2) - thickness * 0.5
                bottom = max(y1, y2) + thickness * 0.5
                left = min(x1, x2)
                right = max(x1, x2)
                page.draw_rect(fitz.Rect(left, top, right, bottom), color=(0, 0, 0))
            elif abs(x1 - x2) < 1e-6:
                left = min(x1, x2) - thickness * 0.5
                right = max(x1, x2) + thickness * 0.5
                top = min(y1, y2)
                bottom = max(y1, y2)
                page.draw_rect(fitz.Rect(left, top, right, bottom), color=(0, 0, 0))
            else:
                left = min(x1, x2)
                right = max(x1, x2)
                top = min(y1, y2)
                bottom = max(y1, y2)
                page.draw_rect(fitz.Rect(left, top, right, bottom), color=(0, 0, 0))
        doc.save(path)
    finally:
        doc.close()


def _write_svg(path: Path, segments: list[tuple[float, float, float, float]]) -> None:
    lines = [
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="black" stroke-width="1" />'
        for x1, y1, x2, y2 in segments
    ]
    svg = "\n".join(lines)
    content = (
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"200\" height=\"200\" viewBox=\"0 0 200 200\">"
        + svg
        + "</svg>"
    )
    path.write_text(content, encoding="utf-8")


def _run_cli(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    pythonpath = env.get("PYTHONPATH", "")
    new_path = str(src_path)
    if pythonpath:
        new_path = os.pathsep.join([new_path, pythonpath])
    env["PYTHONPATH"] = new_path
    return subprocess.run(
        [sys.executable, "-m", "pdfsvg_calibrator.cli", *args],
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
    )


def _write_config(path: Path, extra: str = "") -> None:
    base = textwrap.dedent(
        """
        rot_degrees: [0]
        angle_tol_deg: 6.0
        curve_tol_rel: 0.001
        straight_max_dev_rel: 0.0015
        straight_max_angle_spread_deg: 3.0
        merge:
          gap_max_rel: 0.01
          off_tol_rel: 0.003
        grid_cell_rel: 0.02
        chamfer:
          sigma_rel: 0.004
          hard_mul: 3.0
        ransac:
          iters: 60
          refine_scale_step: 0.002
          refine_trans_px: 2.0
        verify:
          pick_k: 5
          diversity_rel: 0.05
          radius_px: 60
          dir_tol_deg: 6.0
          tol_rel: 0.02
        neighbors:
          use: true
          radius_rel: 0.05
          dt: 0.02
          dtheta_deg: 6.0
          rho_soft: 0.25
          penalty_miss: 1.2
          penalty_empty: 3.0
        cost_weights:
          endpoint: 0.5
          midpoint: 0.3
          direction: 0.2
          neighbors: 0.1
        sampling:
          step_rel: 0.05
          max_points: 1500
        """
    ).strip()
    if extra:
        base += "\n" + extra.strip()
    path.write_text(base + "\n", encoding="utf-8")


def test_cli_happy_path(tmp_path: Path) -> None:
    pdf_path = tmp_path / "calibration.pdf"
    svg_path = tmp_path / "calibration.svg"
    outdir = tmp_path / "out"
    cfg_path = tmp_path / "config.yaml"

    segments = [
        (20.0, 20.0, 180.0, 20.0),
        (20.0, 60.0, 180.0, 60.0),
        (20.0, 100.0, 180.0, 100.0),
        (40.0, 20.0, 40.0, 180.0),
        (80.0, 20.0, 80.0, 180.0),
        (120.0, 20.0, 120.0, 180.0),
    ]

    _write_pdf(pdf_path, segments)
    _write_svg(svg_path, segments)
    _write_config(cfg_path)

    result = _run_cli(
        [
            "run",
            str(pdf_path),
            "--page",
            "0",
            "--config",
            str(cfg_path),
            "--outdir",
            str(outdir),
            "--svg",
            str(svg_path),
            "--rng-seed",
            "7",
        ],
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    stdout = result.stdout
    assert "sx=" in stdout
    assert "overlay_lines.svg" in stdout
    assert "overlay_lines.pdf" in stdout
    assert "_check.csv" in stdout

    assert (outdir / "calibration_p000_overlay_lines.svg").exists()
    assert (outdir / "calibration_p000_overlay_lines.pdf").exists()
    assert (outdir / "calibration_p000_check.csv").exists()


def test_cli_raster_svg(tmp_path: Path) -> None:
    pdf_path = tmp_path / "empty.pdf"
    svg_path = tmp_path / "empty.svg"
    outdir = tmp_path / "out"
    cfg_path = tmp_path / "config.yaml"

    _write_pdf(pdf_path, [(10.0, 10.0, 190.0, 10.0)])
    svg_path.write_text(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"200\" height=\"200\"><image href=\"foo.png\" x=\"0\" y=\"0\" width=\"200\" height=\"200\" /></svg>",
        encoding="utf-8",
    )
    _write_config(cfg_path)

    result = _run_cli(
        [
            "run",
            str(pdf_path),
            "--page",
            "0",
            "--config",
            str(cfg_path),
            "--outdir",
            str(outdir),
            "--svg",
            str(svg_path),
        ],
        cwd=tmp_path,
    )

    assert result.returncode == 2
    combined = result.stdout + result.stderr
    assert "SVG enthält keine Vektoren" in combined


def test_cli_ransac_failure(tmp_path: Path) -> None:
    pdf_path = tmp_path / "invalid.pdf"
    svg_path = tmp_path / "invalid.svg"
    outdir = tmp_path / "out"
    cfg_path = tmp_path / "config.yaml"

    horizontal_segments = [
        (20.0, 20.0, 180.0, 20.0),
        (20.0, 60.0, 180.0, 60.0),
        (20.0, 100.0, 180.0, 100.0),
    ]

    _write_pdf(pdf_path, [])
    _write_svg(svg_path, horizontal_segments)
    _write_config(cfg_path)

    result = _run_cli(
        [
            "run",
            str(pdf_path),
            "--page",
            "0",
            "--config",
            str(cfg_path),
            "--outdir",
            str(outdir),
            "--svg",
            str(svg_path),
        ],
        cwd=tmp_path,
    )

    assert result.returncode == 2
    combined = result.stdout + result.stderr
    assert "Keine PDF-Segmente" in combined
