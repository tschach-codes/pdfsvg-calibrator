from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET


def _find_pdf_to_svg_executable() -> list[str]:
    """Return the first available PDF->SVG executable in PATH."""
    candidates = ["pdftosvg", "pdftosvg.exe", "pdf2svg", "pdf2svg.exe"]
    for name in candidates:
        path = shutil.which(name)
        if path:
            return [path]
    return []


def _which(name: str) -> Optional[str]:
    """Return absolute executable path if available in PATH, else None."""
    return shutil.which(name)


def _is_pdftosvg_style(exe_path: str) -> bool:
    """Return True if the executable appears to be the pdftosvg CLI."""
    return os.path.basename(exe_path).lower().startswith("pdftosvg")


def _pdftosvg_actual_output_path(
    requested_out_svg: Path, page_number_1based: int
) -> Path:
    """Return the output path that pdftosvg.NET actually writes."""

    return requested_out_svg.with_name(
        f"{requested_out_svg.stem}-{page_number_1based}.svg"
    )


def _build_single_page_command(
    exe_path: str, pdf_path: Path, out_svg: Path, page_index: int
) -> list[str]:
    """Build the converter command for a single page."""
    page_number_1based = page_index + 1
    if _is_pdftosvg_style(exe_path):
        return [
            exe_path,
            str(pdf_path),
            str(out_svg),
            "--non-interactive",
            "--pages",
            str(page_number_1based),
        ]
    return [
        exe_path,
        str(pdf_path),
        str(out_svg),
        str(page_number_1based),
    ]


def _run_converter(cmd: list[str], verbose: bool = False) -> None:
    if verbose:
        print("[pdfsvg] Full command:", cmd)
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        if verbose:
            print("[pdfsvg] Failed to execute converter:", exc)
        raise

    stdout = proc.stdout.decode("utf-8", errors="replace") if proc.stdout else ""
    stderr = proc.stderr.decode("utf-8", errors="replace") if proc.stderr else ""

    if verbose:
        print("[pdfsvg] return code:", proc.returncode)
        print("[pdfsvg] STDOUT:", stdout)
        print("[pdfsvg] STDERR:", stderr)

    if proc.returncode != 0:
        raise RuntimeError(
            "Could not convert PDF to SVG via pdftosvg or pdf2svg. "
            "Make sure at least one of them is installed and in PATH."
        )

    return


def _run_pdftosvg(
    pdftosvg_exe: str,
    pdf_path: Path,
    out_svg: Path,
    page_index: int,
    *,
    verbose: bool = False,
) -> None:
    """
    Run pdftosvg (pdftosvg.net style).
    Command shape:
        pdftosvg input.pdf output.svg --pages pageIndex1Based
    """
    cmd = _build_single_page_command(pdftosvg_exe, pdf_path, out_svg, page_index)
    _run_converter(cmd, verbose=verbose)


def _run_pdf2svg(
    pdf2svg_exe: str,
    pdf_path: Path,
    out_svg: Path,
    page_index: int,
    *,
    verbose: bool = False,
) -> None:
    """
    Run pdf2svg.
    pdf2svg is 1-based for the page argument.
    Command shape:
        pdf2svg input.pdf output.svg pageIndex1Based
    """
    cmd = _build_single_page_command(pdf2svg_exe, pdf_path, out_svg, page_index)
    _run_converter(cmd, verbose=verbose)


def _svg_contains_text(svg_path: Path, verbose: bool = False) -> bool:
    """
    Returns True if the SVG contains text-like content.
    Namespace-agnostic; streaming parse; early exit on first hit.
    Detects <text>, <tspan>, and Inkscape flow text (<flowRoot>, <flowPara>).
    """
    TEXTY = {"text", "tspan", "flowRoot", "flowPara"}
    try:
        for _event, elem in ET.iterparse(str(svg_path), events=("start",)):
            tag = elem.tag
            if '}' in tag:
                tag = tag.rsplit('}', 1)[-1]  # strip namespace
            if tag in TEXTY:
                return True
        if verbose:
            print(f"[pdfsvg] No text-like elements found in {svg_path.name}")
        return False
    except ET.ParseError as e:
        if verbose:
            print(f"[pdfsvg] SVG parse error in {svg_path.name}: {e}")
        # be conservative on parse errors
        return False


def export_pdf_page_to_svg(
    pdf_path: str | Path,
    out_svg: str | Path,
    page: int = 0,
    *,
    require_text: bool = False,
    verbose: bool = False,
) -> Path:
    """
    Convert one PDF page to an SVG with semantic fidelity (paths + text).
    Strategy:
      1. Try pdftosvg (preferred)
      2. Fall back to pdf2svg if necessary

    require_text: if True, only accept SVGs that contain text-like elements.
    """
    pdf_path = Path(pdf_path)
    out_svg = Path(out_svg)
    out_svg.parent.mkdir(parents=True, exist_ok=True)

    cmd_base = _find_pdf_to_svg_executable()
    if verbose:
        print("[pdfsvg] PATH seen by Python:", os.environ.get("PATH", ""))
        print(
            "[pdfsvg] Converter executable from lookup:",
            cmd_base[0] if cmd_base else None,
        )

    if not cmd_base:
        raise RuntimeError(
            "Could not convert PDF to SVG via pdftosvg or pdf2svg. "
            "Make sure at least one of them is installed and in PATH."
        )

    primary_exe = cmd_base[0]
    primary_is_pdftosvg = _is_pdftosvg_style(primary_exe)

    pdftosvg_exe = primary_exe if primary_is_pdftosvg else None
    pdf2svg_exe = primary_exe if not primary_is_pdftosvg else None

    if pdftosvg_exe is None:
        pdftosvg_exe = _which("pdftosvg") or _which("pdftosvg.exe")
    if pdf2svg_exe is None:
        pdf2svg_exe = _which("pdf2svg") or _which("pdf2svg.exe")

    last_exception: Optional[Exception] = None

    def _output_ready(svg_path: Path) -> bool:
        return svg_path.exists() and svg_path.stat().st_size > 0

    def _evaluate_success(
        converter_name: str, svg_display: str, text_detected: bool
    ) -> bool:
        if verbose:
            print(f"[pdfsvg] converter: {converter_name} -> {svg_display}")
            print(f"[pdfsvg] text-detected: {text_detected}")
        if not require_text:
            if verbose:
                decision = "success" if text_detected else "warned"
                print(f"[pdfsvg] accept: {decision}")
            return True
        if text_detected:
            if verbose:
                print("[pdfsvg] accept: success")
            return True
        if verbose:
            print("[pdfsvg] accept: raised")
        return False

    def _attempt_converter(run_func, converter_name: str, svg_path: Path) -> Optional[Path]:
        nonlocal last_exception
        try:
            run_func()
        except Exception as exc:  # noqa: BLE001 - surface original exception
            last_exception = exc
            if verbose:
                print(f"[pdfsvg] {converter_name} failed:", exc)
            return None

        if not _output_ready(svg_path):
            if verbose:
                print(f"[pdfsvg] {converter_name} produced no output")
            return None

        check_text = require_text or verbose
        text_detected = (
            _svg_contains_text(svg_path, verbose=verbose) if check_text else False
        )
        if _evaluate_success(converter_name, svg_path.name, text_detected):
            return svg_path
        return None

    if pdftosvg_exe is not None:
        page_number_1based = page + 1
        actual_svg = _pdftosvg_actual_output_path(out_svg, page_number_1based)
        produced_svg = _attempt_converter(
            lambda: _run_pdftosvg(
                pdftosvg_exe,
                pdf_path,
                out_svg,
                page_index=page,
                verbose=verbose,
            ),
            "pdftosvg",
            actual_svg,
        )
        if produced_svg is not None:
            return produced_svg

    if pdf2svg_exe is not None and (pdf2svg_exe != pdftosvg_exe or pdftosvg_exe is None):
        produced_svg = _attempt_converter(
            lambda: _run_pdf2svg(
                pdf2svg_exe,
                pdf_path,
                out_svg,
                page_index=page,
                verbose=verbose,
            ),
            "pdf2svg",
            out_svg,
        )
        if produced_svg is not None:
            return produced_svg

    if verbose and last_exception is not None:
        print("[pdfsvg] Final conversion failure:", last_exception)
        if isinstance(last_exception, subprocess.CalledProcessError) and last_exception.stderr:
            print("[pdfsvg] STDERR:", last_exception.stderr)

    raise RuntimeError(
        "Could not convert PDF to SVG via pdftosvg or pdf2svg. "
        "Make sure at least one of them is installed and in PATH."
    )


def ensure_svg_for_pdf(
    pdf_path: str | Path,
    svg_path_opt: Optional[str | Path],
    outdir: str | Path,
    page: int = 0,
    *,
    require_text: bool = False,
    verbose: bool = False,
) -> Path:
    """
    Main entry point for the pipeline.
    - If the caller already gave us an SVG path via --svg and it exists, return it.
    - Else: export the requested page of the PDF to SVG using export_pdf_page_to_svg().
      Output file pattern: <outdir>/<pdf_stem>_p{page}.svg
    """
    pdf_path = Path(pdf_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # User-supplied SVG?
    if svg_path_opt:
        svg_path = Path(svg_path_opt)
        if svg_path.exists():
            return svg_path
        else:
            raise FileNotFoundError(f"--svg was provided, but file not found: {svg_path}")

    # Auto-generate
    auto_svg = outdir / f"{pdf_path.stem}_p{page}.svg"
    produced_svg = export_pdf_page_to_svg(
        pdf_path,
        auto_svg,
        page=page,
        require_text=require_text,
        verbose=verbose,
    )
    return produced_svg
