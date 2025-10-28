from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional


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


def _run_converter(cmd: list[str], verbose: bool = False) -> subprocess.CompletedProcess:
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

    return proc


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


def _svg_contains_text(svg_path: Path) -> bool:
    """
    Quick semantic validation: is there any <text ...> in the SVG?
    We don't fully parse XML here, we just do a cheap substring check.
    If substring is present, we consider it 'has text semantics'.
    """
    data = svg_path.read_bytes()
    # do a binary lowercase check for robustness:
    lower = data.lower()
    return b"<text" in lower


def export_pdf_page_to_svg(
    pdf_path: str | Path,
    out_svg: str | Path,
    page: int = 0,
    *,
    verbose: bool = False,
) -> Path:
    """
    Convert one PDF page to an SVG with semantic fidelity (paths + text).
    Strategy:
      1. Try pdftosvg (preferred)
         - If successful, check for <text>. If yes -> done.
      2. Otherwise, or if no <text> was found, try pdf2svg.
         - If this succeeds, replace out_svg with that result.
      3. If both fail, raise RuntimeError.

    pdf_path: input PDF file path
    out_svg: desired output .svg file path (will be overwritten)
    page:    0-based page index in the PDF
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

    # 1. Try pdftosvg first, if available
    pdftosvg_ok = False
    last_exception: Optional[Exception] = None
    if pdftosvg_exe is not None:
        try:
            _run_pdftosvg(
                pdftosvg_exe,
                pdf_path,
                out_svg,
                page_index=page,
                verbose=verbose,
            )
            if out_svg.exists() and out_svg.stat().st_size > 0:
                pdftosvg_ok = True
        except Exception as exc:
            pdftosvg_ok = False
            last_exception = exc
            if verbose:
                print("[pdfsvg] pdftosvg failed:", exc)

    # If pdftosvg ran AND SVG has <text>, accept it immediately.
    if pdftosvg_ok and _svg_contains_text(out_svg):
        return out_svg

    # Otherwise fallback: try pdf2svg (this may overwrite/replace the file)
    if pdf2svg_exe is not None and (pdf2svg_exe != pdftosvg_exe or not pdftosvg_ok):
        try:
            _run_pdf2svg(
                pdf2svg_exe,
                pdf_path,
                out_svg,
                page_index=page,
                verbose=verbose,
            )
            if out_svg.exists() and out_svg.stat().st_size > 0:
                # even if there's still no <text>, we accept this as last resort
                return out_svg
        except Exception as exc:
            last_exception = exc
            if verbose:
                print("[pdfsvg] pdf2svg failed:", exc)

    # If we get here, we failed both routes in a meaningful way
    if verbose and last_exception is not None:
        print("[pdfsvg] Final conversion failure:", last_exception)
        if isinstance(last_exception, subprocess.CalledProcessError):
            if last_exception.stderr:
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
    export_pdf_page_to_svg(pdf_path, auto_svg, page=page, verbose=verbose)
    return auto_svg
