from __future__ import annotations
import subprocess
import shutil
from pathlib import Path
from typing import Optional


def _which(name: str) -> Optional[str]:
    """Return absolute executable path if available in PATH, else None."""
    return shutil.which(name)


def _run_pdftosvg(pdftosvg_exe: str, pdf_path: Path, out_svg: Path, page_index: int) -> None:
    """
    Run pdftosvg (pdftosvg.net style).
    Assumption: page index is 0-based for this tool.
    Command shape:
        pdftosvg input.pdf output.svg pageIndex0Based
    """
    cmd = [pdftosvg_exe, str(pdf_path), str(out_svg), str(page_index)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _run_pdf2svg(pdf2svg_exe: str, pdf_path: Path, out_svg: Path, page_index: int) -> None:
    """
    Run pdf2svg.
    pdf2svg is 1-based for the page argument.
    Command shape:
        pdf2svg input.pdf output.svg pageIndex1Based
    """
    cmd = [pdf2svg_exe, str(pdf_path), str(out_svg), str(page_index + 1)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


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

    pdftosvg_exe = _which("pdftosvg")
    pdf2svg_exe  = _which("pdf2svg")

    # 1. Try pdftosvg first, if available
    pdftosvg_ok = False
    if pdftosvg_exe is not None:
        try:
            _run_pdftosvg(pdftosvg_exe, pdf_path, out_svg, page_index=page)
            if out_svg.exists() and out_svg.stat().st_size > 0:
                pdftosvg_ok = True
        except Exception:
            pdftosvg_ok = False

    # If pdftosvg ran AND SVG has <text>, accept it immediately.
    if pdftosvg_ok and _svg_contains_text(out_svg):
        return out_svg

    # Otherwise fallback: try pdf2svg (this may overwrite/replace the file)
    if pdf2svg_exe is not None:
        try:
            _run_pdf2svg(pdf2svg_exe, pdf_path, out_svg, page_index=page)
            if out_svg.exists() and out_svg.stat().st_size > 0:
                # even if there's still no <text>, we accept this as last resort
                return out_svg
        except Exception:
            pass

    # If we get here, we failed both routes in a meaningful way
    raise RuntimeError(
        "Could not convert PDF to SVG via pdftosvg or pdf2svg. "
        "Make sure at least one of them is installed and in PATH."
    )


def ensure_svg_for_pdf(
    pdf_path: str | Path,
    svg_path_opt: Optional[str | Path],
    outdir: str | Path,
    page: int = 0,
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
    export_pdf_page_to_svg(pdf_path, auto_svg, page=page)
    return auto_svg
