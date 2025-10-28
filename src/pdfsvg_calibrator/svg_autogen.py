from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Literal, Optional

import fitz  # PyMuPDF
from lxml import etree

SvgMethod = Literal["auto", "inkscape", "pdftocairo", "pdf2svg", "mutool", "pymupdf"]


def exe_on_path(name: str) -> Optional[str]:
    p = shutil.which(name)
    return p


def detect_backends(cfg: dict) -> dict:
    """Return availability dict for inkscape/pdftocairo/pdf2svg/mutool."""
    bins = cfg.get("svg_export", {}).get("bin", {})
    cand = {
        "inkscape": bins.get("inkscape") or "inkscape",
        "pdftocairo": bins.get("pdftocairo") or "pdftocairo",
        "pdf2svg": bins.get("pdf2svg") or "pdf2svg",
        "mutool": bins.get("mutool") or "mutool",
    }
    return {k: (exe_on_path(v) or None) for k, v in cand.items()}


def _inkscape_pdf_to_svg(inkscape: str, pdf: Path, out_svg: Path, page: int, prefer_poppler: bool, timeout: int):
    """Try multiple CLI variants for Inkscape (CLI changed across versions)."""
    # Try poppler importer if requested
    base_cmds = []
    if prefer_poppler:
        base_cmds.append([inkscape, "--pdf-poppler"])
        base_cmds.append([inkscape, "--pdf-import-use-poppler"])
    base_cmds.append([inkscape])  # vanilla as fallback
    # page numbering in inkscape is 1-based
    page1 = page + 1
    candidates = []
    for base in base_cmds:
        # Newer Inkscape:
        candidates.append(
            base
            + [
                "--export-type=svg",
                f"--export-filename={str(out_svg)}",
                f"--pdf-page={page1}",
                str(pdf),
            ]
        )
        # Older style:
        candidates.append(
            base
            + [
                f"--export-plain-svg={str(out_svg)}",
                str(pdf),
            ]
        )
    for cmd in candidates:
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
            if out_svg.exists() and out_svg.stat().st_size > 0:
                return
        except Exception:
            continue
    raise RuntimeError("Inkscape conversion failed for all tried flags.")


def _pdftocairo_pdf_to_svg(pdftocairo: str, pdf: Path, out_svg: Path, page: int, timeout: int):
    # Poppler; -svg with page range f=l
    page1 = page + 1
    cmd = [pdftocairo, "-svg", "-f", str(page1), "-l", str(page1), str(pdf), str(out_svg)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    # pdftocairo writes out with suffix; normalize if needed
    if not out_svg.exists():
        # pdftocairo may create basename-1.svg, so rename
        produced = out_svg.with_name(out_svg.stem + "-1.svg")
        if produced.exists():
            produced.rename(out_svg)


def _pdf2svg_pdf_to_svg(pdf2svg: str, pdf: Path, out_svg: Path, page: int, timeout: int):
    page1 = page + 1
    cmd = [pdf2svg, str(pdf), str(out_svg), str(page1)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)


def _mutool_pdf_to_svg(mutool: str, pdf: Path, out_svg: Path, page: int, timeout: int):
    page1 = page + 1
    cmd = [mutool, "draw", "-o", str(out_svg), "-F", "svg", str(pdf), str(page1)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)


def _pymupdf_pdf_to_svg(pdf: Path, out_svg: Path, page: int, text_as_path: bool):
    with fitz.open(pdf) as doc:
        if page < 0 or page >= len(doc):
            raise ValueError(f"page {page} out of range")
        p = doc[page]
        svg = p.get_svg_image(text_as_path=text_as_path)  # keep text as text if False
        out_svg.write_bytes(svg.encode("utf-8"))


def _postprocess_layers(svg_path: Path, keep_layers: bool):
    """
    Light-touch postprocess:
    - If produced by Inkscape/Poppler, layers may already be <g inkscape:groupmode="layer" ...>.
    - If not, leave as-is (we do not fabricate fake layers).
    - Ensure XML has proper xmlns + preserve style attributes.
    """
    try:
        xml = etree.parse(str(svg_path))
        root = xml.getroot()
        # no-op unless we need to normalize namespaces or add helpful data-* attributes
        xml.write(str(svg_path), encoding="utf-8", xml_declaration=True)
    except Exception:
        pass


def export_pdf_page_to_svg(
    pdf_path: str | Path,
    out_svg: str | Path,
    method: SvgMethod = "auto",
    page: int = 0,
    keep_text: bool = True,
    keep_layers: bool = True,
    cfg: Optional[dict] = None,
) -> Path:
    """
    Convert one PDF page to SVG, prioritizing semantic richness (text, paths, layers).
    Returns the path to the written SVG.
    """
    pdf = Path(pdf_path)
    out_svg = Path(out_svg)
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    timeout = int((cfg or {}).get("svg_export", {}).get("timeout_sec", 120))
    prefer_poppler = bool((cfg or {}).get("svg_export", {}).get("prefer_poppler", True))
    avail = detect_backends(cfg or {})
    chosen = method
    order = ["inkscape", "pdftocairo", "pdf2svg", "mutool", "pymupdf"]
    if method == "auto":
        # pick first available (except PyMuPDF which is always available)
        chosen = next((m for m in order if avail.get(m) or m == "pymupdf"), "pymupdf")
    if chosen == "inkscape":
        if not avail["inkscape"]:
            raise RuntimeError("Inkscape not available on PATH")
        _inkscape_pdf_to_svg(avail["inkscape"], pdf, out_svg, page, prefer_poppler, timeout)
    elif chosen == "pdftocairo":
        if not avail["pdftocairo"]:
            raise RuntimeError("pdftocairo not available on PATH")
        _pdftocairo_pdf_to_svg(avail["pdftocairo"], pdf, out_svg, page, timeout)
    elif chosen == "pdf2svg":
        if not avail["pdf2svg"]:
            raise RuntimeError("pdf2svg not available on PATH")
        _pdf2svg_pdf_to_svg(avail["pdf2svg"], pdf, out_svg, page, timeout)
    elif chosen == "mutool":
        if not avail["mutool"]:
            raise RuntimeError("mutool not available on PATH")
        _mutool_pdf_to_svg(avail["mutool"], pdf, out_svg, page, timeout)
    elif chosen == "pymupdf":
        _pymupdf_pdf_to_svg(pdf, out_svg, page, text_as_path=not keep_text)
    else:
        raise ValueError(f"Unknown method {method}")
    _postprocess_layers(out_svg, keep_layers=keep_layers)
    return out_svg


def ensure_svg_for_pdf(
    pdf_path: str | Path,
    svg_path_opt: Optional[str | Path],
    outdir: str | Path,
    cfg: dict,
) -> Path:
    """
    If svg_path_opt exists -> return it.
    Else render page cfg['svg_export']['page'] using export_pdf_page_to_svg according to cfg.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if svg_path_opt:
        p = Path(svg_path_opt)
        if p.exists():
            return p
        raise FileNotFoundError(f"SVG not found: {p}")
    se = cfg.get("svg_export", {})
    page = int(se.get("page", 0))
    suffix = se.get("out_suffix", "_p{page}.svg").format(page=page)
    out_svg = outdir / (Path(pdf_path).stem + suffix)
    method = se.get("method", "auto")
    keep_text = bool(se.get("keep_text", True))
    keep_layers = bool(se.get("keep_layers", True))
    return export_pdf_page_to_svg(
        pdf_path,
        out_svg,
        method=method,
        page=page,
        keep_text=keep_text,
        keep_layers=keep_layers,
        cfg=cfg,
    )
