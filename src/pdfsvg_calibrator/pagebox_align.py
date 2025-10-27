from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Optional, Tuple, Union

import pypdfium2 as pdfium


@dataclass
class PageBoxInfo:
    width: float
    height: float


@dataclass
class SvgViewBoxInfo:
    min_x: float
    min_y: float
    width: float
    height: float


@dataclass
class PageBoxAlignmentResult:
    rotation_deg: float
    flip_x: bool
    flip_y: bool
    scale_x: float
    scale_y: float
    pdf_size: Tuple[float, float]
    svg_size: Tuple[float, float]
    svg_viewbox: SvgViewBoxInfo
    confidence: str                # "low", "medium"
    reason: str                    # human-readable hint


def _approx(a: float, b: float, rel: float = 0.05) -> bool:
    ma = max(abs(a), abs(b), 1e-9)
    return abs(a - b) <= rel * ma


def get_pdf_page_size(pdf_path: str, page_index: int = 0) -> PageBoxInfo:
    """
    Liest die Seitengröße in PDF-User-Units (typisch 1/72 inch).
    Wir nehmen page.get_rect(), das sollte (left,bottom,right,top) liefern.
    """
    doc = pdfium.PdfDocument(pdf_path)
    page = doc[page_index]

    left, bottom, right, top = page.get_rect()
    w = right - left
    h = top - bottom
    return PageBoxInfo(width=float(w), height=float(h))


def parse_svg_viewbox(svg_path: str) -> Optional[SvgViewBoxInfo]:
    """
    Minimalistischer Parser fürs viewBox-Attribut im <svg>-Root.
    """
    try:
        with open(svg_path, "r", encoding="utf-8") as f:
            head = f.read(4096)
    except Exception:
        return None

    m = re.search(r'viewBox\s*=\s*"([^"]+)"', head, flags=re.IGNORECASE)
    if not m:
        return None
    parts = m.group(1).strip().split()
    if len(parts) != 4:
        return None

    try:
        min_x = float(parts[0])
        min_y = float(parts[1])
        w     = float(parts[2])
        h     = float(parts[3])
    except ValueError:
        return None

    return SvgViewBoxInfo(min_x=min_x, min_y=min_y, width=w, height=h)


def _estimate_rotation_flip(pdf_w, pdf_h, svg_w, svg_h) -> Tuple[float, bool, bool, str]:
    """
    Schätze grob rotation_deg & flips anhand Seitenverhältnissen.
    Rückgabe extra: textual_reason.
    """
    # Kandidat 0° (gleiches Seitenverhältnis)
    if _approx(svg_w, pdf_w) and _approx(svg_h, pdf_h):
        return 0.0, False, True, "matching aspect (no swap), SVG likely y-down vs PDF y-up"

    # Kandidat 90° (geswappte Seitenverhältnisse)
    if _approx(svg_w, pdf_h) and _approx(svg_h, pdf_w):
        return 90.0, False, True, "swapped aspect (width≈height), PDF probably rotated vs SVG"

    # Fallback
    return 0.0, False, True, "fallback orientation guess (uncertain)"


def _guess_confidence(pdf_w, pdf_h, svg_w, svg_h, reason: str) -> str:
    """
    Sehr grobe Confidence:
    - Wenn Seitenverhältnisse sauber matchen oder geswappt sind -> "medium"
    - Sonst -> "low"
    """
    # Seitenverhältnis vergleichen
    pdf_ar  = pdf_w / max(pdf_h, 1e-9)
    svg_ar  = svg_w / max(svg_h, 1e-9)
    swap_ar = pdf_h / max(pdf_w, 1e-9)

    if _approx(pdf_ar, svg_ar, rel=0.05) or _approx(pdf_ar, swap_ar, rel=0.05):
        return "medium"
    return "low"


def compute_pagebox_alignment(
    pdf_path: str,
    page_index: int,
    svg_path: str,
) -> Optional[PageBoxAlignmentResult]:
    """
    Liefert grobe globale Transformationshinweise nur auf Basis von:
      - PDF Seitenmaße in User Units
      - SVG viewBox size
    Übersetzt das in:
      - rotation_deg ∈ {0, 90} (heute)
      - flip_y fast immer True (SVG y-down vs PDF y-up)
      - scale_x, scale_y (wie müsste SVG skaliert werden um PDF Blattgröße zu treffen)
      - confidence & reason (diagnostischer Text)

    Kein tx/ty hier, nur globale Ausrichtung/Skalierung.
    """
    pagebox = get_pdf_page_size(pdf_path, page_index)
    svgbox  = parse_svg_viewbox(svg_path)
    if svgbox is None:
        return None

    pdf_w, pdf_h = pagebox.width, pagebox.height
    svg_w, svg_h = svgbox.width, svgbox.height

    rot_deg, flip_x, flip_y, reason = _estimate_rotation_flip(pdf_w, pdf_h, svg_w, svg_h)

    # Skalenfaktoren ableiten.
    # Bei 90°-Fall tauschen wir w/h.
    if abs(rot_deg - 90.0) < 1e-3:
        sx = (pdf_h / svg_w) if svg_w else 1.0
        sy = (pdf_w / svg_h) if svg_h else 1.0
    else:
        sx = (pdf_w / svg_w) if svg_w else 1.0
        sy = (pdf_h / svg_h) if svg_h else 1.0

    conf = _guess_confidence(pdf_w, pdf_h, svg_w, svg_h, reason)

    return PageBoxAlignmentResult(
        rotation_deg = rot_deg,
        flip_x       = flip_x,
        flip_y       = flip_y,
        scale_x      = sx,
        scale_y      = sy,
        pdf_size     = (pdf_w, pdf_h),
        svg_size     = (svg_w, svg_h),
        svg_viewbox  = svgbox,
        confidence   = conf,
        reason       = reason,
    )


def explain_alignment_reason(result: PageBoxAlignmentResult) -> str:
    """Gibt einen menschenlesbaren Hinweis zur Orientierung der Heuristik."""
    segments = []
    base_reason = result.reason.strip()
    if base_reason:
        segments.append(base_reason)

    orientation_bits = [f"rotation {result.rotation_deg:.0f}°"]
    if result.flip_x:
        orientation_bits.append("flip_x")
    if result.flip_y:
        orientation_bits.append("flip_y")
    segments.append("orientation: " + ", ".join(orientation_bits))

    scale_bits = []
    if math.isfinite(result.scale_x):
        scale_bits.append(f"scale_x≈{result.scale_x:.3f}")
    if math.isfinite(result.scale_y):
        scale_bits.append(f"scale_y≈{result.scale_y:.3f}")
    if scale_bits:
        segments.append("scales: " + ", ".join(scale_bits))

    return "; ".join(segments)


def confidence_label(value: Union[str, PageBoxAlignmentResult]) -> str:
    """Liefert ein standardisiertes Confidence-Label für Log-Ausgaben."""
    if isinstance(value, PageBoxAlignmentResult):
        confidence_value = value.confidence
    else:
        confidence_value = value

    confidence_value = confidence_value.strip().lower()
    if not confidence_value:
        confidence_value = "unknown"
    return f"pagebox:{confidence_value}"
