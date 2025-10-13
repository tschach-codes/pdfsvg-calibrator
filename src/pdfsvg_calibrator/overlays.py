from __future__ import annotations

import csv
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import fitz
from lxml import etree

from .types import Match, Model

SVG_RED = "#F44336"
SVG_BLUE = "#2979FF"
LABEL_BG = "#FFFFFF"
LABEL_STROKE = "#212121"


def _apply_model(model: Model, x: float, y: float) -> Tuple[float, float]:
    rot = model.rot_deg % 360
    if rot == 0:
        rx, ry = x, y
    elif rot == 180:
        rx, ry = -x, -y
    elif rot == 90:
        rx, ry = -y, x
    elif rot == 270:
        rx, ry = y, -x
    else:
        raise ValueError(f"Unsupported rotation: {model.rot_deg}")
    return model.sx * rx + model.tx, model.sy * ry + model.ty


def _format_float(value: Optional[float], precision: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    fmt = f"{{:.{precision}f}}"
    return fmt.format(value)


def _ns_tag(root: etree._Element, tag: str) -> str:
    ns = root.nsmap.get(None)
    if ns:
        return f"{{{ns}}}{tag}"
    return tag


def _midpoint(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float]:
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _normalize_matches(matches: Sequence[Match]) -> Dict[int, Match]:
    return {match.id: match for match in matches}


def _pdf_hex_to_rgb(color: str) -> Tuple[float, float, float]:
    color = color.lstrip("#")
    if len(color) != 6:
        raise ValueError("Expected 6-digit hex color")
    r = int(color[0:2], 16) / 255.0
    g = int(color[2:4], 16) / 255.0
    b = int(color[4:6], 16) / 255.0
    return r, g, b


def _compute_ratio(pdf_len: Optional[float], svg_len: Optional[float]) -> Optional[float]:
    if pdf_len is None or svg_len is None:
        return None
    if math.isclose(pdf_len, 0.0, abs_tol=1e-12):
        return None
    return svg_len / pdf_len


def _expected_scale(axis: Optional[str], model: Model) -> Optional[float]:
    if axis == "H":
        return model.sx
    if axis == "V":
        return model.sy
    return None


def _compute_rel_error(ratio: Optional[float], expected: Optional[float]) -> Optional[float]:
    if ratio is None or expected is None:
        return None
    if math.isclose(expected, 0.0, abs_tol=1e-12):
        return None
    return abs(ratio - expected) / abs(expected)


def write_svg_overlay(
    svg_path: str,
    outdir: str,
    pdf_path: str,
    page_index: int,
    model: Model,
    matches: Sequence[Match],
) -> str:
    tree = etree.parse(svg_path)
    root = tree.getroot()
    os.makedirs(outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    out_svg = os.path.join(outdir, f"{base}_p{page_index:03d}_overlay_lines.svg")

    tag_g = _ns_tag(root, "g")
    tag_line = _ns_tag(root, "line")
    tag_rect = _ns_tag(root, "rect")
    tag_text = _ns_tag(root, "text")

    overlay_group = etree.Element(
        tag_g,
        id="CHECK_LINES",
        style="pointer-events:none;paint-order:stroke;font-family:'DejaVu Sans',Arial,sans-serif",
    )

    matches_by_id = _normalize_matches(matches)

    for idx in range(1, 6):
        match = matches_by_id.get(idx)
        if match is None:
            continue
        pdf_seg = match.pdf_seg
        x1, y1 = _apply_model(model, pdf_seg.x1, pdf_seg.y1)
        x2, y2 = _apply_model(model, pdf_seg.x2, pdf_seg.y2)

        etree.SubElement(
            overlay_group,
            tag_line,
            id=f"pdf_line_{idx}",
            x1=_format_float(x1, 3),
            y1=_format_float(y1, 3),
            x2=_format_float(x2, 3),
            y2=_format_float(y2, 3),
            stroke=SVG_RED,
            **{
                "stroke-width": "2.2",
                "opacity": "0.95",
                "vector-effect": "non-scaling-stroke",
                "fill": "none",
            },
        )

        svg_seg = match.svg_seg
        if svg_seg is not None:
            etree.SubElement(
                overlay_group,
                tag_line,
                id=f"svg_line_{idx}",
                x1=_format_float(svg_seg.x1, 3),
                y1=_format_float(svg_seg.y1, 3),
                x2=_format_float(svg_seg.x2, 3),
                y2=_format_float(svg_seg.y2, 3),
                stroke=SVG_BLUE,
                **{
                    "stroke-width": "2.2",
                    "opacity": "0.95",
                    "vector-effect": "non-scaling-stroke",
                    "fill": "none",
                },
            )
        else:
            etree.SubElement(
                overlay_group,
                tag_line,
                id=f"svg_line_{idx}",
                x1=_format_float(x1, 3),
                y1=_format_float(y1, 3),
                x2=_format_float(x2, 3),
                y2=_format_float(y2, 3),
                stroke=SVG_BLUE,
                **{
                    "stroke-width": "0.0",
                    "opacity": "0.0",
                    "vector-effect": "non-scaling-stroke",
                    "fill": "none",
                },
            )

        cx, cy = _midpoint(x1, y1, x2, y2)
        label = f"{idx}"
        if svg_seg is None:
            label = f"{idx} (no match)"
        label_group = etree.SubElement(
            overlay_group,
            tag_g,
            id=f"label_{idx}",
            transform=f"translate({_format_float(cx, 3)},{_format_float(cy, 3)})",
            style="vector-effect:non-scaling-stroke;paint-order:stroke",
        )

        text_len = max(len(label), 1)
        rect_width = 16 + text_len * 6
        rect_height = 18
        etree.SubElement(
            label_group,
            tag_rect,
            x=_format_float(-rect_width / 2, 3),
            y=_format_float(-rect_height / 2, 3),
            width=_format_float(rect_width, 3),
            height=_format_float(rect_height, 3),
            rx="4.0",
            ry="4.0",
            fill=LABEL_BG,
            stroke=LABEL_STROKE,
            **{
                "stroke-width": "0.8",
                "opacity": "0.96",
                "vector-effect": "non-scaling-stroke",
            },
        )

        text_node = etree.SubElement(
            label_group,
            tag_text,
            x="0",
            y="0",
            fill="#000000",
            **{
                "font-size": "12",
                "text-anchor": "middle",
                "dominant-baseline": "middle",
            },
        )
        text_node.text = label

    root.append(overlay_group)
    tree.write(out_svg, encoding="utf-8", xml_declaration=True)
    return out_svg


def write_pdf_overlay(
    pdf_path: str,
    page_index: int,
    outdir: str,
    matches: Sequence[Match],
) -> str:
    os.makedirs(outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    out_pdf = os.path.join(outdir, f"{base}_p{page_index:03d}_overlay_lines.pdf")

    src_doc = fitz.open(pdf_path)
    try:
        out_doc = fitz.open()
        out_doc.insert_pdf(src_doc, from_page=page_index, to_page=page_index)
        page = out_doc[0]
        shape = page.new_shape()

        color = _pdf_hex_to_rgb(SVG_RED)
        matches_by_id = _normalize_matches(matches)
        for idx in range(1, 6):
            match = matches_by_id.get(idx)
            if match is None:
                continue
            seg = match.pdf_seg
            shape.draw_line(fitz.Point(seg.x1, seg.y1), fitz.Point(seg.x2, seg.y2))
        shape.finish(color=color, width=1.8, closePath=False)

        for idx in range(1, 6):
            match = matches_by_id.get(idx)
            if match is None:
                continue
            seg = match.pdf_seg
            cx, cy = _midpoint(seg.x1, seg.y1, seg.x2, seg.y2)
            label = f"{idx}"
            if match.svg_seg is None:
                label = f"{idx} (no match)"
            text_len = max(len(label), 1)
            rect_width = 16 + text_len * 6
            rect_height = 18
            rect = fitz.Rect(
                cx - rect_width / 2,
                cy - rect_height / 2,
                cx + rect_width / 2,
                cy + rect_height / 2,
            )
            page.draw_rect(
                rect,
                color=_pdf_hex_to_rgb(LABEL_STROKE),
                fill=_pdf_hex_to_rgb(LABEL_BG),
                width=0.7,
                overlay=True,
            )
            page.insert_textbox(
                rect,
                label,
                fontsize=10,
                fontname="helv",
                align=fitz.TEXT_ALIGN_CENTER,
                color=(0.0, 0.0, 0.0),
            )

        out_doc.save(out_pdf)
    finally:
        src_doc.close()
        try:
            out_doc.close()
        except Exception:
            pass
    return out_pdf


def write_report_csv(
    outdir: str,
    basename_or_pdf: str,
    page_index: int,
    model: Model,
    matches: Sequence[Match],
) -> str:
    os.makedirs(outdir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(basename_or_pdf))[0]
    csv_path = os.path.join(outdir, f"{base_name}_p{page_index:03d}_check.csv")

    fieldnames = [
        "pdf_file",
        "page_index",
        "rot_deg",
        "sx",
        "sy",
        "tx",
        "ty",
        "score",
        "rmse",
        "p95",
        "median",
        "id",
        "axis",
        "pdf_len",
        "svg_len",
        "ratio",
        "axis_scale_expected",
        "rel_error",
        "pass01",
        "confidence",
        "notes",
    ]

    matches_by_id = _normalize_matches(matches)

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(1, 6):
            match = matches_by_id.get(idx)
            axis = match.axis if match is not None else ""
            pdf_len = match.pdf_len if match is not None else None
            svg_len = match.svg_len if match is not None else None
            ratio = _compute_ratio(pdf_len, svg_len)
            expected = _expected_scale(axis, model)
            rel_error = _compute_rel_error(ratio, expected)

            notes: List[str] = []
            if match is None:
                notes.append("missing match entry")
            elif match.svg_seg is None:
                notes.append("no match")
            elif match.confidence is not None and match.confidence < 0.3:
                notes.append("low confidence")

            writer.writerow(
                {
                    "pdf_file": os.path.basename(basename_or_pdf),
                    "page_index": page_index,
                    "rot_deg": model.rot_deg,
                    "sx": _format_float(model.sx, 5),
                    "sy": _format_float(model.sy, 5),
                    "tx": _format_float(model.tx, 3),
                    "ty": _format_float(model.ty, 3),
                    "score": _format_float(model.score, 4),
                    "rmse": _format_float(model.rmse, 4),
                    "p95": _format_float(model.p95, 4),
                    "median": _format_float(model.median, 4),
                    "id": idx,
                    "axis": axis,
                    "pdf_len": _format_float(pdf_len, 3),
                    "svg_len": _format_float(svg_len, 3),
                    "ratio": _format_float(ratio, 4),
                    "axis_scale_expected": _format_float(expected, 4),
                    "rel_error": _format_float(rel_error, 4),
                    "pass01": "" if match is None or match.pass01 is None else int(match.pass01),
                    "confidence":
                        _format_float(match.confidence, 3)
                        if match is not None and match.confidence is not None
                        else "",
                    "notes": "; ".join(notes),
                }
            )
    return csv_path
