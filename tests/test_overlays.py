from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import fitz
import pytest
from lxml import etree

from pdfsvg_calibrator.cli import HARDCODED_DEFAULTS, _alignment_diagnostics
from pdfsvg_calibrator.overlays import (
    write_pdf_overlay,
    write_report_csv,
    write_svg_overlay,
)
from pdfsvg_calibrator.types import Match, Model, Segment


def _make_model() -> Model:
    return Model(
        rot_deg=0,
        sx=1.1,
        sy=0.9,
        tx=5.0,
        ty=-3.0,
        score=0.87,
        rmse=0.12,
        p95=0.25,
        median=0.08,
    )


def _make_matches() -> list[Match]:
    return [
        Match(
            id=1,
            axis="H",
            pdf_seg=Segment(0.0, 10.0, 120.0, 10.0),
            svg_seg=Segment(12.0, 7.0, 144.0, 7.0),
            cost=0.01,
            confidence=0.92,
            pdf_len=120.0,
            svg_len=132.0,
            rel_error=0.0,
            pass01=1,
        ),
        Match(
            id=2,
            axis="V",
            pdf_seg=Segment(30.0, 0.0, 30.0, 150.0),
            svg_seg=Segment(38.0, -4.0, 38.0, 131.0),
            cost=0.02,
            confidence=0.85,
            pdf_len=150.0,
            svg_len=135.0,
            rel_error=0.0,
            pass01=1,
        ),
        Match(
            id=3,
            axis="H",
            pdf_seg=Segment(0.0, 60.0, 110.0, 60.0),
            svg_seg=None,
            cost=0.5,
            confidence=0.1,
            pdf_len=110.0,
            svg_len=None,
            rel_error=None,
            pass01=None,
        ),
        Match(
            id=4,
            axis="H",
            pdf_seg=Segment(0.0, 90.0, 140.0, 90.0),
            svg_seg=Segment(5.0, 88.0, 159.0, 88.0),
            cost=0.04,
            confidence=0.22,
            pdf_len=140.0,
            svg_len=154.0,
            rel_error=0.0,
            pass01=0,
        ),
        Match(
            id=5,
            axis="V",
            pdf_seg=Segment(80.0, -10.0, 80.0, 140.0),
            svg_seg=Segment(93.0, -15.0, 93.0, 111.0),
            cost=0.03,
            confidence=0.65,
            pdf_len=150.0,
            svg_len=126.0,
            rel_error=0.0,
            pass01=1,
        ),
    ]


@pytest.fixture()
def sample_pdf(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    doc.new_page(width=200, height=200)
    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture()
def sample_svg(tmp_path: Path) -> Path:
    svg_path = tmp_path / "sample.svg"
    svg_path.write_text(
        """
        <svg xmlns='http://www.w3.org/2000/svg' width='200' height='200'>
            <rect x='0' y='0' width='200' height='200' fill='white' stroke='#000'/>
        </svg>
        """.strip()
    )
    return svg_path


def test_write_svg_overlay(sample_svg: Path, sample_pdf: Path, tmp_path: Path) -> None:
    model = _make_model()
    alignment = _alignment_diagnostics(model, (200.0, 200.0), HARDCODED_DEFAULTS)
    out_path = write_svg_overlay(
        str(sample_svg),
        str(tmp_path),
        str(sample_pdf),
        page_index=0,
        model=model,
        matches=_make_matches(),
        alignment=alignment,
    )
    assert os.path.exists(out_path)

    tree = etree.parse(out_path)
    root = tree.getroot()
    ns = root.nsmap.get(None)
    if ns:
        group = root.find(f".//{{{ns}}}g[@id='CHECK_LINES']")
        assert group is not None
        line_tag = f"{{{ns}}}line"
        label_tag = f"{{{ns}}}g"
        text_tag = f"{{{ns}}}text"
    else:
        group = root.find(".//g[@id='CHECK_LINES']")
        assert group is not None
        line_tag = "line"
        label_tag = "g"
        text_tag = "text"

    lines = group.findall(line_tag)
    assert len(lines) == 10

    labels = [
        node
        for node in group
        if node.tag == label_tag and node.get("id", "").startswith("label_")
    ]
    assert len(labels) == 5
    texts = [child.text for node in labels for child in node if child.tag == text_tag]
    assert any(text == "3 (no match)" for text in texts)

    meta_group = (
        group.find(f"{label_tag}[@id='CHECK_METADATA']")
        if ns
        else group.find("g[@id='CHECK_METADATA']")
    )
    if meta_group is None:
        meta_group = next((node for node in group if getattr(node, "attrib", {}).get("id") == "CHECK_METADATA"), None)
    assert meta_group is not None
    meta_texts = [
        child.text
        for child in meta_group
        if child.tag == text_tag and child.text is not None
    ]
    assert any("rot=" in text for text in meta_texts)
    assert any("|t|=" in text for text in meta_texts)


def test_write_pdf_overlay(sample_pdf: Path, tmp_path: Path) -> None:
    model = _make_model()
    alignment = _alignment_diagnostics(model, (200.0, 200.0), HARDCODED_DEFAULTS)
    out_path = write_pdf_overlay(
        str(sample_pdf),
        page_index=0,
        outdir=str(tmp_path),
        model=model,
        matches=_make_matches(),
        alignment=alignment,
    )
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0

    doc = fitz.open(out_path)
    assert doc.page_count == 1
    text = doc[0].get_text()
    assert "rot=" in text
    assert "Bounds:" in text
    doc.close()


def test_write_report_csv(sample_pdf: Path, tmp_path: Path) -> None:
    model = _make_model()
    alignment = _alignment_diagnostics(model, (200.0, 200.0), HARDCODED_DEFAULTS)
    out_path = write_report_csv(
        str(tmp_path),
        str(sample_pdf),
        page_index=0,
        model=model,
        matches=_make_matches(),
        alignment=alignment,
    )
    assert os.path.exists(out_path)

    with open(out_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames
        assert header == [
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
            "calibration_notes",
            "flip_xy",
            "shift_abs",
            "shift_tol_px",
            "shift_tol_source",
            "scale_tol_rel",
            "bounds",
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
        rows = list(reader)
    assert len(rows) == 5

    assert all(row["calibration_notes"] == "" for row in rows)

    for row in rows:
        assert row["flip_xy"] == "none"
        assert row["shift_abs"] != "-"
        assert row["bounds"].startswith("Bounds:")

    row1 = next(row for row in rows if row["id"] == "1")
    assert pytest.approx(float(row1["ratio"])) == 1.1
    assert row1["axis_scale_expected"] == "1.1000"

    row3 = next(row for row in rows if row["id"] == "3")
    assert row3["svg_len"] == ""
    assert row3["ratio"] == ""
    assert row3["pass01"] == ""
    assert row3["notes"] == "no match"

    row4 = next(row for row in rows if row["id"] == "4")
    assert "low confidence" in row4["notes"]
