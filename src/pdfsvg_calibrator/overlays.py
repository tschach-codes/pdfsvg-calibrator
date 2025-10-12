import os, csv
from typing import List
from .types import Match, Model

def write_svg_overlay(svg_path: str, outdir: str, pdf_path: str, page: int, model: Model, matches: List[Match]):
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    out_svg = os.path.join(outdir, f"{base}_p{page:03d}_overlay_lines.svg")
    os.makedirs(outdir, exist_ok=True)
    with open(out_svg, "w", encoding="utf-8") as f:
        f.write(f"<!-- placeholder overlay for {svg_path} -->\n")
    return out_svg

def write_pdf_overlay(pdf_path: str, page: int, outdir: str, matches: List[Match]):
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    out_pdf = os.path.join(outdir, f"{base}_p{page:03d}_overlay_lines.pdf")
    os.makedirs(outdir, exist_ok=True)
    with open(out_pdf, "wb") as f:
        f.write(b"%PDF-1.4\n% placeholder overlay\n%%EOF")
    return out_pdf

def write_report_csv(outdir: str, pdf_path: str, page: int, model: Model, matches: List[Match], cfg: dict):
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    out_csv = os.path.join(outdir, f"{base}_p{page:03d}_check.csv")
    os.makedirs(outdir, exist_ok=True)
    rows=[]
    rows.append({
        "pdf_file": pdf_path, "page_index": page, "rot_deg": model.rot_deg,
        "sx": model.sx, "sy": model.sy, "tx": model.tx, "ty": model.ty,
        "score": model.score, "rmse": model.rmse, "p95": model.p95, "median": model.median
    })
    for m in matches:
        rows.append({
            "id": m.id, "pdf_len": m.pdf_len, "svg_len": m.svg_len or "",
            "rel_error": m.rel_error or "", "pass01": m.pass01 or ""
        })
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return out_csv
