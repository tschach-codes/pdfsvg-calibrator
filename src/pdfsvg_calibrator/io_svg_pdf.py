from typing import List, Tuple
import os
import fitz
from .types import Segment
from .svg_path import parse_svg_segments

def load_pdf_segments(pdf_path: str, page_index: int, cfg: dict)->Tuple[List[Segment], Tuple[float,float]]:
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_index)
    drawings = page.get_drawings()
    segs=[]
    for dr in drawings:
        path = dr.get("path", None)
        if path:
            for sp in path:
                cur=None
                for cmd in sp:
                    if cmd[0]=='m': cur=(cmd[1],cmd[2])
                    elif cmd[0]=='l' and cur is not None:
                        x0,y0=cur; x,y=cmd[1],cmd[2]
                        if (x0!=x) or (y0!=y):
                            segs.append(Segment(x0,y0,x,y))
                        cur=(x,y)
                    elif cmd[0]=='re':
                        x,y,w,h = cmd[1],cmd[2],cmd[3],cmd[4]
                        if w>0 and h>0:
                            segs.extend([
                                Segment(x,y,x+w,y), Segment(x+w,y,x+w,y+h),
                                Segment(x+w,y+h,x,y+h), Segment(x,y+h,x,y)
                            ])
        else:
            cur=None
            for it in dr.get("items", []):
                op = it[0]
                if op=='m': cur=it[1]
                elif op=='l' and cur is not None:
                    (x0,y0)=cur; (x,y)=it[1]
                    if (x0!=x) or (y0!=y):
                        segs.append(Segment(x0,y0,x,y))
                    cur=(x,y)
                elif op=='re':
                    x,y,w,h = it[1]
                    if w>0 and h>0:
                        segs.extend([
                            Segment(x,y,x+w,y), Segment(x+w,y,x+w,y+h),
                            Segment(x+w,y+h,x,y+h), Segment(x,y+h,x,y)
                        ])
    w = page.rect.width; h = page.rect.height
    doc.close()
    return segs, (w,h)

def convert_pdf_to_svg_if_needed(pdf_path: str, page_index: int, outdir: str)->str:
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    svg_out = os.path.join(outdir, f"{base}_p{page_index:03d}.svg")
    os.makedirs(outdir, exist_ok=True)
    if not os.path.exists(svg_out):
        raise FileNotFoundError(
            "No SVG found. Please export a vector SVG to:\n"
            f"  {svg_out}\n"
            "Hook an external converter if desired."
        )
    return svg_out

def load_svg_segments(svg_path: str, cfg: dict):
    segs = parse_svg_segments(svg_path)
    size = (0.0, 0.0)  # could parse viewBox later
    return segs, size
