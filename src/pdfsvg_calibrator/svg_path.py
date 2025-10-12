from typing import List
from lxml import etree as ET
from .types import Segment

def parse_svg_segments(svg_path: str) -> List[Segment]:
    # Minimal placeholder: extract <line> only. (Path-approx comes later.)
    segs: List[Segment] = []
    parser = ET.XMLParser(huge_tree=True, recover=True, remove_blank_text=False)
    tree = ET.parse(svg_path, parser=parser)
    root = tree.getroot()
    ns = root.tag.split('}')[0].strip('{') if '}' in root.tag else ''
    def tagname(t): return f"{{{ns}}}{t}" if ns else t
    for el in root.iter(tagname('line')):
        try:
            x1 = float(el.get('x1') or 0); y1 = float(el.get('y1') or 0)
            x2 = float(el.get('x2') or 0); y2 = float(el.get('y2') or 0)
            if (x1!=x2) or (y1!=y2):
                segs.append(Segment(x1,y1,x2,y2))
        except:
            pass
    return segs
