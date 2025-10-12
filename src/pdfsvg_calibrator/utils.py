import math

def seg_len(x1,y1,x2,y2)->float:
    return math.hypot(x2-x1, y2-y1)

def angle_deg(x1,y1,x2,y2)->float:
    return math.degrees(math.atan2(y2-y1, x2-x1))
