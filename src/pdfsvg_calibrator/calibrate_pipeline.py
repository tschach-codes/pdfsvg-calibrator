from __future__ import annotations

from typing import Any, Dict, Mapping

from .raster_align import coarse_raster_align
from .dimension_scale_integration import refine_metric_scale


def _clone_config(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(config, Mapping):
        return {}
    cloned: Dict[str, Any] = dict(config)
    orientation = config.get("orientation")
    if isinstance(orientation, Mapping):
        cloned["orientation"] = dict(orientation)
    return cloned


def calibrate_pdf_svg_preprocess(pdf_bytes: bytes, svg_bytes: bytes, config: dict) -> dict:
    """
    High-level Preprocessing Pipeline (neue schlanke Variante):

    1. coarse_alignment = coarse_raster_align(pdf_bytes, svg_bytes, config)
       - liefert rotation_deg, flip_horizontal, tx_px, ty_px, sx_seed, sy_seed, score

    2. metric = refine_metric_scale(svg_bytes, coarse_alignment, config)
       - ruft die bestehende Maßketten-/DimScale-Logik auf (OHNE sie umzuschreiben)
       - gibt pixels_per_meter o.ä. zurück

    3. Falls metric["ok"] == False:
       - OPTIONAL: eskaliere die Auflösung / DPI für coarse_raster_align ein zweites Mal, falls dein orientation-Code das kennt,
         und ruf refine_metric_scale nochmal auf. (Wenn die existierende DimScale-Logik das nicht braucht, kannst du einfach metric so lassen.)

    4. Rückgabe:
       {
         "coarse_alignment": coarse_alignment,
         "metric_scale": metric,
       }

    Diese Funktion ersetzt NICHT die bisherige ausführliche Kalibrier-/Reporting-Pipeline.
    Sie dient als Vorverarbeitungsschritt fürs nächste Skript ("Filterung"),
    um SVG-Geometrie in einen global einheitlichen Raum zu legen.
    """
    print("[preprocess] starte coarse_raster_align")
    cfg = _clone_config(config)
    coarse_alignment = coarse_raster_align(pdf_bytes, svg_bytes, cfg)
    print(
        "[preprocess] coarse_alignment: "
        f"rotation={coarse_alignment.get('rotation_deg')} deg, "
        f"flip={'H' if coarse_alignment.get('flip_horizontal') else 'none'}, "
        f"tx={coarse_alignment.get('tx_px')}, ty={coarse_alignment.get('ty_px')}, "
        f"sx_seed={coarse_alignment.get('sx_seed')}, sy_seed={coarse_alignment.get('sy_seed')}, "
        f"score={coarse_alignment.get('score')}"
    )

    print("[preprocess] starte refine_metric_scale")
    metric = refine_metric_scale(svg_bytes, coarse_alignment, cfg)
    print(
        "[preprocess] metric_scale: "
        f"ok={metric.get('ok')}, pixels_per_meter={metric.get('pixels_per_meter')}"
    )

    if not metric.get("ok"):
        orientation_cfg = cfg.get("orientation")
        raster_size = None
        if isinstance(orientation_cfg, Mapping):
            raw_size = orientation_cfg.get("raster_size")
            if isinstance(raw_size, (int, float)):
                raster_size = int(raw_size)
        if raster_size is None:
            raster_size = 512
        upgraded_size = max(raster_size * 2, raster_size + 256)
        if upgraded_size != raster_size:
            print(
                "[preprocess] metric ok==False, erhöhe raster_size für zweiten Versuch: "
                f"{raster_size} -> {upgraded_size}"
            )
            upgraded_cfg = _clone_config(cfg)
            orientation_cfg = upgraded_cfg.setdefault("orientation", {})
            if isinstance(orientation_cfg, dict):
                orientation_cfg["raster_size"] = upgraded_size
            print("[preprocess] starte coarse_raster_align (zweiter Versuch)")
            coarse_alignment = coarse_raster_align(pdf_bytes, svg_bytes, upgraded_cfg)
            print(
                "[preprocess] coarse_alignment (zweiter Versuch): "
                f"rotation={coarse_alignment.get('rotation_deg')} deg, "
                f"flip={'H' if coarse_alignment.get('flip_horizontal') else 'none'}, "
                f"tx={coarse_alignment.get('tx_px')}, ty={coarse_alignment.get('ty_px')}, "
                f"sx_seed={coarse_alignment.get('sx_seed')}, sy_seed={coarse_alignment.get('sy_seed')}, "
                f"score={coarse_alignment.get('score')}"
            )
            print("[preprocess] starte refine_metric_scale (zweiter Versuch)")
            metric = refine_metric_scale(svg_bytes, coarse_alignment, upgraded_cfg)
            print(
                "[preprocess] metric_scale (zweiter Versuch): "
                f"ok={metric.get('ok')}, pixels_per_meter={metric.get('pixels_per_meter')}"
            )

    return {
        "coarse_alignment": coarse_alignment,
        "metric_scale": metric,
    }
