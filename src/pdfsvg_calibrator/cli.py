import typer
from rich import print
from .config import load_config
from .io_svg_pdf import load_pdf_segments, convert_pdf_to_svg_if_needed, load_svg_segments
from .fit_model import calibrate
from .match_verify import select_lines, match_lines
from .overlays import write_svg_overlay, write_pdf_overlay, write_report_csv

app = typer.Typer(help="PDF→SVG calibration & verification")

@app.command("run")
def run(pdf: str, page: int = 0, config: str = "configs/default.yaml", outdir: str = "out"):
    cfg = load_config(config)
    print(f"[bold green]Calibrating[/bold green] {pdf} (page {page})")
    svg_path = convert_pdf_to_svg_if_needed(pdf, page, outdir)
    pdf_segs, pdf_size = load_pdf_segments(pdf, page, cfg)
    svg_segs, svg_size = load_svg_segments(svg_path, cfg)

    model = calibrate(pdf_segs, svg_segs, pdf_size, svg_size, cfg)
    pdf_lines = select_lines(pdf_segs, model, cfg)
    matches = match_lines(pdf_lines, svg_segs, model, cfg)

    os_svg = write_svg_overlay(svg_path, outdir, pdf, page, model, matches)
    os_pdf = write_pdf_overlay(pdf, page, outdir, matches)
    os_csv = write_report_csv(outdir, pdf, page, model, matches, cfg)
    print(f"[green]Done.[/green] Outputs:\n- {os_svg}\n- {os_pdf}\n- {os_csv}")

if __name__ == "__main__":
    app()
