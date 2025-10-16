# pdfsvg-calibrator

## TL;DR
`pdfsvg-calibrator` gleicht eine PDF-Seite mit einer passenden SVG-Version ab, findet fünf markante Linienpaare und zeigt Ihnen farbige Overlays plus eine Prüfliste, damit Druckereien, Planer:innen oder Werkstätten schnell sehen, ob die Größen zueinander passen.

## Installation
1. **Python vorbereiten**
   - Benötigt wird Python 3.9 oder neuer.
   - Abhängigkeiten: `numpy`, `lxml`, `PyMuPDF` (wird als `fitz` importiert), `PyYAML`, `typer`, `rich`.
2. **Virtuelle Umgebung (empfohlen)**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
3. **Paket installieren**
   - Nur Tool verwenden:
     ```bash
     pip install .
     ```
   - Entwicklung & Tests:
     ```bash
     pip install -e .[dev]
     ```

## 60-Sekunden-Schnellstart
1. **PDF bereitstellen** – Legen Sie Ihre Zeichnung oder Ihren Plan als einzelne PDF-Datei ab. Ohne `--svg` exportiert das Tool die gewählte Seite automatisch als SVG nach `--outdir`.
2. **Befehl ausführen** – Ersetzen Sie die Platzhalter und starten Sie den Lauf:
   ```bash
   pdfsvg-calibrate run plan.pdf --page 0 --outdir out
   ```
   Optional: `--config configs/default.yaml` (wird sonst automatisch geladen).
3. **Ergebnis prüfen** – Im Ordner `out/` finden Sie drei Dateien:
   - `plan_p000_overlay_lines.svg` – SVG mit roten (PDF) und blauen (SVG) Linien.
   - `plan_p000_overlay_lines.pdf` – gleiches Overlay als PDF.
   - `plan_p000_check.csv` – Tabelle mit Modellwerten, pro Linie Länge, Fehler und PASS/FAIL.

### Was zeigt die Konsole?
Nach einem erfolgreichen Lauf erscheint eine Zusammenfassung mit:
- Modellparametern (`rot`, `sx`, `sy`, `tx`, `ty`), erkannte Spiegelungen und Qualitätskennzahlen (Score, RMSE, P95, Median).
- Einer zusätzlichen Zeile mit dem Gesamtoffset `|t|` und einem tolerierten Grenzwert sowie dem mittleren Skalierungsfaktor.
- Einer 5-zeiligen Tabelle: ID, Achse (H/V), Längen, Verhältnis, relativer Fehler, PASS/FAIL-Flag (`Pass01`), Vertrauenswert und Hinweise (z. B. „no match“).
- Einer Liste der erzeugten Dateien (`SVG`, `PDF`, `CSV`) sowie etwaiger Warnungen (z. B. zu wenig Segmente).

## Output-Dateien
- `*_overlay_lines.svg` – Zur Kontrolle in der CAD- oder Layout-Anwendung laden; Layer `CHECK_LINES` enthält nummerierte Linien und Hinweisboxen.
- `*_overlay_lines.pdf` – Für Stakeholder ohne SVG-Viewer, gleiche Markierungen wie im SVG.
- `*_check.csv` – Kann in Excel geöffnet werden; enthält Modellwerte (Zeile 1) und darunter pro Linie ID, Achse, Längen, Verhältnis, relativer Fehler, PASS/FAIL und Confidence.

## Troubleshooting
- **Keine Vektoren gefunden** – Enthält die PDF-Seite nur Rastergrafiken, erzeugt der automatische Export zwar eine SVG-Datei, die Analyse findet jedoch keine Linien und bricht mit „SVG enthält keine Vektoren“ ab.
- **„Zu wenig Segmente“** – Erhöhen Sie die Zeichnungsqualität oder lockern Sie `straight_max_dev_rel`/`angle_tol_deg` in der Konfiguration, damit mehr Linien zugelassen werden. Alternativ prüfen Sie, ob im PDF Linien wirklich exakt horizontal/vertikal sind.
- **180°-Rotation** – Das Tool testet 0° und 180°. Wenn das Ergebnis gespiegelt wirkt, aktivieren Sie weitere Rotationen über `rot_degrees` (z. B. `[0, 90, 180, 270]`).
- **Y-Achse wirkt vertauscht** – Negative Skalen (`sx < 0`/`sy < 0`) werden als „flip“ in der Zusammenfassung gemeldet. Passen Sie ggf. die SVG-Export-Einstellungen (z. B. Ursprung oben links) an.

## FAQ
- **Warum nur horizontale und vertikale Linien?** – Sie sind in technischen Zeichnungen stabil, erlauben eine eindeutige Skalenermittlung und lassen sich robust vergleichen.
- **Wie streng ist die ±0,01-Grenze?** – `verify.tol_rel` steuert, wann eine Linie als PASS gilt. Standard: 1 % Unterschied zwischen erwarteter und gemessener Länge.
- **Kann ich 90°/270° aktivieren?** – Ja, setzen Sie in Ihrer YAML `rot_degrees: [0, 90, 180, 270]`. Dadurch prüft der Algorithmus auch Hochformat-Lagen oder gedrehte Scans.
- **Was passiert, wenn weniger als fünf Linien gefunden werden?** – Die Tabelle wird mit Platzhaltern aufgefüllt; fehlende Linien erscheinen als „no match“ und werden im CSV markiert.

## Support & Mitmachen
Fehler gefunden oder Erweiterungsidee? Eröffnen Sie bitte ein Issue oder einen Pull Request. Für Beiträge gelten die bestehenden Tests und Formatierungen (`pytest`, `ruff`, `black`, `mypy`).

## Lizenz
Siehe `LICENSE` im Repository.
