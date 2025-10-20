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

## Wie es funktioniert
Die Kalibrierung läuft in vier Stufen ab:
1. **Orientierung & Flip-Hypothesen** – PDF- und SVG-Segmente werden in ein 512²-Raster projiziert, Flip-Kombinationen und die Rotationsliste (`rot_degrees`) getestet.【F:src/pdfsvg_calibrator/orientation.py†L246-L329】
2. **Phase-Correlation & Translation-Seed** – Für jede Hypothese wird per FFT der Versatz zwischen beiden Rasterbildern geschätzt; die beste Kombination liefert `tx₀`/`ty₀`.【F:src/pdfsvg_calibrator/orientation.py†L179-L275】
3. **Skalen-Seed & Schranken** – Aus den Seitengrößen werden `sx₀`/`sy₀` abgeleitet und Clamp-Fenster für die Suche gesetzt.【F:src/pdfsvg_calibrator/calibrate.py†L91-L195】
4. **Lokale Verfeinerung** – RANSAC + Chamfer-Grid optimieren das Modell innerhalb der Seeds, bevor Matching und Reporting starten.【F:src/pdfsvg_calibrator/calibrate.py†L139-L355】【F:src/pdfsvg_calibrator/fit_model.py†L186-L316】

Mehr Hintergründe finden Sie in `docs/HOW_IT_WORKS.md`.

## Konfiguration & empfohlene Defaults
`configs/default.yaml` bündelt alle Schalter. Für robuste Läufe (<10 s) haben sich folgende Werte bewährt:

| Schlüssel | Empfehlung | Warum |
| --- | --- | --- |
| `rot_degrees` | `[0, 180]` (ggf. 90/270 ergänzen) | Reduziert Hypothesen, solange keine Querformat-Pläne erwartet werden.【F:configs/default.yaml†L1-L1】【F:src/pdfsvg_calibrator/orientation.py†L246-L329】 |
| `orientation.enabled` & `use_phase_correlation` | `true` | Liefert zuverlässige Seeds für Translation und Flip.【F:src/pdfsvg_calibrator/calibrate.py†L66-L124】【F:src/pdfsvg_calibrator/orientation.py†L179-L275】 |
| `orientation.raster_size` | `512` | Größere Werte nur bei extrem feinen Plänen nötig; sonst guter Kompromiss aus Präzision/Laufzeit.【F:src/pdfsvg_calibrator/orientation.py†L179-L329】 |
| `refine.scale_max_dev_rel` / `trans_max_dev_px` | `0.02` / `8.0` | Enges Fenster sorgt für schnelle Konvergenz, kann bei verzogenen Exports erweitert werden.【F:src/pdfsvg_calibrator/calibrate.py†L141-L195】 |
| `refine.max_samples` | `1500` | Begrenzung der Samplingpunkte hält Chamfer-Auswertungen flott.【F:src/pdfsvg_calibrator/calibrate.py†L157-L215】 |
| `grid.initial_cell_rel` / `final_cell_rel` | `0.05` / `0.02` | Zweistufiges Grid: zuerst grob cachen, dann fein validieren.【F:src/pdfsvg_calibrator/calibrate.py†L216-L220】【F:src/pdfsvg_calibrator/fit_model.py†L203-L316】 |
| `verify.tol_rel` | `0.01` | PASS/FAIL-Kriterium für ±1 % Längenfehler; bei strengeren Plänen senken.【F:configs/default.yaml†L21-L26】【F:src/pdfsvg_calibrator/match_verify.py†L680-L703】 |

Weitere relative Toleranzen (`curve_tol_rel`, `straight_max_dev_rel`, …) skalieren automatisch mit der PDF-Diagonale; große Pläne benötigen daher keine manuellen Anpassungen.【F:configs/default.yaml†L3-L9】【F:src/pdfsvg_calibrator/io_svg_pdf.py†L235-L308】 Die vorbereiteten `merge.*`-Parameter helfen beim Zusammenführen kollinearer Segmente, falls Ihre Pre-Processing-Kette das nutzt.【F:configs/default.yaml†L10-L18】

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

## Performance-Tipps
- **Kollineare Segmente mergen** – Zusammenhängende Linien vorab vereinen (oder `merge.*` in der YAML vorbereiten), damit Chamfer/RANSAC weniger Duplikate prüfen müssen.【F:configs/default.yaml†L10-L18】【F:src/pdfsvg_calibrator/fit_model.py†L186-L316】
- **PDF-Diagonale als Maßstab nutzen** – Relative Toleranzen (`curve_tol_rel`, `straight_max_dev_rel`, `sampling.step_rel`) skalieren automatisch mit der Seitengröße; belassen Sie sie bei großen Plänen unverändert.【F:src/pdfsvg_calibrator/io_svg_pdf.py†L235-L308】【F:src/pdfsvg_calibrator/calibrate.py†L141-L175】
- **Chamfer-Grid zweistufig fahren** – `grid.initial_cell_rel` (grob) und `grid.final_cell_rel` (fein) halten die Laufzeit unter 10 s bei stabiler Genauigkeit.【F:src/pdfsvg_calibrator/calibrate.py†L216-L220】【F:src/pdfsvg_calibrator/fit_model.py†L203-L316】
- **Rotationen gezielt aktivieren** – Bleiben Sie bei `[0,180]`, solange keine gedrehten Seiten erwartet werden; zusätzliche Rotationen verlängern die Laufzeit proportional.【F:src/pdfsvg_calibrator/orientation.py†L246-L329】

## Troubleshooting
| Symptom | Ursache | Lösung |
| --- | --- | --- |
| Ergebnis gespiegelt | Falscher Flip/Rotation gewann die Orientierung | `rot_degrees` um 90°-Schritte erweitern, Orientation aktiv lassen, ggf. Phase-Korrelation deaktivieren, wenn die Zeichnung hochsymmetrisch ist.【F:src/pdfsvg_calibrator/orientation.py†L246-L329】 |
| Riesiger Translationsoffset | Phase-Korrelation traf Nebenmaximum | `orientation.raster_size` erhöhen, `refine.trans_max_dev_px` auf 20–30 setzen und Quality-Gate-Fallback zulassen.【F:src/pdfsvg_calibrator/orientation.py†L179-L275】【F:src/pdfsvg_calibrator/calibrate.py†L141-L355】 |
| Lauf dauert >10 s | Zu viele Samples/Hypothesen | `sampling.max_points` auf ~1200 reduzieren, `rot_degrees` straffen, Segmente vorab mergen.【F:src/pdfsvg_calibrator/calibrate.py†L199-L215】【F:configs/default.yaml†L1-L18】 |
| Plan wirkt verrauscht | Viele kurze/gebrochene Segmente | `straight_max_dev_rel` etwas erhöhen oder externe Glättung anwenden; `verify.radius_px` moderat anheben.【F:src/pdfsvg_calibrator/io_svg_pdf.py†L235-L308】【F:src/pdfsvg_calibrator/match_verify.py†L557-L706】 |

## FAQ
- **Warum nur horizontale und vertikale Linien?** – Sie sind in technischen Zeichnungen stabil, erlauben eine eindeutige Skalenermittlung und lassen sich robust vergleichen.
- **Wie streng ist die ±0,01-Grenze?** – `verify.tol_rel` steuert, wann eine Linie als PASS gilt. Standard: 1 % Unterschied zwischen erwarteter und gemessener Länge.
- **Kann ich 90°/270° aktivieren?** – Ja, setzen Sie in Ihrer YAML `rot_degrees: [0, 90, 180, 270]`. Dadurch prüft der Algorithmus auch Hochformat-Lagen oder gedrehte Scans.
- **Was passiert, wenn weniger als fünf Linien gefunden werden?** – Die Tabelle wird mit Platzhaltern aufgefüllt; fehlende Linien erscheinen als „no match“ und werden im CSV markiert.

## Support & Mitmachen
Fehler gefunden oder Erweiterungsidee? Eröffnen Sie bitte ein Issue oder einen Pull Request. Für Beiträge gelten die bestehenden Tests und Formatierungen (`pytest`, `ruff`, `black`, `mypy`).

## Lizenz
Siehe `LICENSE` im Repository.
