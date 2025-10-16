# Wie pdfsvg-calibrator arbeitet

## Überblick
Das Tool sucht eine affine Abbildung der Form `X = sx * x + tx`, `Y = sy * y + ty`. Diese vier Skalierungs- und Verschiebungswerte werden für jede geprüfte Rotationshypothese (standardmäßig 0° und 180°) ermittelt, indem horizontale und vertikale Linien aus PDF und SVG miteinander verglichen werden.【F:src/pdfsvg_calibrator/fit_model.py†L230-L316】【F:src/pdfsvg_calibrator/cli.py†L230-L305】

## Pipeline von A→Z
1. **Konfiguration laden** – `cli.run()` lädt YAML + Defaults, setzt optionalen RNG-Seed und erzwingt Nachbarschaftsnutzung.【F:src/pdfsvg_calibrator/cli.py†L323-L355】
2. **Eingaben prüfen & SVG exportieren** – Ohne expliziten SVG-Pfad wird die gewünschte Seite automatisch via PyMuPDF nach `outdir` exportiert.【F:src/pdfsvg_calibrator/io_svg_pdf.py†L316-L332】【F:src/pdfsvg_calibrator/cli.py†L356-L374】
3. **PDF analysieren** – `load_pdf_segments()` extrahiert Vektorzeichnungen, approximiert Kurven, filtert nicht-lineare Pfade via TLS und liefert Segmentliste + Seitengröße.【F:src/pdfsvg_calibrator/io_svg_pdf.py†L235-L310】
4. **SVG analysieren** – `load_svg_segments()` parst die Datei transformationsbewusst, liefert Segmentliste + Maße.【F:src/pdfsvg_calibrator/io_svg_pdf.py†L378-L392】【F:src/pdfsvg_calibrator/svg_path.py†L14-L190】
5. **Kalibrierung** – `fit_model.calibrate()` baut ein Chamfer-Grid, sampelt PDF-Linien, führt RANSAC über H/V-Segmente aus und verfeinert die besten Kandidaten.【F:src/pdfsvg_calibrator/fit_model.py†L186-L316】
6. **Linienauswahl** – `match_verify.select_lines()` bewertet PDF-Segmente, erzwingt Achsenvielfalt und wählt maximal fünf Kandidaten.【F:src/pdfsvg_calibrator/match_verify.py†L334-L548】
7. **Matching & Prüfung** – `match_verify.match_lines()` transformiert die PDF-Linien, baut Nachbarschaftssignaturen, berechnet Kosten und löst ein ungarisches Matching inklusive PASS/FAIL-Klassifizierung.【F:src/pdfsvg_calibrator/match_verify.py†L551-L705】
8. **Bericht** – `write_svg_overlay()`, `write_pdf_overlay()`, `write_report_csv()` erzeugen Visualisierungen und CSV; `_summarize()` gibt die Konsoleausgabe aus.【F:src/pdfsvg_calibrator/overlays.py†L92-L205】【F:src/pdfsvg_calibrator/cli.py†L258-L412】

### Sequenzdiagramm-ähnliche Schrittfolge
- `cli.run()`
  - `_load_config_with_defaults()` → zusammengeführte Konfiguration (`dict`).
  - `convert_pdf_to_svg_if_needed()` (falls nötig) → Pfad zur erwarteten SVG.
  - `load_pdf_segments()` → `(pdf_segments, (width, height))`.
  - `load_svg_segments()` → `(svg_segments, (width, height))`.
  - `fit_model.calibrate()` → `Model(rot_deg, sx, sy, tx, ty, score, rmse, p95, median)`.
  - `select_lines()` → `(pdf_lines, info)`.
  - `match_lines()` → `List[Match]`.
  - `write_*()` → Pfade zu Overlay/CSV; `_summarize()` meldet Status.

### Datenflüsse
- **Segmentlisten** – `load_pdf_segments()` und `load_svg_segments()` liefern `List[Segment]`, die in `fit_model.calibrate()` und später `select_lines()`/`match_lines()` weiterverwendet werden.【F:src/pdfsvg_calibrator/io_svg_pdf.py†L235-L392】【F:src/pdfsvg_calibrator/types.py†L4-L27】
- **Modelldaten** – `Model` umfasst Rotation, Skalen, Translation und Qualitätsmetriken; wird in Matching, Overlays und CSV wiederverwendet.【F:src/pdfsvg_calibrator/types.py†L10-L27】【F:src/pdfsvg_calibrator/overlays.py†L19-L204】
- **Nachbarschaftssignaturen** – `neighbor_signature()` beschreibt angrenzende Segmente (`t`, `Δθ`, `ρ`) und speist `candidate_cost()` sowie Konfidenzberechnung.【F:src/pdfsvg_calibrator/match_verify.py†L171-L305】【F:src/pdfsvg_calibrator/match_verify.py†L685-L690】

## Datenmodelle
- `Segment(x1, y1, x2, y2)` – Repräsentiert eine Linie im Ursprungskoordinatensystem.【F:src/pdfsvg_calibrator/types.py†L4-L9】
- `Model(rot_deg, sx, sy, tx, ty, score, rmse, p95, median)` – Ergebnis der Kalibrierung inklusive Gütekriterien.【F:src/pdfsvg_calibrator/types.py†L10-L14】
- `Match(id, axis, pdf_seg, svg_seg, cost, confidence, pdf_len, svg_len, rel_error, pass01)` – Ergebnis einer Zeile in CSV/Overlays.【F:src/pdfsvg_calibrator/types.py†L16-L27】【F:src/pdfsvg_calibrator/overlays.py†L117-L205】

## Parameter- und Konstantenverzeichnis
### Konfigurierbare Parameter
| Parameter | Default | Bedeutung & Wirkung | Fundstelle & Hinweise |
| --- | --- | --- | --- |
| `rot_degrees` | `[0, 180]` | Liste der getesteten Rotationen. Zusätzliche Einträge erlauben 90°/270°-Hypothesen, erhöhen aber Laufzeit.【F:configs/default.yaml†L1-L1】 | RANSAC iteriert über jede Rotation, dreht PDF-Segmente vor dem Matching.【F:src/pdfsvg_calibrator/fit_model.py†L230-L238】 |
| `angle_tol_deg` | `6.0` | Maximaler Winkelabstand zu exakt horizontal/vertikal für die Achsklassifikation.【F:configs/default.yaml†L2-L2】 | Wird in `classify_hv()` für Kalibrierung und Matching genutzt – engere Werte erzwingen präzisere Linien, riskieren aber Ausschluss.【F:src/pdfsvg_calibrator/fit_model.py†L222-L238】【F:src/pdfsvg_calibrator/match_verify.py†L436-L587】 |
| `curve_tol_rel` | `0.001` | Relative Toleranz (bezogen auf PDF-Diagonale) zur Approximation von Kurven. In der YAML doppelt vorhanden; der spätere Eintrag überschreibt den ersten.【F:configs/default.yaml†L3-L8】 | Skaliert auf absolute Länge und steuert, wie fein Kurven unterteilt werden, bevor sie als Linien geprüft werden.【F:src/pdfsvg_calibrator/io_svg_pdf.py†L245-L259】 |
| `straight_max_dev_rel` | `0.0015` | Zulässige maximale Abweichung eines Punktes von der TLS-Linie (relativ zur PDF-Diagonale). Zweiter Eintrag überschreibt den ersten.【F:configs/default.yaml†L4-L9】 | Bestimmt, wie streng Linien als „gerade“ gelten; höhere Werte lassen mehr Kandidaten, senken aber Genauigkeit.【F:src/pdfsvg_calibrator/io_svg_pdf.py†L245-L259】 |
| `straight_max_angle_spread_deg` | `3.0` | Erlaubter Winkelfächer innerhalb einer Polyline, bevor sie verworfen wird.【F:configs/default.yaml†L5-L9】 | `_angle_spread_deg()` verwirft Linien mit größerer Streuung, um gebogene Pfade auszuschließen.【F:src/pdfsvg_calibrator/io_svg_pdf.py†L15-L58】 |
| `min_len_raw` | `6.0` | Mindestlänge in Rohkoordinaten (derzeit Platzhalter, z. B. für Vorfilter).【F:configs/default.yaml†L6-L6】 | Wird aktuell nur als Default geführt; künftige Erweiterungen können damit kurze Linien herausfiltern.【F:src/pdfsvg_calibrator/cli.py†L44-L78】 |
| `merge.gap_max_rel` | `0.01` | Maximaler relativer Lückenschluss bei Segment-Merges (Reserveschalter).【F:configs/default.yaml†L10-L11】 | Noch nicht aktiv genutzt; vorgesehen, um getrennte Linienstücke zusammenzuführen.【F:src/pdfsvg_calibrator/cli.py†L44-L78】 |
| `merge.off_tol_rel` | `0.003` | Zulässige Versatzabweichung für Merge-Operationen.【F:configs/default.yaml†L10-L12】 | Wie oben – derzeit keine Implementierung, bleibt für zukünftige Verarbeitung reserviert.【F:src/pdfsvg_calibrator/cli.py†L44-L78】 |
| `grid_cell_rel` | `0.02` | Größe der Chamfer-Grid-Zellen relativ zur SVG-Diagonale.【F:configs/default.yaml†L13-L13】 | Zu kleine Zellen verlangsamen Chamfer-Abfragen; zu große verschlechtern Genauigkeit.【F:src/pdfsvg_calibrator/fit_model.py†L203-L205】 |
| `chamfer.sigma_rel` | `0.004` | Breite der Gauß-Glocke für Chamfer-Scores (relativ zur SVG-Diagonale).【F:configs/default.yaml†L14-L15】 | Niedrige Werte bestrafen Abweichungen stärker; beeinflusst Score und Residualstatistik.【F:src/pdfsvg_calibrator/fit_model.py†L206-L275】 |
| `chamfer.hard_mul` | `3.0` | Faktor für harte Distanzgrenze im Chamfer-Score.【F:configs/default.yaml†L14-L16】 | Erweitert oder begrenzt den Suchradius beim Distanzvergleich.【F:src/pdfsvg_calibrator/fit_model.py†L206-L275】 |
| `ransac.iters` | `900` | Anzahl der RANSAC-Iterationen pro Rotationshypothese.【F:configs/default.yaml†L17-L18】 | Mehr Iterationen erhöhen die Trefferchance, verlängern aber die Laufzeit linear.【F:src/pdfsvg_calibrator/fit_model.py†L214-L316】 |
| `ransac.refine_scale_step` | `0.004` | Schrittweite für lokale Skalierungs-Feintuning-Schleifen.【F:configs/default.yaml†L17-L19】 | Bestimmt die Rasterung beim Feintuning der Skalen; kleinere Werte = feinere Suche, mehr Auswertungen.【F:src/pdfsvg_calibrator/fit_model.py†L280-L294】 |
| `ransac.refine_trans_px` | `3.0` | Offset-Gitter (in SVG-Pixeln) für Translationsfeinsuche.【F:configs/default.yaml†L17-L20】 | Höhere Werte erlauben gröbere Verschiebungen, erhöhen aber die Zahl der Chamfer-Bewertungen.【F:src/pdfsvg_calibrator/fit_model.py†L232-L294】 |
| `verify.pick_k` | `5` | Anzahl der Linien, die für die Prüfung ausgewählt werden sollen.【F:configs/default.yaml†L21-L22】 | `select_lines()` nutzt den Wert direkt; niedrigere Werte beschleunigen die Analyse, liefern aber weniger Kontrolle.【F:src/pdfsvg_calibrator/match_verify.py†L400-L521】 |
| `verify.diversity_rel` | `0.10` | Mindestabstand (relativ zur PDF-Diagonale) zwischen ausgewählten Linien, um Ballungen zu vermeiden.【F:configs/default.yaml†L21-L24】 | Verringern für eng gebündelte Strukturen; erhöhen, wenn Linien über die Seite verteilt sein sollen.【F:src/pdfsvg_calibrator/match_verify.py†L425-L505】 |
| `verify.radius_px` | `80` | Suchradius in SVG-Pixeln für Matching und Kostenbewertung.【F:configs/default.yaml†L21-L25】 | Steuert Kandidatenauswahl und Endpunkt-/Midpoint-Kosten; zu klein -> keine Treffer, zu groß -> mehr Verwechslungen.【F:src/pdfsvg_calibrator/match_verify.py†L557-L706】 |
| `verify.dir_tol_deg` | `6.0` | Winkelgrenze, damit Richtungsvergleich nicht als Fehler gewertet wird.【F:configs/default.yaml†L21-L25】 | Wird im Matching und bei Nachbarschaften genutzt; enger = strenger Richtungstest.【F:src/pdfsvg_calibrator/match_verify.py†L216-L305】【F:src/pdfsvg_calibrator/match_verify.py†L557-L706】 |
| `verify.tol_rel` | `0.01` | Relative Fehlerschwelle für PASS/FAIL-Flag (`Pass01`).【F:configs/default.yaml†L21-L26】 | Setzt die ±1 %-Grenze im CSV/Overlay; anpassen für strengere oder lockerere Freigaben.【F:src/pdfsvg_calibrator/match_verify.py†L680-L703】 |
| `neighbors.use` | `true` | Aktiviert Nachbarschaftssignaturen im Matching.【F:configs/default.yaml†L27-L28】 | CLI erzwingt `True`; deaktivieren ist nicht vorgesehen.【F:src/pdfsvg_calibrator/cli.py†L353-L355】【F:src/pdfsvg_calibrator/match_verify.py†L248-L305】 |
| `neighbors.radius_rel` | `0.06` | Radius (relativ zur SVG-Diagonale) für Nachbarschaftssignaturen.【F:configs/default.yaml†L27-L29】 | Wird in `neighbor_signature()` genutzt; größer = mehr Kontext, aber auch mehr Rauschen.【F:src/pdfsvg_calibrator/match_verify.py†L557-L575】 |
| `neighbors.dt` | `0.03` | Erlaubte Differenz im Signaturparameter `t` (Position entlang der Referenz).【F:configs/default.yaml†L27-L30】 | Dient als Schwelle beim Nachbarschafts-Costing; kleiner macht Matching sensibler.【F:src/pdfsvg_calibrator/match_verify.py†L248-L285】 |
| `neighbors.dtheta_deg` | `8.0` | Maximale Differenz der Nachbarschaftswinkel.【F:configs/default.yaml†L27-L31】 | Verhindert das Paaren von Nachbarn mit stark abweichendem Winkel.【F:src/pdfsvg_calibrator/match_verify.py†L248-L285】 |
| `cost_weights.endpoint` | `0.5` | Gewicht des Endpunkt-Terms im Kandidatenkostenmodell.【F:configs/default.yaml†L32-L33】 | Anheben, wenn Endpunktgenauigkeit wichtiger ist als Richtung/Umfeld.【F:src/pdfsvg_calibrator/match_verify.py†L288-L315】 |
| `cost_weights.midpoint` | `0.3` | Gewicht des Mittelpunkt-Terms.【F:configs/default.yaml†L32-L34】 | Höhere Werte bevorzugen Linien mit passenden Mittelpunkten.【F:src/pdfsvg_calibrator/match_verify.py†L288-L315】 |
| `cost_weights.direction` | `0.2` | Gewicht der Richtungsübereinstimmung.【F:configs/default.yaml†L32-L35】 | Anheben, um kleinste Winkelabweichungen stärker zu bestrafen.【F:src/pdfsvg_calibrator/match_verify.py†L216-L315】 |
| `cost_weights.neighbors` | `0.1` | Gewicht der Nachbarschaftskosten.【F:configs/default.yaml†L32-L36】 | Bei unruhigem Umfeld ggf. reduzieren, um robuste Zuordnungen zu behalten.【F:src/pdfsvg_calibrator/match_verify.py†L248-L315】 |
| `sampling.step_rel` | `0.02` | Schrittweite für Stichproben entlang PDF-Linien (relativ zur PDF-Diagonale).【F:src/pdfsvg_calibrator/cli.py†L44-L78】 | Beeinflusst Chamfer-Bewertung und Linienstützung; kleinere Schritte = genauer, aber langsamer.【F:src/pdfsvg_calibrator/fit_model.py†L210-L241】【F:src/pdfsvg_calibrator/match_verify.py†L343-L378】 |
| `sampling.max_points` | `5000` | Obergrenze der Stichprobenpunkte pro Modellbewertung.【F:src/pdfsvg_calibrator/cli.py†L44-L78】 | Schützt vor explodierenden Laufzeiten bei vielen Segmenten.【F:src/pdfsvg_calibrator/fit_model.py†L210-L241】 |
| `neighbors.rho_soft` | `0.25` | Normierungsfaktor für relative Längenunterschiede in Nachbarschaften (nur in Defaults).【F:src/pdfsvg_calibrator/cli.py†L62-L68】 | Kleine Werte bestrafen Längenabweichungen stärker.【F:src/pdfsvg_calibrator/match_verify.py†L248-L285】 |
| `neighbors.penalty_miss` | `1.5` | Strafwert, wenn kein passender Nachbar gefunden wird.【F:src/pdfsvg_calibrator/cli.py†L62-L69】 | Erhöhen, um fehlende Nachbarn stärker zu sanktionieren.【F:src/pdfsvg_calibrator/match_verify.py†L248-L285】 |
| `neighbors.penalty_empty` | `5.0` | Strafwert, wenn Signatur leer ist.【F:src/pdfsvg_calibrator/cli.py†L62-L69】 | Setzt Grundkosten bei fehlendem Kontext; reduziert Fehlalarme bei dünnen Zeichnungen.【F:src/pdfsvg_calibrator/match_verify.py†L248-L285】 |
| `verify.max_cost` | `None` | Optionaler Deckel für Matching-Kosten; fehlt im Default, kann aber gesetzt werden.【F:src/pdfsvg_calibrator/match_verify.py†L638-L644】 | Kleinere Werte erzwingen strengere Zuordnungen, evtl. mehr „no match“ Einträge.【F:src/pdfsvg_calibrator/match_verify.py†L646-L705】 |

### Hartkodierte Konstanten
| Konstante | Wert | Verwendung | Fundstelle |
| --- | --- | --- | --- |
| `neighbor_signature` ρ-Klammer | `≤ 5.0` | Kappung des Längenverhältnisses, damit Ausreißer die Signatur nicht dominieren.【F:src/pdfsvg_calibrator/match_verify.py†L204-L213】 |
| `SegmentGrid.cell_size` Mindestwert | `1e-6` | Verhindert Division durch 0 und winzige Zellen bei Matching/Chamfer.【F:src/pdfsvg_calibrator/match_verify.py†L124-L168】【F:src/pdfsvg_calibrator/fit_model.py†L38-L57】 |
| Chamfer-Sigma-Mindestwert | `1e-9` | Schützt vor extrem kleinen Gaußbreiten beim Sampling.【F:src/pdfsvg_calibrator/match_verify.py†L357-L374】 |
| Dummy-/Unavailable-Kosten | `5e5` / `1e6` | Platzhalterkosten für unbesetzte Spalten in der ungarischen Matrix.【F:src/pdfsvg_calibrator/match_verify.py†L608-L627】 |
| PASS/FAIL Grenzwertschutz | `max(max_cost, 1e-6)` | Verhindert Division durch 0 bei Konfidenzberechnung.【F:src/pdfsvg_calibrator/match_verify.py†L683-L690】 |
| Overlay-Farben | Rot `#F44336`, Blau `#2979FF` | Farbliche Unterscheidung PDF↔SVG im Overlay.【F:src/pdfsvg_calibrator/overlays.py†L13-L205】 |
| PDF-Maßeinheiten | `pt→px = 96/72`, `mm→px = 96/25.4` etc. | Einheitennormalisierung beim SVG-Parsen.【F:src/pdfsvg_calibrator/io_svg_pdf.py†L330-L361】 |

## Matching & Kostenmodell im Detail
1. **Transformation anwenden** – `match_lines()` wendet das Modell auf PDF-Segmente an.【F:src/pdfsvg_calibrator/match_verify.py†L568-L573】
2. **Signaturen bilden** – Für jedes Segment werden Nachbarn gesammelt (`t`, Winkel, Längenverhältnis) und nach Koordinaten sortiert.【F:src/pdfsvg_calibrator/match_verify.py†L171-L213】
3. **Kostenbestandteile** –
   - Endpunkte und Mittelpunkte werden per euklidischem Abstand verglichen.【F:src/pdfsvg_calibrator/match_verify.py†L228-L246】
   - Richtungsunterschiede liefern `∞`, wenn sie größer als `dir_tol_deg` sind.【F:src/pdfsvg_calibrator/match_verify.py†L216-L226】
   - Nachbarschaften addieren Trefferkosten bzw. Strafen für fehlende Partner.【F:src/pdfsvg_calibrator/match_verify.py†L248-L285】
4. **Gewichte anwenden** – `cost_weights` mischen die Terme zum Gesamtkostenwert.【F:src/pdfsvg_calibrator/match_verify.py†L288-L315】
5. **Ungarische Zuordnung** – `hungarian_solve()` sucht minimale Gesamtkosten; `verify.max_cost` oder Median-basierte Grenzwerte filtern schlechte Paare heraus.【F:src/pdfsvg_calibrator/match_verify.py†L629-L704】

## Performance-Tipps
- **Sampling begrenzen** – Reduzieren Sie `sampling.max_points` oder erhöhen Sie `sampling.step_rel`, wenn Chamfer-Berechnungen zu langsam werden.【F:src/pdfsvg_calibrator/fit_model.py†L210-L241】
- **RANSAC gezielt einsetzen** – Senken Sie `ransac.iters`, wenn Ihre Zeichnung sauber ist; erhöhen Sie ihn bei vielen Störlinien.【F:src/pdfsvg_calibrator/fit_model.py†L214-L316】
- **Nachbarschaftsradius abstimmen** – Ein kleinerer `neighbors.radius_rel` reduziert Signaturgröße und Kostenmatrix, beschleunigt Matching in dichtem Layout.【F:src/pdfsvg_calibrator/match_verify.py†L557-L605】
- **Rotationen einschränken** – Entfernen Sie ungenutzte Einträge aus `rot_degrees`, um weniger Hypothesen durchrechnen zu müssen.【F:src/pdfsvg_calibrator/fit_model.py†L230-L316】

## Debugging falscher Matches
1. **Fehler ±0.01 verstehen** – `Pass01` basiert auf `verify.tol_rel`. Erhöhen Sie den Wert temporär, um zu prüfen, ob Abweichungen knapp außerhalb liegen.【F:src/pdfsvg_calibrator/match_verify.py†L680-L703】
2. **Nachbarschaften inspizieren** – Aktivieren Sie `--verbose`, um zusätzliche Logs zu sehen, und drucken Sie Signaturen, indem Sie in `neighbor_signature()` temporär Debug-Ausgaben setzen.【F:src/pdfsvg_calibrator/cli.py†L339-L412】【F:src/pdfsvg_calibrator/match_verify.py†L171-L213】
3. **RNG fixieren** – Nutzen Sie `--rng-seed`, um deterministische Ergebnisse zu erzwingen und RANSAC/Line-Auswahl reproduzierbar zu machen.【F:src/pdfsvg_calibrator/cli.py†L219-L221】【F:src/pdfsvg_calibrator/cli.py†L339-L414】
4. **Kandidatenradius erhöhen** – Falls Linien nicht gefunden werden, erhöhen Sie `verify.radius_px` oder `neighbors.radius_rel` schrittweise.【F:src/pdfsvg_calibrator/match_verify.py†L557-L706】
5. **Chamfer prüfen** – Reduzieren Sie `sampling.step_rel` und `chamfer.sigma_rel`, um die Punktwolke dichter und sensibler zu machen; beobachten Sie dabei die Laufzeit.【F:src/pdfsvg_calibrator/fit_model.py†L206-L275】

## Erweiterungen & Anpassungen
- **90°/270° aktivieren** – `rot_degrees` erweitern und ggf. `angle_tol_deg` auf 4° senken, wenn viele schräge Linien vorhanden sind.【F:configs/default.yaml†L1-L2】【F:src/pdfsvg_calibrator/fit_model.py†L230-L238】
- **Toleranzen justieren** – Für sehr präzise Pläne `verify.tol_rel` Richtung 0.005 senken; für raue Scans `angle_tol_deg` anheben, damit leicht schräge Linien akzeptiert werden.【F:configs/default.yaml†L2-L26】【F:src/pdfsvg_calibrator/match_verify.py†L216-L703】
- **Segment-Preprocessing ergänzen** – Die `merge`-Parameter stehen bereit, um künftig kurze Lücken zu schließen. Anpassungen erfolgen im I/O-Modul (`io_svg_pdf`).【F:configs/default.yaml†L10-L12】【F:src/pdfsvg_calibrator/io_svg_pdf.py†L235-L308】
