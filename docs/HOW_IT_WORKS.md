# Wie pdfsvg-calibrator arbeitet

## Überblick
`pdfsvg-calibrator` bestimmt eine affine Abbildung `X = sx * x + tx`, `Y = sy * y + ty`. Die vier unbekannten Skalierungs- und Verschiebungsparameter werden in einer gestuften Pipeline ermittelt, die Rotationen und Spiegelungen (standardmäßig 0°/180°) berücksichtigt und robuste Seeds an das RANSAC-/Chamfer-Matching weitergibt.【F:src/pdfsvg_calibrator/calibrate.py†L38-L355】【F:src/pdfsvg_calibrator/fit_model.py†L186-L316】

## Gestufte Pipeline
1. **Orientierung & Flip-Hypothesen** – `pick_flip_and_rot()` rasterisiert PDF- und SVG-Segmente (inkl. Hilfsrahmen) auf das Standardraster (`512×512`) und probiert vier Flip-Kombinationen über die konfigurierten Rotationen.【F:src/pdfsvg_calibrator/orientation.py†L246-L329】
2. **Rasterprojektion & Phase-Korrelation** – Für jede Hypothese wird per FFT-basierter Phasenkorrelation der Versatz (`dx`, `dy`) geschätzt. Der Score kombiniert Spitzenhöhe und normalisierte Überlappung und liefert `tx₀`/`ty₀` in Seitenkoordinaten.【F:src/pdfsvg_calibrator/orientation.py†L179-L275】
3. **Skalen-Seed & Schranken** – Aus den Seitenabmessungen werden `sx₀`/`sy₀` berechnet und mit dem Flip-Vorzeichen versehen. `calibrate()` hinterlegt daraus Seeds, Clamp-Bereiche und Fenster für die spätere Feinoptimierung.【F:src/pdfsvg_calibrator/calibrate.py†L91-L195】
4. **DT-Qualitätsprüfung** – Abhängig von `verify.mode` werden Distanztransformationen gesampelt (`distmap`) oder Linien-Matches berechnet, um RMSE/P95 zu bestimmen.【F:src/pdfsvg_calibrator/calibrate.py†L253-L313】【F:src/pdfsvg_calibrator/cli.py†L857-L875】
5. **Optionale Feinabstimmung & Reporting** – `fit_model.calibrate()` nutzt Chamfer-Distanzen auf einem grob→fein Raster, sampelt Segmentpunkte (max. 1 500) und führt RANSAC aus; anschließend erfolgen Matching, Overlays und CSV.【F:src/pdfsvg_calibrator/calibrate.py†L139-L355】【F:src/pdfsvg_calibrator/fit_model.py†L186-L316】【F:src/pdfsvg_calibrator/overlays.py†L92-L205】

Je nach Modus endet der Lauf nach Schritt 4 (`verify.mode=distmap`) oder ergänzt Schritt 5 mit Matching/CSV-Ausgabe.

### Sequenzdiagramm-ähnlicher Ablauf
- `cli.run()`
  - `_load_config_with_defaults()` → zusammengeführte Konfiguration (`dict`).
  - `convert_pdf_to_svg_if_needed()` → optionaler SVG-Export über PyMuPDF.【F:src/pdfsvg_calibrator/cli.py†L323-L374】
  - `load_pdf_segments()` / `load_svg_segments()` → Segmentlisten + Seitengrößen.【F:src/pdfsvg_calibrator/io_svg_pdf.py†L235-L392】
  - `calibrate.calibrate()` → Orientierungssaat + lokales `Model` Ergebnis.【F:src/pdfsvg_calibrator/calibrate.py†L38-L355】
  - `select_lines()` → Auswahl bis zu fünf repräsentativer Linien.【F:src/pdfsvg_calibrator/match_verify.py†L334-L548】
  - `match_lines()` → Ungarisches Matching + PASS/FAIL.【F:src/pdfsvg_calibrator/match_verify.py†L551-L705】
  - `write_*()` / `_summarize()` → Overlays, CSV, Konsolenausgabe.【F:src/pdfsvg_calibrator/cli.py†L258-L412】【F:src/pdfsvg_calibrator/overlays.py†L92-L205】

### Datenflüsse
- **Segmentlisten** – Werden durch Orientation, RANSAC und Matching unverändert weitergereicht.【F:src/pdfsvg_calibrator/orientation.py†L246-L329】【F:src/pdfsvg_calibrator/match_verify.py†L334-L705】
- **Modelldaten** – `Model(rot_deg, sx, sy, tx, ty, score, rmse, p95, median)` wird im Matching, in Overlays und im CSV wiederverwendet.【F:src/pdfsvg_calibrator/types.py†L10-L27】【F:src/pdfsvg_calibrator/overlays.py†L19-L204】
- **Nachbarschaftssignaturen** – `neighbor_signature()` beschreibt lokale Umgebung (`t`, `Δθ`, `ρ`) und beeinflusst Kosten & Konfidenz.【F:src/pdfsvg_calibrator/match_verify.py†L171-L315】【F:src/pdfsvg_calibrator/match_verify.py†L685-L690】

## Datenmodelle
- `Segment(x1, y1, x2, y2)` – Repräsentiert eine Linie im Ursprungskoordinatensystem.【F:src/pdfsvg_calibrator/types.py†L4-L9】
- `Model(...)` – Enthält Transformation + Qualitätskennzahlen (Score, RMSE, P95, Median) und optionale Qualitätsnotizen.【F:src/pdfsvg_calibrator/types.py†L10-L27】【F:src/pdfsvg_calibrator/calibrate.py†L332-L384】
- `Match(...)` – Zeileneintrag für CSV/Overlay inkl. Kosten, Confidence und PASS-Flags.【F:src/pdfsvg_calibrator/types.py†L16-L27】【F:src/pdfsvg_calibrator/overlays.py†L117-L205】

## Parameter- und Konstantenverzeichnis
### Schlüsselkonfiguration & empfohlene Defaults
| Schlüssel | Empfohlen | Wirkung & Hinweise |
| --- | --- | --- |
| `rot_degrees` | `[0, 180]` (ggf. 90/270 ergänzen) | Grobe Rotationshypothesen für Stufe 1.【F:configs/default.yaml†L1-L1】【F:src/pdfsvg_calibrator/orientation.py†L246-L329】 |
| `orientation.enabled` | `true` | Aktiviert den Orientation-Gate. Abschalten nur zum Debuggen, da Seeds fehlen würden.【F:src/pdfsvg_calibrator/calibrate.py†L66-L124】 |
| `orientation.raster_size` | `512` | Größer = genauere Korrelation, aber teurer im FFT.【F:src/pdfsvg_calibrator/orientation.py†L179-L329】 |
| `orientation.use_phase_correlation` | `true` | Liefert den stabilsten Translationsseed; `false` nur bei starkem Rauschen verwenden.【F:src/pdfsvg_calibrator/orientation.py†L179-L275】 |
| `orientation.sample_topk_rel` | `0.2` | Nutzt die längsten ~20 % der Segmente für Hypothesen; mehr Anteil hilft bei sehr dünnen Plänen.【F:configs/default.yaml†L6-L8】【F:src/pdfsvg_calibrator/orientation.py†L300-L316】 |
| `refine.scale_max_dev_rel` | `0.02` | ±2 % Fenster um `sx₀`/`sy₀`. Enger für präzise Layouts, weiter bei verzogenen Exporten.【F:src/pdfsvg_calibrator/calibrate.py†L141-L175】 |
| `refine.trans_max_dev_px` | `8.0` | Fenster in SVG-Pixeln für lokale Suche; größer wählen, wenn Phase-Seeds driftig sind.【F:src/pdfsvg_calibrator/calibrate.py†L141-L195】 |
| `refine.max_iters` | `60` | Obergrenze für lokale Auswertungen. Wird als `ransac.iters` gespiegelt.【F:src/pdfsvg_calibrator/calibrate.py†L156-L215】 |
| `refine.max_samples` | `1500` | Sicherer Deckel für Samplingpunkte, damit Läufe <10 s bleiben.【F:src/pdfsvg_calibrator/calibrate.py†L157-L215】 |
| `refine.quality_gate.*` | siehe YAML | Fallback mit größeren Fenstern, falls Score/RMSE zu schlecht bleiben.【F:src/pdfsvg_calibrator/calibrate.py†L222-L355】 |
| `grid.initial_cell_rel` / `grid.final_cell_rel` | `0.05` / `0.02` | Zweistufiges Chamfer-Grid (grob→fein). `initial` nicht kleiner als 0.03 wählen.【F:src/pdfsvg_calibrator/calibrate.py†L216-L220】【F:src/pdfsvg_calibrator/fit_model.py†L203-L316】 |
| `sampling.step_rel` | `0.03` | Relative Abtastweite entlang PDF-Linien. 0.02 für höhere Genauigkeit, <0.01 nur bei Spezialfällen.【F:src/pdfsvg_calibrator/calibrate.py†L210-L214】【F:src/pdfsvg_calibrator/fit_model.py†L210-L241】 |
| `sampling.max_points` | `1500` | Deckt sich mit `refine.max_samples`; schützt Chamfer vor Explosionen.【F:src/pdfsvg_calibrator/calibrate.py†L210-L215】 |
| `ransac.iters` | `=refine.max_iters` | Automatisch gesetzt, ausreichend für stabile Seeds.【F:src/pdfsvg_calibrator/calibrate.py†L199-L208】【F:src/pdfsvg_calibrator/fit_model.py†L214-L316】 |
| `ransac.refine_scale_step` / `refine_trans_px` | auto | Werden aus den Fenstern abgeleitet (1/10 bzw. 1/3). Nur manuell setzen, wenn Sampling extrem fein ist.【F:src/pdfsvg_calibrator/calibrate.py†L199-L208】 |
| `verify.mode` | `lines` (oder `distmap`) | Steuert, ob Linien gematcht werden oder Distanzkarten die QA liefern (`--opts verify.mode distmap`).【F:src/pdfsvg_calibrator/cli.py†L857-L875】【F:src/pdfsvg_calibrator/calibrate.py†L253-L313】 |
| `verify.*` | Defaults übernehmen | Matching-/PASS-Checks. 1 %-Grenze über `verify.tol_rel` regulierbar.【F:configs/default.yaml†L51-L52】【F:src/pdfsvg_calibrator/match_verify.py†L334-L705】 |
| `neighbors.*` | aktiv lassen | Stabilisiert Kostenmatrix. Radius 0.05–0.07 bewährt; Strafen nur bei Spezialfällen justieren.【F:configs/default.yaml†L27-L31】【F:src/pdfsvg_calibrator/match_verify.py†L171-L315】 |
| `cost_weights.*` | `0.5/0.3/0.2/0.1` | Ausbalancierte Mischung von Endpunkt, Mittelpunkt, Richtung, Nachbarn.【F:configs/default.yaml†L32-L36】【F:src/pdfsvg_calibrator/match_verify.py†L288-L315】 |

Weitere I/O-Parameter wie `curve_tol_rel`, `straight_max_dev_rel` oder `min_len_raw` orientieren sich an der PDF-Diagonale und regeln, wie großzügig Linien als „gerade“ akzeptiert werden.【F:configs/default.yaml†L3-L9】【F:src/pdfsvg_calibrator/io_svg_pdf.py†L235-L308】 Die `merge.*`-Werte sind vorbereitet, um kollineare Segmente zusammenzuführen; aktuell wird der Merge noch außerhalb der Pipeline vorgenommen.【F:configs/default.yaml†L10-L18】

### Hartkodierte Konstanten
| Konstante | Wert | Verwendung | Fundstelle |
| --- | --- | --- | --- |
| `neighbor_signature` ρ-Klammer | `≤ 5.0` | Kappung des Längenverhältnisses, damit Ausreißer die Signatur nicht dominieren.【F:src/pdfsvg_calibrator/match_verify.py†L204-L213】 |
| `SegmentGrid.cell_size` Mindestwert | `1e-6` | Schutz vor extrem kleinen Chamfer-Zellen.【F:src/pdfsvg_calibrator/match_verify.py†L124-L168】【F:src/pdfsvg_calibrator/fit_model.py†L38-L57】 |
| Chamfer-Sigma-Mindestwert | `1e-9` | Verhindert numerische Ausreißer beim Gauß-Weighting.【F:src/pdfsvg_calibrator/match_verify.py†L357-L374】 |
| Dummy-/Unavailable-Kosten | `5e5` / `1e6` | Platzhalterkosten für leere Matching-Spalten.【F:src/pdfsvg_calibrator/match_verify.py†L608-L627】 |
| PASS/FAIL Grenzwertschutz | `max(max_cost, 1e-6)` | Verhindert Division durch 0 bei Konfidenzberechnung.【F:src/pdfsvg_calibrator/match_verify.py†L683-L690】 |
| Overlay-Farben | Rot `#F44336`, Blau `#2979FF` | Farbcodierung PDF↔SVG im Overlay.【F:src/pdfsvg_calibrator/overlays.py†L13-L205】 |
| PDF-Maßeinheiten | `pt→px = 96/72`, `mm→px = 96/25.4` etc. | Einheitennormalisierung beim SVG-Parsen.【F:src/pdfsvg_calibrator/io_svg_pdf.py†L330-L361】 |

## Matching & Kostenmodell im Detail
1. **Transformation anwenden** – `match_lines()` transformiert PDF-Segmente mit dem best-fit Modell.【F:src/pdfsvg_calibrator/match_verify.py†L568-L573】
2. **Signaturen bilden** – Nachbarn werden gesammelt, sortiert und gegen Kandidaten verglichen.【F:src/pdfsvg_calibrator/match_verify.py†L171-L285】
3. **Kostenbestandteile** – Endpunkte, Mittelpunkte, Richtungen und Nachbarn tragen gewichtet zum Score bei; Toleranzverstöße ergeben `∞`.【F:src/pdfsvg_calibrator/match_verify.py†L216-L315】
4. **Ungarische Zuordnung** – `hungarian_solve()` findet die Minimalzuordnung; `verify.max_cost`/Median-Filtern entfernt schlechte Paare.【F:src/pdfsvg_calibrator/match_verify.py†L629-L705】

## Performance-Tipps
- **Kollineare Segmente zusammenführen** – Entfernt doppelte Linien, reduziert Chamfer-Abtastungen und beschleunigt RANSAC. Extern mergen oder die vorbereiteten `merge.*`-Grenzen nutzen.【F:configs/default.yaml†L10-L18】【F:src/pdfsvg_calibrator/fit_model.py†L186-L316】
- **PDF-Diagonale als Referenz** – Alle relativen Toleranzen skalieren mit der Seitendiagonale. Größere Pläne benötigen daher keine manuelle Anpassung von `curve_tol_rel`, `straight_max_dev_rel` oder `sampling.step_rel`.【F:src/pdfsvg_calibrator/io_svg_pdf.py†L235-L308】【F:src/pdfsvg_calibrator/calibrate.py†L141-L175】
- **Chamfer-Grid zweistufig fahren** – Mit `grid.initial_cell_rel` (grob) und `grid.final_cell_rel` (fein) lassen sich Läufe unter 10 s halten, ohne Genauigkeit einzubüßen.【F:src/pdfsvg_calibrator/calibrate.py†L216-L220】【F:src/pdfsvg_calibrator/fit_model.py†L203-L316】
- **Rotationen einschränken** – Entfernen Sie ungenutzte Einträge aus `rot_degrees`, um weniger Hypothesen zu prüfen.【F:src/pdfsvg_calibrator/orientation.py†L246-L329】

## Debugging & Troubleshooting
| Symptom | Mögliche Ursache | Gegenmittel |
| --- | --- | --- |
| Ergebnis wirkt gespiegelt | Falscher Flip/Rotation gewann Stufe 1 | `rot_degrees` um 90°-Schritte erweitern, Orientation aktiv lassen, optional `orientation.use_phase_correlation=false` bei extrem symmetrischen Plänen testen.【F:src/pdfsvg_calibrator/orientation.py†L246-L329】 |
| Translation riesig oder driftet | Phase-Korrelation landet auf Nebenmaximum | `orientation.raster_size` erhöhen, `refine.trans_max_dev_px` auf 20–30 anheben und Quality-Gate-Fallback zulassen; wenn `|t|` über der Konsole-Toleranz bleibt, SVG-Exportursprung prüfen.【F:src/pdfsvg_calibrator/orientation.py†L179-L275】【F:src/pdfsvg_calibrator/calibrate.py†L141-L355】【F:src/pdfsvg_calibrator/cli.py†L390-L408】 |
| Laufzeiten >10 s | Zu viele Samples/Hypothesen | `sampling.max_points` (→1200) senken, `rot_degrees` reduzieren, Segmente vorab mergen.【F:src/pdfsvg_calibrator/calibrate.py†L199-L215】【F:configs/default.yaml†L1-L18】 |
| Linien-Report ist verrauscht | Viele kurze/gebrochene Segmente | `straight_max_dev_rel` leicht erhöhen oder externe Glättung anwenden; `verify.radius_px` moderat vergrößern.【F:src/pdfsvg_calibrator/io_svg_pdf.py†L235-L308】【F:src/pdfsvg_calibrator/match_verify.py†L557-L706】 |

## Debugging falscher Matches
1. **±1 %-Grenze prüfen** – `Pass01` basiert auf `verify.tol_rel`. Temporär erhöhen, um Grenzfälle sichtbar zu machen.【F:src/pdfsvg_calibrator/match_verify.py†L680-L703】
2. **Nachbarschaften inspizieren** – `--verbose` aktivieren oder in `neighbor_signature()` Logging setzen, um Umfeldfehler zu sehen.【F:src/pdfsvg_calibrator/cli.py†L339-L412】【F:src/pdfsvg_calibrator/match_verify.py†L171-L285】
3. **Determinismus erzwingen** – `--rng-seed` nutzen, damit RANSAC und Linienauswahl reproduzierbar werden.【F:src/pdfsvg_calibrator/cli.py†L219-L221】【F:src/pdfsvg_calibrator/cli.py†L339-L414】
4. **Kandidatenradius erweitern** – Falls Linien nicht gematcht werden, `verify.radius_px` und `neighbors.radius_rel` stufenweise erhöhen.【F:src/pdfsvg_calibrator/match_verify.py†L557-L706】
5. **Chamfer-Parameter feintunen** – Kleinere `sampling.step_rel` und `chamfer.sigma_rel` machen Scores sensibler, kosten aber Laufzeit.【F:src/pdfsvg_calibrator/fit_model.py†L206-L275】

## Erweiterungen & Anpassungen
- **90°/270° aktivieren** – `rot_degrees` erweitern und ggf. `angle_tol_deg` auf 4° senken, wenn viele schräge Linien vorliegen.【F:configs/default.yaml†L1-L2】【F:src/pdfsvg_calibrator/fit_model.py†L230-L238】
- **Toleranzen justieren** – Für sehr präzise Pläne `verify.tol_rel` Richtung 0.005 senken; für raue Scans `angle_tol_deg` erhöhen.【F:configs/default.yaml†L2-L26】【F:src/pdfsvg_calibrator/match_verify.py†L216-L703】
- **Segment-Preprocessing erweitern** – Die `merge`-Parameter sind vorbereitet, um kurze Lücken zu schließen; Anpassungen erfolgen im I/O-Modul (`io_svg_pdf`).【F:configs/default.yaml†L10-L18】【F:src/pdfsvg_calibrator/io_svg_pdf.py†L235-L308】
