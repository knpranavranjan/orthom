# OA Screening Kiosk — Build Plan

Bilingual (English / हिन्दी), offline-first osteoarthritis screening station on a
3.5" Raspberry Pi panel, shipping from the same React codebase as an Android app.

---

## 1. The number that governs everything: 480 × 320

Every 3.5" Pi TFT — SPI, high-speed SPI, or HDMI — is 480×320. Design to it first.

    header       36 px
    content     236 px
    action bar   48 px
                -------
                320 px

Consequences:
- Movement screens get NO shared action bar. They render `bare` and supply their own
  button row. Next appears only once a capture exists, so a movement can't be skipped.
- 44 px minimum touch target (48 preferred) — resistive touch is imprecise.
- No dropdowns, no free-text numerals. Sex = 3-way segmented control. Age and the
  0–10 score = steppers with large +/- targets.
- Body text bottoms out at 14 px.
- One idea per screen.

## 2. Which panel do you have? (decides the video question)

    ls /sys/class/graphics/
    dmesg | grep -iE 'ili9486|ili9341|fb_|spi|vc4'
    grep -iE 'dtoverlay|hdmi|framebuffer' /boot/firmware/config.txt
    # Physical: 40-pin header only -> SPI.  Mini-HDMI bridge -> HDMI.

**Recommendation regardless of the answer: do not use generated video for the
movement demonstrations.**

- Performance: on SPI, video decode + framebuffer copy competes with the sensor loop.
- Correctness: generated clips of clinical movements routinely show the wrong thing —
  a knee flexing past physiological range, or a sit-to-stand pushing off with hands
  when *without hands* is the whole point of the test.

Use inline SVG + CSS keyframes. ~30 lines, a few KB, crisp at 480×320, anatomically
correct by construction, identical in the Android build. Keep the slot polymorphic:
`media: { kind: 'svg' | 'video' | 'image', src }` so real phone footage drops in later.
Real footage beats both — faster to record than to prompt, and correct.

## 3. Stack

| Layer | Choice | Why |
|---|---|---|
| Build | Vite + React 18 + TS | Fast HMR when iterating over SSH to a Pi |
| Mobile | **Capacitor** | Same build wrapped natively. React Native = full rewrite |
| Flow | Step-index state machine | Linear kiosk flow; router adds unwanted URL semantics |
| Session | Zustand | ~1 kB, resettable between patients |
| Persistence | Dexie (IndexedDB) | Works on Pi Chromium AND in Capacitor WebView |
| i18n | react-i18next | Locale = JSON file; third language is a data change |
| Styling | Plain CSS + custom properties | Hand-tuned density tokens beat framework defaults here |
| Charts | Hand-rolled SVG | Library defaults are all wrong at 480×320 |
| Keyboard | react-simple-keyboard | In-app; no OS keyboard needed, works on mobile too |

Keep the mobile path open: never use `window.innerWidth` for layout, put every
platform capability behind an interface in `src/services/`, and run
`npx cap add android` in week one.

## 4. The sensor bridge (browser can't read I2C)

    sensors (I2C / SPI / ADC)
         |
    sensord.py            <- Python on the Pi: reads sensors, computes ROM,
         |                   cadence, symmetry, crepitus events
         |  WebSocket ws://localhost:8765
         v
    src/services/sensors.ts   <- one interface, two implementations
         |                       MockSensorSource   (dev + demo fallback)
         |                       SocketSensorSource (the Pi)
         v
    MovementCapture.tsx

Message contract — fix early, don't let it drift:

    { t: 1724..., movement: 'flexion', frame: { angle: 87.4, velocity: 42.1,
      load_l: 0.38, load_r: 0.62, acoustic_rms: 0.09 } }

Ship `MockSensorSource` switchable from a hidden long-press on home. If a lead fails
before the demo, you show the full flow on recorded data instead of not showing it.

## 5. Screen map — 9 screens

Your eight, plus registration. Without it the encounter has no patient attached, so
screen 9's summary has no name and the record has nothing to key on.

| # | Screen | What it does |
|---|---|---|
| 1 | Home | What the tool is, language toggle, Start, Records |
| 2 | **Patient registration** (new) | Name, age, sex, consent. Mints `patientId` |
| 3 | Movement 1 — knee flexion/extension | Max ROM, velocity, hesitation/catching, crepitus by angle. IMU pair, flex strip, acoustic |
| 4 | Movement 2 — sit-to-stand | Time, weight-shift asymmetry, trunk lean. IMU pair, FSR insole |
| 5 | Movement 3 — gait, 5–10 steps | Stride, cadence, L/R symmetry, offloading. IMU pair, FSR insole |
| 6 | Sensor summary + manual score | Computed values + health worker's 0–10 stepper |
| 7 | X-ray upload | Camera or file pick, stored as Blob |
| 8 | X-ray comparison + heatmap | Side-by-side, opacity slider, KL distribution |
| 9 | Result + summary | Risk band, description, guidance, PDF, save to records |
| — | Records | Search, reopen encounter, per-row sync status |

Registration specifics:
- Records key on generated `patientId` (PT-xxxxx), never the name. Names get typed
  three ways across three visits and the record fragments.
- Consent is a latch, not a field. Next stays dark until ticked.
- Layout: labels above fields = ~200 px, won't fit. Labels inline-left = 40+40+40+36
  = 156 px + gaps. Fits, but the name field can't have a floating label.

OPEN: what happens when someone taps the name field? Resistive kiosk has no keyboard
unless you add one. `react-simple-keyboard` as a sheet = one evening. Fallback is a
numeric patient code on a keypad you draw, names added later on a laptop.

## 6. Movement capture — one component, three configs

`movements.ts` holds instruction key, media descriptor, sensor list, rep/duration
target, metrics to surface. A fourth movement is an array entry, not a page.

    idle --calibrate--> calibrating --3s--> ready --start--> recording
      ^                      |                |                 |
      |                      +--stop(abort)---+                 | stop
      +------------- retry <--- review <-------------------------+

Calibrate holds 3 s and stores a baseline QUALITY figure shown in the status line.
A bad baseline vanishing silently into the score is worse than no baseline.

| State | Calibrate | Start | Stop | Retry | Next |
|---|---|---|---|---|---|
| idle | on | off | off | off | off |
| calibrating | off | off | on | off | off |
| ready | on | on | off | off | off |
| recording | off | off | on | off | off |
| review | on | off | off | on | on |

Stop is live during calibration and recording only — it means *abort* in one and
*finish* in the other, so its label changes with state. Export the table as `CONTROLS`
so buttons render from data and can't drift from the diagram.

## 7. Bilingual — en / hi

    src/i18n/index.ts
    src/i18n/locales/en.json
    src/i18n/locales/hi.json

Devanagari is not just a different string table:
- **Vertical space.** शिरोरेखा headline + matras above and below. At 14 px with
  line-height 1.3 it clips. Use 1.6 via `:lang(hi)`. Every screen was budgeted at
  236 px — test the longest HINDI string on every screen, not the English one.
- **Bundle the font.** Offline is a hard requirement, so no font CDN. Subset Noto
  Sans Devanagari as a local @font-face. Set `lang` on `<html>`.
- **Clinical numerals stay Latin** in both languages. Degrees, seconds, 0–10 score,
  KL grade. Devanagari digits (०१२) in a clinical readout invite misreading.
- **Never concatenate.** Named interpolation only: `t('rom', { deg })`.
- **Toggle on every screen**, persisted to Dexie so it survives a kiosk restart.

Worth stating in the submission: the brief targets NER, where Hindi is a link language
but not the first language of much of the population (Assamese, Bodo, Khasi, Mizo,
Manipuri). en+hi is the right build scope — say explicitly that a further language is
one JSON file plus a font subset, no code change.

## 8. Data, records, offline

    patients    id, name, age, sex, consentAt, createdAt
    encounters  id, patientId, startedAt, completedAt, locale, synced
    captures    id, encounterId, movement, state, metrics, quality, rawRef
    xrays       id, encounterId, blob, capturedAt
    reports     id, encounterId, riskBand, klGrade, klDist, manualScore, pdfBlob

Everything writes locally first with a `synced` flag. Outbox pushes when connectivity
appears. Nothing in the UI blocks on network. Records list shows per-row sync state.

TRAP: jsPDF does not do complex text shaping. Hindi comes out with broken conjuncts
and misplaced matras even after embedding the font. Either render the report as HTML
and rasterise with html2canvas before placing in the PDF, or generate PDFs in English
only. Decide now, not while exporting a demo report.

## 9. X-ray, heatmap, model boundary

Screen 8 does both views: side-by-side + opacity slider over a canvas composite +
KL probability distribution beneath. A borderline 0.52 / 0.41 split between adjacent
grades is more honest and more interesting to demo than a confident 99.94% — and it's
the case where a screening tool's advice actually matters.

| Backend | Latency on Pi | Use |
|---|---|---|
| Precomputed fixtures | instant | Ship this. Sample X-rays + stored Grad-CAM overlays. Zero demo risk |
| TF.js on-device | seconds | Feasible for a small CNN on Pi 4. Determinate progress, never a frozen screen |
| LAN inference server | fast | Laptop runs the real model. Best quality, needs a network |

`computeRisk()` fuses movement metrics + manual 0–10 + KL grade into a risk band.
~10 lines, clearly a placeholder. The interface is the commitment, not the arithmetic.

NON-NEGOTIABLE: this is a SCREENING tool, not diagnostic. Result screen says so in
both languages, PDF carries the same line, wording routes to a clinician:
"indicators consistent with moderate OA risk; clinical assessment recommended",
not "Grade 3 osteoarthritis".

## 10. Running it on the Pi

    npx vite preview --host --port 4173

    chromium-browser --kiosk --app=http://localhost:4173 \
      --window-size=480,320 --window-position=0,0 \
      --disable-pinch --overscroll-history-navigation=0 \
      --noerrdialogs --disable-session-crashed-bubble \
      --disable-features=TranslateUI

    xset s off; xset -dpms; xset s noblank

Calibrate resistive touch with `xinput_calibrator` or a libinput calibration matrix,
and persist it. An uncalibrated panel registers taps tens of pixels off, which reads
as "the app is broken". Autostart via systemd user service or desktop autostart.
Do NOT add `--incognito` — it wipes IndexedDB, and with it your records.

DO THIS FIRST: boot the scaffold in kiosk mode with placeholder content and calibrate
touch. Both are unbounded unknowns and neither depends on the app existing.

## 11. Build order — hardware risk first, model last

1. Scaffold + boot on the panel (Vite/React/TS, `npx cap add android`, kiosk, calibrate touch)
2. Shell, CSS tokens at 480×320, i18n + working toggle
3. Registration, Dexie, records list
4. Movement capture against mock sensors (all three screens at once)
5. Sensor summary + manual score
6. X-ray upload, overlay, KL distribution (on fixtures)
7. Result screen + PDF + disclaimer wording
8. Real sensors over the WebSocket bridge (mock stays as fallback)
9. Real model behind the same interface — last, deliberately

## 12. Coverage against the brief

| Req | Requirement | Where |
|---|---|---|
| a | Joint movement, gait/posture, pain input, sensor/imaging | Screens 3–5, 6, 7–8 |
| b | AI/ML analysis identifying high-risk cases | `inference.ts` + `computeRisk()`, screens 8–9 |
| c | PHCs, rural camps, outreach | Portable Pi kiosk + Android from one codebase |
| d | Preliminary risk assessment and severity | Screen 9 risk band + KL distribution |
| e | Digitally record symptoms and reports | Dexie schema, records list, PDF |
| f | Multilingual, simple interface | en/hi; further locales are a JSON file |
| g | Low-connectivity, offline | IndexedDB-first, `synced` flag, outbox |
| h | Awareness and preventive guidance | Screen 9 guidance block, localised, in PDF |
| — | Secure data mgmt, analytics dashboard | **GAP** — phase two. Name it explicitly rather than leave it unaddressed |

## 13. Open decisions

1. Which panel — SPI or HDMI? Decides the media slot.
2. On-screen keyboard, or numeric patient code? Affects screen 2, ~one evening.
3. Deadline date? The build order is sequenced but not dated.
4. Sample knee X-rays with KL grades? Lets screens 7–8 be built against something real.
