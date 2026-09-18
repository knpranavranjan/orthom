# OA Screening Kiosk

Bilingual (English / हिन्दी), offline-first knee osteoarthritis screening station
for a 3.5" Raspberry Pi panel — and the same React codebase as an Android app.

**Screening tool. Not a diagnostic device.**

---

## Run it

```bash
npm install
npm run dev        # http://localhost:5173
npm run build      # -> dist/
npm run preview    # http://localhost:4173
npm run typecheck
```

### On the Pi, in kiosk mode

```bash
npm run build && npm run preview &
npm run kiosk                      # chromium --kiosk at 480x320
xset s off; xset -dpms; xset s noblank
```

Calibrate the resistive panel with `xinput_calibrator` (or a libinput
calibration matrix) and persist the matrix. An uncalibrated panel registers
taps tens of pixels from the finger and reads as "the app is broken".

**Do not add `--incognito`** — it wipes IndexedDB, and with it every saved record.

### As an Android app

```bash
npx cap add android
npm run build && npx cap sync && npx cap open android
```

Nothing in the UI reads `window.innerWidth`; every platform capability sits
behind an interface in `src/services/`. That is what keeps this a build step
rather than a rewrite.

---

## Screen map — 10 screens plus records

| # | Route id | File | Notes |
|---|---|---|---|
| 1 | `home` | `pages/Home.tsx` | Language toggle, start, records |
| 2 | `reg` | `pages/Registration.tsx` | Name / age / sex / height / weight / consent. Mints `patientId` |
| 3 | `clinical` | `pages/ClinicalAssessment.tsx` | History, symptom duration, WOMAC-structured questionnaire (one scrollable page) → clinical score |
| 4 | `mv0` | `pages/MovementCapture.tsx` | Knee flexion–extension |
| 5 | `mv1` | same component | Sit to stand |
| 6 | `mv2` | same component | Walking gait |
| 7 | `summary` | `pages/SensorSummary.tsx` | The rig's reported metrics |
| 8 | `xray` | `pages/XrayUpload.tsx` | Camera or file |
| 9 | `compare` | `pages/XrayCompare.tsx` | Side-by-side, opacity slider, KL distribution |
| 10 | `result` | `pages/Result.tsx` | Risk band, guidance, export, save |
| — | `records` | `pages/Records.tsx` | Reachable from home. Tap a row to open the record |
| — | `detail` | `pages/PatientDetail.tsx` | Full saved record, rebuilt from the database |

The flow lives in `src/App.tsx` as a step index, not a router — the kiosk
flow is strictly linear and URL semantics are unwanted in kiosk mode.

---

## Layout — responsive, not a fixed panel

Originally the layout was pinned at exactly 480×320 (the 3.5" Pi panel) and
magnified to fit the browser window via CSS `zoom`, so a desktop preview was
a faithful photograph of the physical panel. That constraint has been
dropped: the app is now a normal responsive layout — full-bleed on a small
touchscreen, a comfortably centered column (`max-width:820px`) on a desktop
browser — built and maintained like an ordinary website rather than a scaled
mockup of one physical device. `main.tsx` no longer computes a zoom factor;
there is no `?scale=` param.

```
header       36 px  (fixed)
content      flexible, scrolls if content is taller than the viewport
action bar   48 px  (fixed)
```

`.content` is always `overflow-y:auto` now — a screen with more fields than
fit the viewport scrolls instead of silently clipping content below the fold
(this was a real bug: Registration's consent checkbox was getting pushed off
a fixed-height panel with no scroll affordance before this changed).

Rules that still apply, since the app can still run on a small touchscreen:
44 px minimum touch target, no dropdowns, body text never below 14 px. Age,
height and weight are steppers **and** directly-typeable number inputs (not
one or the other) so a desktop/keyboard user isn't stuck tapping +/- dozens
of times.

Devanagari gets `line-height: 1.62` via `:lang(hi)` — the शिरोरेखा plus matras
above and below the baseline clip at Latin line heights. **Test the longest
Hindi string on every screen, not the English one.**

---

## Where to plug in real hardware and the real model

### Sensors — `src/services/sensors.ts`

A browser cannot read I²C / SPI / ADC. On the Pi, `sensord.py` reads the
hardware and exposes frames over a localhost WebSocket:

```
sensors → sensord.py → ws://localhost:8765 → SocketSensorSource → MovementCapture
```

Frame contract:

```json
{ "t": 1724000000000, "movement": "flexion",
  "frame": { "angle": 87.4, "velocity": 42.1, "load_l": 0.38,
             "load_r": 0.62, "acoustic_rms": 0.09 } }
```

`MockSensorSource` is the default and ships in production as the demo
fallback. Switch with `setSensorSource(new SocketSensorSource())`.

Metrics are **not computed in the app** — the rig reports them. `sensors.ts`
only parses (`parseDeviceResult`), and `MockSensorSource` emits the same
field names so demo data and hardware data render through identical code.

Presentation lives in `metrics.ts` + `components/MovementMetrics.tsx`, used
by the summary screen, the saved-record view and the PDF alike. Two rules
are enforced there rather than in each screen:

- **`repCount: 0` withholds every timing.** A rig on a bench reports
  `repCount 0` and then `avgRepTime 0.00`. Shown plainly that reads as
  "stood up instantly" — worse than showing nothing — so the timings are
  suppressed and the reason stated (`gateOnZero` in `movements.ts`).
- **An implausibly low headline is flagged.** `maxROM` under 20° means the
  sensor almost certainly is not on a leg (`implausibleBelow`).

`baselineTempC` is tagged `tier: 'qa'` — shown, but greyed and separated,
because it is an engineering field, not a clinical finding.

### KneeFit wearable — `src/services/kneefit/` + `src/hooks/useKneeFit.ts`

The three movement screens (steps 3–5) drive the **KneeFit** knee-cap sensor
directly over Web Bluetooth. Everything is under `src/services/kneefit/`:

| file | role |
|---|---|
| `types.ts` | `KneeFitReading`, `FlexionReading`, `SitToStandReading`, `KneeFitSessionState`, `KneeFitConnectionInfo`, `KneeFitSnapshot` |
| `protocol.ts` | UUIDs (both `65a3` and `b5a3` NUS variants), RX command builders (`calibrate` / `start` / `stop`), TX message classification |
| `parser.ts` | robust buffered TX parser — brace-balanced, string-aware, newline-agnostic, depth- and size-bounded, never throws |
| `bluetooth.ts` | Web Bluetooth transport: pair, discover NUS (tries both variants), subscribe TX, write RX, surface disconnects |
| `session.ts` | the explicit state machine + calibration baseline + per-attempt session boundary |

`useKneeFit()` binds the module-level `KneeFitBluetooth` + `KneeFitSession`
singletons (they outlive the per-screen page mounts) into React state and
exposes `addDevice / calibrate / start / finish / retry / reconnect`.

**State machine:** `DISCONNECTED → CONNECTED → CALIBRATING → CALIBRATED →
READY → RECORDING → FINISHED → RETRYING → READY …`; link loss → `DISCONNECTED`
(or `ERROR` if it drops mid-recording).

**Firmware contract (`sih.ino`, ESP32-S3) — RX is plain ASCII, no newline:**

| App action | RX byte(s) | Firmware |
|---|---|---|
| Calibrate | `C` | zero the IMUs (~6 s) → TX `CALIB_DONE` |
| Start (flexion) | `S1` | needs calibration → TX `RECORDING_START`, records **20 s** |
| Start (sit-to-stand) | `S2` | needs calibration → TX `RECORDING_START`, records **30 s** |
| Stop | `X` | **aborts — discards the buffer, TX `STOPPED`, NO result** |
| (Retake `R` exists but the app re-sends `S1`/`S2` for mode safety) | | |

TX status words: `CALIB_DONE`, `NOT_CALIBRATED`, `RECORDING_START`, `STOPPED`,
then the result JSON — each is one un-delimited notification; the parser
handles bare words and MTU-fragmented JSON alike.

**The test is duration-driven.** After `S1`/`S2` the firmware records for its
fixed 20 s / 30 s and then sends the `{"testType":…}` result on its own —
`RECORDING → FINISHED` happens with no button press. Pressing **Stop** (`X`)
before then throws the data away (this is the firmware's behaviour; to make
Stop finalise-and-send, `stopRecording()` in the firmware must route to the
`PROCESSING` path instead of discarding).

**CALIBRATE ≠ START.** `C` only zeroes; it never starts a recording.

**RETRY / Retake** clears the current attempt, bumps a monotonic `sessionId`
(so attempt N data can't leak into N+1), and re-sends `S1`/`S2`. It keeps the
BLE link, the calibration baseline, patient identity, saved records and the
X-ray result. Stop/Finish never disconnects the device.

The BLE UUIDs: the running device exposes the `…-65a3-…` NUS variant (the
`.ino` source shows `b5a3`, so the flashed binary differs); the app offers
and tries both plus a generic notify/write fallback, so either binds. RX is
`WRITE` (with response), so the app writes with `writeValue()`.

### Bluetooth (legacy sensor rig) — `src/services/ble.ts`

The home screen still has a **Connect** control that pairs the older generic
BLE sensor rig over Web Bluetooth (`store/link.ts` + `MockSensorSource`). It
is independent of the KneeFit workflow above and left in place for
compatibility.

**Your firmware needs to expose the Nordic UART Service:**

| | UUID |
|---|---|
| Service | `6e400001-b5a3-f393-e0a9-e50e24dcca9e` |
| TX (notify, rig → kiosk) | `6e400003-b5a3-f393-e0a9-e50e24dcca9e` |
| RX (write, kiosk → rig) | `6e400002-b5a3-f393-e0a9-e50e24dcca9e` |

Two message kinds arrive on TX, both **newline-delimited JSON**.

**1. The test result** — the rig computes its own metrics and sends one
object per test. This is what the app displays; nothing is recomputed here.

```json
{"testType":"flexion","maxROM":104.6,"maxAngle":109.2,"minAngle":4.6,
 "avgFlexVel":48.3,"avgExtVel":-41.7,"hesitation":2,"crepitusCount":3,
 "crepitusAngles":[71.4,88.2,93.9],"baselineTempC":30.1}

{"testType":"sit_to_stand","repCount":5,"avgRepTime":2.31,"firstRepTime":1.98,
 "lastRepTime":2.74,"fatigueDelta":0.76,"avgRiseVel":34.5,"avgDescentVel":-29.8,
 "hesitation":1,"crepitusCount":1,"crepitusAngles":[66.8],"baselineTempC":30.1}
```

Field names in `movements.ts` are the rig's own, **verbatim**. Renaming them
in the app is how a firmware change becomes a silent display bug.

**The display is spec-driven.** Only fields listed in a movement's `metrics`
array are rendered. Fields the rig sends that are not listed are still
parsed and stored with the capture — nothing is thrown away — but they do
not appear on screen. So adding a reading means adding a `MetricSpec`, and
dropping one from the UI means deleting its spec, not touching the firmware.

`flexAvg` was removed from the flexion spec: the rig no longer carries a flex
strip, and the field is a stale ADC read.

`crepitusAngles` is rendered as angle chips beside the count. Crepitus at
specific points in the arc is the diagnostic signal — a bare count throws
away where in the movement it happened.

**2. Optional live frames**, for the readout during capture. Not required;
if the rig only sends a result at the end, the readout stays blank until
review and everything else works unchanged.

```json
{"t":1724000000000,"movement":"flexion","frame":{"angle":87.4,"velocity":42.1}}
```

The kiosk writes commands to RX, also newline-delimited:

```json
{"cmd":"calibrate","movement":"flexion","ms":3000}
{"cmd":"start","movement":"flexion"}
{"cmd":"stop"}
```

Optionally answer calibrate with `{"type":"calibrated","quality":91}`. If you
don't, the kiosk derives a baseline figure from frame delivery rate over the
still window instead — a real proxy for link quality, not a placeholder.

A BLE notification carries ~20–244 bytes, so one JSON frame arrives in
several pieces. `BleSensorSource` buffers until a newline before parsing.
**Terminate every frame with `\n`** or nothing will ever parse.

**Where Web Bluetooth works**

| Target | Works |
|---|---|
| Pi kiosk (Chromium) | yes — this is the deployment target |
| Desktop Chrome / Edge | yes |
| Android WebView (Capacitor) | **no** |
| iOS Safari / Capacitor | **no** |

Android and iOS builds need `@capacitor-community/bluetooth-le` as a third
implementation of `SensorSource`. Nothing above `services/` changes.

It also needs a **secure context**. `localhost` counts; serving the kiosk
over plain `http://192.168.x.x` makes `navigator.bluetooth` undefined and the
control says so rather than failing silently.

Every failure path falls back to the mock and keeps the flow completable: no
Bluetooth in the browser, chooser dismissed, or the rig dropping mid-camp.
Movement screens show a **DEMO DATA** marker whenever the source is not a
live link, so nobody presents mock numbers believing they are real.

### Clinical / WOMAC intake — `pages/ClinicalAssessment.tsx` + `services/clinicalScore.ts`

Screen 3 collects, on one scrollable page rather than one item per screen:
history (previous injury / surgery / prior musculoskeletal problem), symptom
duration band, and a 24-item WOMAC-structured questionnaire (5 pain, 2
stiffness, 17 physical-function items, each a 5-point None–Extreme severity
picker). Height/weight were added to Registration (screen 2) so BMI can feed
the same score. Next stays disabled until all 24 items are answered, same
"latch" pattern as consent on Registration.

**The item wording is not the licensed WOMAC instrument.** WOMAC is a
copyrighted clinical instrument; the items here follow its published
three-subscale topic structure for prototype purposes only
(`data/womac.ts`), stored with `instrumentVersion: 'womac-unlicensed-prototype'`
so it's traceable and swappable. Before any real deployment, source the
actual licensed item text (and a validated translation for any non-English
locale — none is included here; the screen is English-only by design, see
`clinical.notValidated`).

`computeClinicalScore()` in `clinicalScore.ts` combines the normalized WOMAC
score (60% weight) with age/BMI/history/duration points into one 0–100
score and a `low`/`moderate`/`high` band — **PLACEHOLDER FUSION**, same
status as `computeRisk()` below: indicative thresholds, not a validated or
fitted model. It is stored (`db.clinicalAssessments`) and shown in the
report as its own "Clinical findings" evidence section, but it is **not**
currently blended into the final screening `riskBand` — that multimodal
fusion (clinical + functional + radiographic) is separate, not-yet-built
work.

### X-ray grading — `src/services/inference.ts`

`FixtureBackend` returns a deliberately borderline distribution (0.52 / 0.41
across adjacent grades) and a synthetic overlay, so screen 8 is demonstrable
before the model exists. Replace via `setInference(...)` with TF.js on-device
or a LAN inference server — the interface does not change.

### Risk fusion — `computeRisk()` in `src/services/inference.ts`

**Placeholder.** About 30 lines of thresholds over movement metrics, the
manual score and the KL grade. Swap in the real fusion model. The interface
is the commitment; the arithmetic is not.

### Movement graphics — `src/components/MovementFigure.tsx`

The media slot is polymorphic:

```ts
media: {
  kind: 'svg' | 'video' | 'image',
  src: string,
  sources?: { src: string; type: string }[],   // ordered, for kind 'video'
  poster?: string,
}
```

**Movements 1 and 2 play real footage** from `public/movements/`; movement 3
(gait) is still SVG. All clips are 20 fps and **silent**:

| Movement | Asset | Size | Length | H.264 / VP9 |
|---|---|---|---|---|
| Knee bend | `knee-flexion.*` | 460×460 | 5.3 s | 31 kB / 30 kB |
| Sit to stand | `sit-to-stand.*` | 302×472 | 10.1 s | 73 kB / 88 kB |

The knee-bend clip is encoded at **1.5× slower than the source** so a patient
can follow it. That is a property of the file (`setpts=1.5*PTS`), not a
`playbackRate` set at runtime — the browser reports `rate: 1`. Re-encode
rather than reaching for `playbackRate`, which the Pi has to correct in
software frame by frame.

Sit-to-stand is portrait because the movement is tall — letterboxed in the
landscape stage, which keeps the figure as large as a 236 px stage allows.

Both were cropped to drop a generator watermark in the source clips' corner.

Silent is not a preference. Movement 1 listens for crepitus on the acoustic
sensor — a soundtrack feeds your own audio into the measurement.

All three movement screens are the **same component**, so React reuses one
`<video>` element as you advance between them. Swapping `<source>` children
does not reload a video: without the explicit `load()` in `VideoFigure`, the
element keeps playing the *previous* movement's clip under the new
movement's title. Keep that effect if you touch this file.

Two encodings ship on purpose. H.264 is listed first so the Pi uses its
hardware decoder, but plain Chromium builds are compiled without H.264 and
fail *silently* — the poster frame just sits there forever with no error in
the console. VP9 is the fallback that makes that impossible.

To swap in a different clip: drop the files in `public/movements/`, point the
`sources` array at them, and re-encode with

```bash
# setpts=1.5*PTS slows playback 1.5x; drop it to keep the source pace
ffmpeg -i in.mp4 -vf "crop=W:H:X:Y,setpts=1.5*PTS,scale=-2:472,fps=20" -an \
  -c:v libx264 -profile:v baseline -level 3.0 -pix_fmt yuv420p \
  -crf 30 -movflags +faststart out.mp4
ffmpeg -i out.mp4 -c:v libvpx-vp9 -crf 36 -b:v 0 -cpu-used 4 -an out.webm
ffmpeg -i out.mp4 -vf "select=eq(n\,0)" -frames:v 1 -q:v 6 out.jpg
```

Filenames are stable, so replacing a clip needs no code change — but a
browser that already cached the old file will keep serving it. Hard-reload
(⌘⇧R) after swapping one.

Brief state loops the animation; capture state freezes it or mirrors the live
sensor angle. This is deliberate — a looping animation during capture makes
the patient synchronise to it, which is no longer natural movement and
corrupts gait measurement in particular.

---

## Data — `src/db.ts`

Dexie/IndexedDB. Works in Chromium on the Pi and inside the Capacitor WebView.

```
patients             id, name, age, sex, heightCm, weightKg, consentAt, createdAt
encounters           id, patientId, startedAt, completedAt, locale, synced
clinicalAssessments  id, encounterId, history, durationBand, womacItems, painScore,
                      stiffnessScore, functionScore, womacTotal, bmi, clinicalScore, clinicalBand
captures             id, encounterId, movementId, metrics, quality, createdAt
xrays                id, encounterId, blob, capturedAt
reports              id, encounterId, patientId, riskBand, klGrade, klDist, manualScore, heatmap
```

Schema version 2 (`db.ts`) — `clinicalAssessments` is a new table, additive
only; existing installs migrate with no data loss since Dexie only needs a
version bump for new/changed stores, not for new fields on existing ones.

`reports.heatmap` holds the Grad-CAM overlay as a data URL. It is not an
index, so adding it needed no Dexie migration — and it is what lets a saved
record re-open with both images and re-export a complete PDF weeks later.

Everything writes locally first and carries a `synced` flag. Nothing in the
UI blocks on a network call.

### Report export — `src/services/report.ts` + `src/services/pdf.ts`

Screen 9 offers **PDF** (primary) and **HTML** (secondary). Both render from
one source: `REPORT_CSS` + `reportBody()` in `report.ts`.

The PDF is **rasterised**, not typeset with jsPDF text calls. jsPDF does no
complex text shaping, so Devanagari written through its text API comes out
with broken conjuncts and misplaced matras even after the font is embedded —
and you do not notice until a Hindi reader opens it. `pdf.ts` renders the
report into an offscreen node, rasterises it with html2canvas, and slices it
onto A4 pages. Shaping is whatever the browser already got right, in any
language you add later.

Pagination breaks at `<section class="sec">` boundaries rather than slicing
blindly, so a page break never lands mid-radiograph or orphans a heading from
its table. An element taller than one page is still sliced — that is the
fallback, not the normal path.

`pdf.ts` is loaded with a dynamic `import()` because jsPDF plus html2canvas
are ~600 kB. A kiosk should not pay that on first paint for a button most
sessions press once at the end. Main bundle stays ~339 kB; the PDF chunk is
fetched on the first tap of Download.

The HTML export stays because it is smaller, opens on any phone, and prints
to PDF from the browser if someone wants selectable text.

Both exports are also available from a saved record (`PatientDetail`), which
reconstructs the same `ReportInput` from `patients` + `captures` + `xrays` +
`reports` rather than from session state. One report format, two entry
points — a record exported today and the same record exported next month
produce identical documents.

---

## Adding a language

1. Copy `src/i18n/locales/en.json` to `<code>.json` and translate.
2. Register it in `src/i18n/index.ts` (`LANGS` + `resources`).
3. Bundle a subset font locally if the script needs one — **no CDN**, offline
   is a hard requirement.

Clinical numerals stay Latin in every language. Degrees, seconds, the 0–10
score and the KL grade — Devanagari digits in a clinical readout invite
misreading.

---

## Still open

- **Panel bus.** Run the `dtoverlay` check in `PLAN.md` §2. If SPI, keep SVG
  and do not add video.
- **`computeRisk()`** is a placeholder.
- **`sensord.py`** does not exist yet; the mock stands in.
- **Sync target and analytics dashboard** are out of scope for the kiosk —
  the `synced` flag and outbox shape are in place for them.
