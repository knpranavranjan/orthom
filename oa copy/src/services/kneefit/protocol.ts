/**
 * KneeFit BLE protocol — the ACTUAL commands/UUIDs this project uses.
 *
 * ───────────────────────────────────────────────────────────────────────────
 * WHAT IS KNOWN (audit + live device inspection, 2026-08-30)
 * ───────────────────────────────────────────────────────────────────────────
 * The firmware source (`sih.ino`, ESP32-S3) was NOT in this repo; what is
 * known comes from `src/services/ble.ts`, the README, an nRF Connect dump of
 * the running device, and the Arduino serial log.
 *
 *   • Running device advertises name "KNEEFIT" and exposes the Nordic UART
 *     Service under the **`…-65a3-…`** UUID (NOT the standard `b5a3` — the
 *     .ino source shows b5a3, so the flashed binary differs from that source).
 *   • RX  `6e400002-65a3-…`  properties: **WRITE** (with response).
 *   • TX  `6e400003-65a3-…`  properties: NOTIFY, READ. Observed value: the
 *     bare word `STOPPED` — the device notifies plain STATUS WORDS, not only
 *     JSON. The parser handles both.
 *   • CALIBRATE command = the single ASCII char **`C`** written to RX
 *     (operator-confirmed). The firmware also has a hardware calibrate button
 *     on GPIO 17 and auto-calibrates on boot; the serial log prints
 *     "Calibrating... hold knee still." → "Calibration complete. Ready to test."
 *   • Full command set confirmed from sih.ino: C / S1 / S2 / X / R (see
 *     RX_RAW below). Status words on TX: CALIB_DONE, NOT_CALIBRATED,
 *     RECORDING_START, STOPPED, then the result JSON.
 *
 * UUIDs: both `65a3` and `b5a3` are offered to `requestDevice` and tried at
 * GATT time (see `bluetooth.ts`), plus a generic notify/write fallback, so
 * whichever the device exposes will bind.
 */
import type { KneeFitControlMsg, KneeFitLiveFrame, KneeFitReading } from './types';

/** Nordic UART Service — as reported on the KneeFit device card. */
export const NUS_OBSERVED = {
  service: '6e400001-65a3-f393-e0a9-e50e24dcca9e',
  rx: '6e400002-65a3-f393-e0a9-e50e24dcca9e', // WRITE  (kiosk → device)
  tx: '6e400003-65a3-f393-e0a9-e50e24dcca9e', // NOTIFY (device → kiosk)
} as const;

/** Nordic UART Service — standard UUIDs, used by the app's original ble.ts. */
export const NUS_LEGACY = {
  service: '6e400001-b5a3-f393-e0a9-e50e24dcca9e',
  rx: '6e400002-b5a3-f393-e0a9-e50e24dcca9e',
  tx: '6e400003-b5a3-f393-e0a9-e50e24dcca9e',
} as const;

/** Both variants, observed first. */
export const NUS_VARIANTS = [
  { id: 'observed-65a3' as const, ...NUS_OBSERVED },
  { id: 'legacy-b5a3' as const, ...NUS_LEGACY },
];

export const KNEEFIT_NAME_PREFIXES = ['KneeFit', 'Knee', 'KNEEFIT', 'NUS'];

// ───────────────────────────── RX commands ─────────────────────────────
//
// EXACT firmware contract (sih.ino → CommandCallbacks::onWrite):
//
//   String cmd = c->getValue().c_str();
//   if (cmd == "C")  startCalibration();          // zero the IMUs, ~6 s, then TX "CALIB_DONE"
//   else if (cmd == "S1") startRecording(FLEXION);     // needs hasCalibrated; TX "RECORDING_START"; 20 s
//   else if (cmd == "S2") startRecording(SIT_STAND);   // needs hasCalibrated; TX "RECORDING_START"; 30 s
//   else if (cmd == "X")  stopRecording();        // ABORT — buffer discarded, TX "STOPPED", NO result
//   else if (cmd == "R")  retakeTest();           // restart lastMode immediately
//
// Commands are matched with `==` on the raw characteristic value — plain ASCII,
// NO newline, NO JSON.
//
// The test is DURATION-DRIVEN: results (the `{"testType":…}` JSON) are only sent
// when the test runs its full length (20 s flexion / 30 s sit-to-stand) or the
// sample buffer fills. Pressing X before then throws the data away.
export const RX_RAW = {
  CALIBRATE: 'C',
  START_FLEXION: 'S1',
  START_SIT_TO_STAND: 'S2',
  STOP: 'X',        // abort — discards the recording
  RETAKE: 'R',      // firmware restarts lastMode; the app uses S1/S2 instead for mode safety
} as const;

/** Movement id (KneeFit `testType`) → the start command the firmware expects. */
export function startCmdFor(testType: string): string {
  return testType === 'sit_to_stand' ? RX_RAW.START_SIT_TO_STAND : RX_RAW.START_FLEXION;
}

/** Firmware-fixed recording duration for a movement (ms). Flexion 20 s,
 *  sit-to-stand 30 s (`TEST_DURATION_*_MS` in sih.ino). */
export function deviceDurationMs(testType: string): number {
  return testType === 'sit_to_stand' ? 30_000 : 20_000;
}

/** Status words the firmware notifies on the TX characteristic. */
export const TX_STATUS = {
  CALIB_DONE: 'CALIB_DONE',
  NOT_CALIBRATED: 'NOT_CALIBRATED',
  RECORDING_START: 'RECORDING_START',
  STOPPED: 'STOPPED',
} as const;

/** How long the app waits during calibration before falling back to "trust the
 *  'C' write". The firmware takes ~6 s (2 s settle + 350×10 ms sampling + 0.5 s)
 *  before it emits CALIB_DONE, so wait a little past that. */
export const CALIBRATION_HOLD_MS = 7_000;

/** Serialise a raw text command (e.g. 'C', 'S1', 'X') — no newline, as the
 *  firmware compares the raw value with ==. */
export function encodeRaw(text: string): Uint8Array {
  return new TextEncoder().encode(text);
}

// ─────────────────────────── TX classification ─────────────────────────

/** A result object — the device computes its own metrics and sends one per
 *  test. Identified by the `testType` key. */
export function isResultMsg(o: Record<string, unknown>): boolean {
  return typeof o.testType === 'string';
}

/** An optional live frame for the readout during recording. */
export function isFrameMsg(o: Record<string, unknown>): o is KneeFitLiveFrame & Record<string, unknown> {
  return typeof o.frame === 'object' && o.frame !== null;
}

/** An optional control message, e.g. `{"type":"calibrated","quality":91}`. */
export function isControlMsg(o: Record<string, unknown>): o is KneeFitControlMsg & Record<string, unknown> {
  return typeof o.type === 'string' && !isResultMsg(o) && !isFrameMsg(o);
}

/**
 * Split a raw result payload into finite scalars + the crepitus angle list,
 * keeping unknown numeric fields. Mirrors the app's existing
 * `parseDeviceResult` so demo data and hardware data render identically.
 */
export function toReading(o: Record<string, unknown>): KneeFitReading {
  const metrics: Record<string, number> = {};
  let crepitusAngles: number[] = [];
  for (const [k, v] of Object.entries(o)) {
    if (k === 'testType') continue;
    if (k === 'crepitusAngles' && Array.isArray(v)) {
      crepitusAngles = v.filter((n): n is number => typeof n === 'number' && Number.isFinite(n));
      continue;
    }
    if (typeof v === 'number' && Number.isFinite(v)) metrics[k] = v;
  }
  return {
    testType: String(o.testType) as KneeFitReading['testType'],
    metrics,
    crepitusAngles,
    receivedAt: Date.now(),
    raw: o,
  };
}
