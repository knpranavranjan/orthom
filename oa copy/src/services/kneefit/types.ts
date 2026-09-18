/**
 * KneeFit wearable — typed data model.
 *
 * Every field name here is the KneeFit firmware's own, verbatim (see the
 * messages observed from the real device and documented in `oa copy/README.md`).
 * Renaming a field in the app is how a firmware change becomes a silent
 * display bug, so the transport/parser never rewrites keys.
 */

/** `testType` the device stamps on every result object. Open set on purpose:
 *  an unknown value must be handled, not crash the UI. */
export type KneeFitTestType = 'flexion' | 'sit_to_stand' | 'gait' | (string & {});

/** Shared tail every result object carries. */
interface KneeFitResultBase {
  testType: KneeFitTestType;
  hesitation?: number;
  crepitusCount?: number;
  crepitusAngles?: number[];
  temperatureDifferenceC?: number;
}

export interface FlexionReading extends KneeFitResultBase {
  testType: 'flexion';
  maxROM?: number;
  maxAngle?: number;
  minAngle?: number;
  avgFlexVel?: number;
  avgExtVel?: number;
}

export interface SitToStandReading extends KneeFitResultBase {
  testType: 'sit_to_stand';
  repCount?: number;
  avgRepTime?: number;
  firstRepTime?: number;
  lastRepTime?: number;
  fatigueDelta?: number;
  avgRiseVel?: number;
  avgDescentVel?: number;
}

export interface GenericReading extends KneeFitResultBase {
  /** any other numeric fields the firmware sends */
  [field: string]: number | number[] | string | undefined;
}

/** A validated end-of-test result from the device. Scalars are split out
 *  from the crepitus angle list by the parser; unknown numeric fields are
 *  kept (nothing the firmware sends is thrown away). */
export interface KneeFitReading {
  testType: KneeFitTestType;
  /** every finite scalar field, keyed by the firmware's own name */
  metrics: Record<string, number>;
  /** knee angles (deg) at which crepitus fired */
  crepitusAngles: number[];
  /** wall-clock ms when this reading was accepted by the app */
  receivedAt: number;
  /** raw JSON, kept for the debug panel only */
  raw: Record<string, unknown>;
}

/** Optional per-sample live frame, for the readout during recording.
 *  `{"t":..,"movement":"flexion","frame":{"angle":87.4,"velocity":42.1}}` */
export interface KneeFitLiveFrame {
  t: number;
  movement: string;
  frame: Record<string, number>;
}

/** Optional control message, e.g. a calibration acknowledgement:
 *  `{"type":"calibrated","quality":91}` */
export interface KneeFitControlMsg {
  type: string;
  quality?: number;
  [k: string]: unknown;
}

/** Discriminated output of the parser for one decoded message. */
export type KneeFitInbound =
  | { kind: 'result'; reading: KneeFitReading }
  | { kind: 'frame'; frame: KneeFitLiveFrame }
  | { kind: 'control'; control: KneeFitControlMsg }
  /** a bare status word the device notified on TX, e.g. "STOPPED", "READY" */
  | { kind: 'text'; text: string }
  | { kind: 'unknown'; value: Record<string, unknown> };

// ─────────────────────────── connection ────────────────────────────

export type KneeFitConnectionState =
  | 'disconnected'
  | 'connecting'
  | 'connected'
  | 'error';

export interface KneeFitConnectionInfo {
  state: KneeFitConnectionState;
  deviceName: string | null;
  /** user-facing, already short enough for the 3.5" panel */
  error: string | null;
  /** which Nordic-UART UUID variant the device actually exposed
   *  ('generic' = a UART-shaped service found under some other UUID) */
  uuidVariant: 'observed-65a3' | 'legacy-b5a3' | 'generic' | null;
}

// ──────────────────────── session state machine ────────────────────

export type KneeFitSessionState =
  | 'DISCONNECTED'
  | 'CONNECTED'
  | 'CALIBRATING'
  | 'CALIBRATED'
  | 'READY'
  | 'RECORDING'
  | 'FINISHED'
  | 'RETRYING'
  | 'ERROR';

/** Baseline established by CALIBRATE — the current physical pose taken as 0. */
export interface KneeFitBaseline {
  /** raw angle (deg) the limb was at when calibrated; live angle is shown
   *  relative to this */
  angle: number;
  /** link-quality figure: firmware-reported if it acked, else derived from
   *  sample delivery over the hold window */
  quality: number;
  /** true only if the device sent `{"type":"calibrated"}` */
  deviceConfirmed: boolean;
  at: number;
}

/** One immutable snapshot the hook exposes to React. Everything the UI
 *  needs, nothing it can mutate. */
export interface KneeFitSnapshot {
  session: KneeFitSessionState;
  connection: KneeFitConnectionInfo;
  /** the movement/test currently selected (KneeFit `testType` / movement id) */
  testType: KneeFitTestType | null;
  baseline: KneeFitBaseline | null;
  /** monotonic id — bumped on every START/RETRY so attempt N data can never
   *  bleed into attempt N+1 */
  sessionId: number;
  /** live values during RECORDING (frame fields, already baseline-adjusted
   *  where meaningful) */
  live: Record<string, number>;
  /** best-known metrics for the CURRENT attempt: live-updated while
   *  RECORDING, frozen on FINISH, empty after RETRY */
  metrics: Record<string, number>;
  crepitusAngles: number[];
  /** recording elapsed seconds (0 unless RECORDING/FINISHED) */
  elapsed: number;
  /** ms of the last accepted inbound message, for the debug panel */
  lastMessageAt: number | null;
  lastRaw: string | null;
  /** literal text of the most recent raw TX notification (debug) */
  lastChunk: string | null;
  /** newest-last event log — commands sent, status words received, transitions.
   *  Mirrors the firmware's serial output for the operator. Capped. */
  log: string[];
}
