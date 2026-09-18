/**
 * KneeFit session manager — the explicit state machine.
 *
 *   DISCONNECTED → CONNECTED → CALIBRATING → CALIBRATED → READY
 *     → RECORDING → FINISHED → RETRYING → READY → RECORDING → …
 *   (any) → DISCONNECTED on link loss
 *   RECORDING → ERROR on BLE failure, then reconnect/retry
 *
 * Responsibilities:
 *   • own the session state + transitions (nothing else may set state)
 *   • establish the CALIBRATE baseline (device ack if any, else from the
 *     live stream) and reset all per-session metrics/timers/buffers
 *   • stamp every attempt with a monotonic sessionId so attempt N data can
 *     never leak into attempt N+1
 *   • fold inbound readings/frames into the current attempt only
 *   • FINISH freezes; RETRY wipes the attempt but keeps the BLE link,
 *     the baseline, and anything outside this session
 *
 * It does NOT touch Web Bluetooth directly (that's `bluetooth.ts`) and does
 * NOT know about React (that's `useKneeFit.ts`). Commands go out through the
 * `send` callback it's constructed with.
 */
import {
  CALIBRATION_HOLD_MS, RX_RAW, TX_STATUS, deviceDurationMs, encodeRaw, startCmdFor,
} from './protocol';
import { KneeFitParser } from './parser';
import type {
  KneeFitBaseline,
  KneeFitConnectionInfo,
  KneeFitInbound,
  KneeFitSessionState,
  KneeFitSnapshot,
  KneeFitTestType,
} from './types';

const log = (...a: unknown[]) => { if (import.meta.env.DEV) console.log('[KneeFit]', ...a); };

/** Returns true if the bytes reached the RX characteristic. */
type Send = (bytes: Uint8Array) => boolean | void | Promise<boolean | void>;

const EMPTY_CONN: KneeFitConnectionInfo = {
  state: 'disconnected', deviceName: null, error: null, uuidVariant: null,
};

/** angle-like fields the baseline is subtracted from for the live readout */
const RELATIVE_FIELDS = new Set(['angle', 'kneeAngle', 'maxAngle', 'minAngle']);

export class KneeFitSession {
  private state: KneeFitSessionState = 'DISCONNECTED';
  private conn: KneeFitConnectionInfo = { ...EMPTY_CONN };
  private testType: KneeFitTestType | null = null;
  private baseline: KneeFitBaseline | null = null;
  private sessionId = 0;

  private live: Record<string, number> = {};
  private metrics: Record<string, number> = {};
  private crepitusAngles: number[] = [];
  private elapsed = 0;
  private lastMessageAt: number | null = null;
  private lastRaw: string | null = null;
  private logLines: string[] = [];
  private static readonly LOG_CAP = 24;

  private readonly parser = new KneeFitParser();
  private tickTimer: ReturnType<typeof setInterval> | null = null;
  private calibrateTimer: ReturnType<typeof setTimeout> | null = null;
  private t0 = 0;
  /** after FINISH, accept an in-flight end-of-test result until this instant,
   *  then hard-freeze so no later packet can mutate a shown result */
  private freezeAt = 0;
  private static readonly FINISH_GRACE_MS = 2000;
  /** ms past the firmware's fixed test duration after which the app finishes
   *  the movement itself. The firmware's PROCESSING is sub-millisecond, so the
   *  result JSON lands ~immediately after the duration; if it hasn't arrived a
   *  few seconds later it was lost (BLE fragmentation) and never will — stop
   *  anyway so the operator gets the same "test over" behaviour as movement 1. */
  private static readonly AUTO_FINISH_GRACE_MS = 3500;
  /** if the app auto-finished with NO data, keep accepting a matching
   *  end-of-test result for this long — a fragmented BLE result (esp. the
   *  larger sit-to-stand JSON over a small MTU) can still be in flight. */
  private static readonly LATE_RESULT_MS = 25000;
  private lateResultUntil = 0;
  /** literal text of the most recent TX notification, for the debug panel */
  private lastChunk: string | null = null;

  // calibration working state
  private calibrating = false;
  private calibSamples: number[] = [];
  private calibFrameCount = 0;
  /** any TX message (frame / result / control) seen after the 'C' write —
   *  the device responding IS the confirmation that it zeroed */
  private calibRespCount = 0;
  /** whether the 'C' byte actually reached the RX characteristic */
  private calibWriteOk = false;
  private calibDeviceQuality: number | null = null;
  private calibHoldMs = CALIBRATION_HOLD_MS;
  /** one-shot "device never sent a result" warning while RECORDING */
  private warnedNoResult = false;

  private listeners = new Set<(s: KneeFitSnapshot) => void>();

  /** test-only timing overrides */
  private readonly durationOverrideMs?: number;
  private readonly autoFinishGraceMs: number;

  constructor(
    private send: Send,
    opts?: { durationMsOverride?: number; autoFinishGraceMs?: number },
  ) {
    this.durationOverrideMs = opts?.durationMsOverride;
    this.autoFinishGraceMs = opts?.autoFinishGraceMs ?? KneeFitSession.AUTO_FINISH_GRACE_MS;
  }

  private testDurationMs(): number {
    return this.durationOverrideMs ?? deviceDurationMs(this.testType ?? 'flexion');
  }

  // ───────────────────────── subscription ─────────────────────────

  subscribe(fn: (s: KneeFitSnapshot) => void): () => void {
    this.listeners.add(fn);
    fn(this.snapshot());
    return () => this.listeners.delete(fn);
  }

  snapshot(): KneeFitSnapshot {
    return {
      session: this.state,
      connection: { ...this.conn },
      testType: this.testType,
      baseline: this.baseline ? { ...this.baseline } : null,
      sessionId: this.sessionId,
      live: { ...this.live },
      metrics: { ...this.metrics },
      crepitusAngles: [...this.crepitusAngles],
      elapsed: this.elapsed,
      lastMessageAt: this.lastMessageAt,
      lastRaw: this.lastRaw,
      lastChunk: this.lastChunk,
      log: [...this.logLines],
    };
  }

  private hasMetrics(): boolean {
    return Object.keys(this.metrics).length > 0;
  }

  private emit(): void {
    const snap = this.snapshot();
    for (const fn of this.listeners) fn(snap);
  }

  /** Append a line to the operator-visible event log (mirrors the firmware's
   *  serial output). `emit` is left to the caller. */
  private pushLog(line: string): void {
    const ts = new Date().toLocaleTimeString([], { hour12: false });
    this.logLines.push(`${ts}  ${line}`);
    if (this.logLines.length > KneeFitSession.LOG_CAP) {
      this.logLines.splice(0, this.logLines.length - KneeFitSession.LOG_CAP);
    }
  }

  private set(next: KneeFitSessionState): void {
    if (next === this.state) return;
    log('state', this.state, '→', next);
    this.state = next;
    this.emit();
  }

  // ───────────────────────── connection events ────────────────────

  onConnecting(): void {
    this.conn = { ...EMPTY_CONN, state: 'connecting' };
    this.emit();
  }

  onConnected(deviceName: string | null, uuidVariant: KneeFitConnectionInfo['uuidVariant']): void {
    this.conn = { state: 'connected', deviceName, error: null, uuidVariant };
    this.parser.reset();
    this.pushLog(`● connected — ${deviceName ?? 'KneeFit'}${uuidVariant ? ` (${uuidVariant})` : ''}`);
    this.set('CONNECTED');
  }

  onConnectError(message: string): void {
    this.conn = { ...EMPTY_CONN, state: 'error', error: shorten(message) };
    // only escalate the session to ERROR if we were mid-recording
    this.set(this.state === 'RECORDING' ? 'ERROR' : 'DISCONNECTED');
  }

  /** Physical link lost (gattserverdisconnected). Preserve whatever data the
   *  current attempt already gathered; stop the clock. */
  onLinkLost(): void {
    log('link lost');
    this.stopClock();
    this.calibrating = false;
    this.clearCalibrateTimer();
    const wasRecording = this.state === 'RECORDING';
    this.conn = {
      state: 'error', deviceName: this.conn.deviceName,
      error: 'KneeFit disconnected.', uuidVariant: this.conn.uuidVariant,
    };
    this.pushLog('⚠ link lost');
    this.set(wasRecording ? 'ERROR' : 'DISCONNECTED');
  }

  onDisconnectByUser(): void {
    this.stopClock();
    this.clearCalibrateTimer();
    this.calibrating = false;
    this.parser.reset();
    this.conn = { ...EMPTY_CONN };
    this.testType = null;
    this.baseline = null;
    this.live = {};
    this.metrics = {};
    this.crepitusAngles = [];
    this.elapsed = 0;
    this.set('DISCONNECTED');
  }

  // ───────────────────────── test selection ───────────────────────

  selectTest(testType: KneeFitTestType): void {
    if (testType === this.testType) return;            // idempotent — no churn on re-render
    if (this.state === 'RECORDING' || this.state === 'CALIBRATING') return;
    this.testType = testType;
    // The firmware's calibration ('C' → hasCalibrated) is device-global: one
    // calibration covers every movement. So keep the baseline across movements
    // — the operator calibrates once, then each movement is immediately READY.
    this.wipeAttempt();
    this.freezeAt = 0;
    this.elapsed = 0;
    this.stopClock();
    if (this.conn.state === 'connected') {
      this.conn = { ...this.conn, error: null };
      this.set(this.baseline ? 'READY' : 'CONNECTED');
    } else if (this.state !== 'DISCONNECTED') {
      this.set('DISCONNECTED');
    } else {
      this.emit();
    }
  }

  // ───────────────────────── CALIBRATE ────────────────────────────

  /**
   * Take the current physical pose as zero.
   *
   * Writes the raw single-char command 'C' to RX — the firmware zeroes every
   * sensor and starts streaming values relative to that pose. The device
   * responding within the hold window IS the confirmation (all values arrive
   * at ~0). A frontend baseline is also captured from any angle field as a
   * belt-and-braces fallback. Reaches CALIBRATED only if the device responded
   * OR a valid sample was seen — success is never faked.
   */
  async calibrate(holdMs: number = CALIBRATION_HOLD_MS): Promise<void> {
    if (!this.testType) { log('calibrate ignored — no test selected'); return; }
    if (this.conn.state !== 'connected') { log('calibrate ignored — not connected'); return; }
    if (this.state === 'RECORDING') return;

    // calibration only ever resets the CURRENT movement session
    this.wipeAttempt();
    this.baseline = null;
    this.calibrating = true;
    this.calibSamples = [];
    this.calibFrameCount = 0;
    this.calibRespCount = 0;
    this.calibDeviceQuality = null;
    this.calibHoldMs = holdMs;
    this.parser.reset();               // start the calibration window clean
    this.pushLog(`→ '${RX_RAW.CALIBRATE}'  (calibrate / zero)`);
    this.pushLog('… calibrating — hold the knee still');
    this.set('CALIBRATING');

    log("calibrate: sending 'C', hold still", holdMs, 'ms');
    this.calibWriteOk = (await this.send(encodeRaw(RX_RAW.CALIBRATE))) !== false;
    if (!this.calibWriteOk) this.pushLog("⚠ could not write 'C' to RX");

    this.clearCalibrateTimer();
    this.calibrateTimer = setTimeout(() => this.finishCalibration(holdMs), holdMs);
  }

  private finishCalibration(holdMs: number): void {
    this.clearCalibrateTimer();
    if (!this.calibrating) return;
    this.calibrating = false;

    const deviceAck = this.calibDeviceQuality !== null;          // explicit {"type":"calibrated"}
    const deviceResponded = this.calibRespCount > 0;             // any telemetry after 'C'
    const haveSample = this.calibSamples.length > 0;

    if (!deviceAck && !deviceResponded && !haveSample) {
      if (!this.calibWriteOk) {
        // could not even deliver 'C' (no RX characteristic / GATT write error)
        this.conn = {
          ...this.conn,
          error: "Could not send calibration command. Reconnect the KneeFit.",
        };
        this.pushLog('✗ calibration failed — command not delivered');
        this.set('CONNECTED');
        return;
      }
      // 'C' was delivered; the firmware zeroes itself on receipt whether or not
      // it echoes telemetry between tests. Trust the confirmed hardware command.
      this.baseline = { angle: 0, quality: 100, deviceConfirmed: true, at: Date.now() };
      this.wipeAttempt();
      log("calibrated: zeroed via 'C' (no telemetry echo)");
      this.pushLog("✓ calibrated — zero set (via 'C')");
      this.set('CALIBRATED');
      this.set('READY');
      return;
    }

    // the device zeroes itself on 'C', so the reference angle is 0; the median
    // of any samples seen is only a sanity value.
    const angle = haveSample ? median(this.calibSamples) : 0;
    const deviceConfirmed = deviceAck || deviceResponded;
    let quality: number;
    if (deviceAck) {
      quality = clamp(Math.round(this.calibDeviceQuality as number), 0, 100);
    } else if (deviceResponded) {
      // telemetry flowing after 'C' — grade it by how many messages landed
      const expected = Math.max(1, Math.round((holdMs / 1000) * 5));
      quality = clamp(Math.round((this.calibRespCount / expected) * 100), 40, 100);
    } else {
      const expected = Math.max(1, Math.round((holdMs / 1000) * 10));
      quality = clamp(Math.round((this.calibFrameCount / expected) * 100), 0, 100);
    }

    this.baseline = { angle: deviceConfirmed ? 0 : angle, quality, deviceConfirmed, at: Date.now() };
    this.wipeAttempt();
    log('calibrated: zero set · quality', quality,
        deviceAck ? '(device ack)' : deviceResponded ? '(device telemetry)' : '(frontend)',
        '· responses', this.calibRespCount);
    this.pushLog(`✓ calibrated — zero set · quality ${quality}%`);
    this.set('CALIBRATED');
    // CALIBRATED auto-advances to READY — the user then prepares and hits START
    this.set('READY');
  }

  private clearCalibrateTimer(): void {
    if (this.calibrateTimer) { clearTimeout(this.calibrateTimer); this.calibrateTimer = null; }
  }

  // ───────────────────────── START / RECORDING ────────────────────

  /**
   * Begin a recording. Sends S1 (flexion) or S2 (sit-to-stand). The firmware
   * is DURATION-DRIVEN — it records for 20 s / 30 s then sends the result JSON
   * on its own; `handle()` promotes RECORDING → FINISHED when that arrives.
   */
  async start(): Promise<void> {
    if (this.state !== 'READY' && this.state !== 'CALIBRATED' && this.state !== 'RETRYING') {
      log('start ignored in state', this.state); return;
    }
    if (!this.testType) return;
    await this.beginRecording('start');
  }

  private async beginRecording(reason: 'start' | 'retry'): Promise<void> {
    const testType = this.testType as string;
    this.sessionId += 1;              // ← hard session boundary
    this.wipeAttempt();
    this.parser.reset();
    this.freezeAt = 0;
    this.lateResultUntil = 0;
    this.warnedNoResult = false;
    this.elapsed = 0;
    this.t0 = Date.now();
    this.startClock();
    this.set('RECORDING');
    const c = startCmdFor(testType);
    log('recording', reason, '· session', this.sessionId, '·', testType, '· cmd', c);
    this.pushLog(`→ ${c}  (${reason} ${testType} · session #${this.sessionId})`);
    this.pushLog(`… recording — auto-completes in ${Math.round(deviceDurationMs(testType) / 1000)}s`);
    const ok = await this.send(encodeRaw(c));
    if (ok === false) {
      this.pushLog(`⚠ could not write '${c}' to RX`);
      this.stopClock();
      this.conn = { ...this.conn, error: 'Could not send the start command.' };
      this.set(this.baseline ? 'READY' : 'CONNECTED');
    }
  }

  // ───────────────────── STOP (abort) ─────────────────────────────

  /**
   * Sends 'X'. Per the firmware this ABORTS the recording — the sample buffer
   * is discarded and NO result is produced (it only notifies "STOPPED").
   * If a result has already arrived, there is nothing to stop and we keep it.
   */
  async finish(): Promise<void> {
    if (this.state !== 'RECORDING') { log('stop ignored in state', this.state); return; }

    if (Object.keys(this.metrics).length > 0) {
      // the device already sent its result before the user hit Stop
      this.stopClock();
      this.freezeAt = Date.now() + KneeFitSession.FINISH_GRACE_MS;
      this.pushLog('→ stop (result already received)');
      this.set('FINISHED');
      return;
    }

    this.stopClock();
    this.pushLog('→ X  (stop — recording aborted, no result)');
    const ok = await this.send(encodeRaw(RX_RAW.STOP));
    if (ok === false) this.pushLog("⚠ could not write 'X' to RX");
    // firmware will notify "STOPPED"; go back to READY (calibration still valid)
    this.conn = { ...this.conn, error: null };
    this.set(this.baseline ? 'READY' : 'CONNECTED');
  }

  // ───────────────────────── RETRY / RETAKE ──────────────────────

  /**
   * Retake this movement. Clears the current attempt and — if connected —
   * immediately re-sends S1/S2 (mode-explicit; safer than the firmware's 'R'
   * which restarts whatever `lastMode` was). Ends in RECORDING.
   * Keeps the BLE link, the calibration baseline, and everything outside
   * this movement session.
   */
  retry(): void {
    if (this.state !== 'FINISHED' && this.state !== 'ERROR' && this.state !== 'READY') {
      log('retry ignored in state', this.state); return;
    }
    this.stopClock();
    this.parser.reset();
    this.wipeAttempt();
    this.freezeAt = 0;
    this.elapsed = 0;
    this.pushLog('↻ retake');
    this.set('RETRYING');

    if (this.conn.state !== 'connected') { this.set('DISCONNECTED'); return; }
    this.conn = { ...this.conn, error: null };
    if (!this.baseline || !this.testType) { this.set('CONNECTED'); return; }
    void this.beginRecording('retry');
  }

  // ───────────────────────── inbound BLE data ─────────────────────

  ingest(chunk: DataView): void {
    const raw = new TextDecoder().decode(chunk);
    this.lastChunk = raw;
    if (import.meta.env.DEV) console.log('[KneeFit] TX chunk', JSON.stringify(raw));
    for (const msg of this.parser.push(chunk)) this.handle(msg);
  }

  private handle(msg: KneeFitInbound): void {
    this.lastMessageAt = Date.now();

    // While calibrating, ANY inbound message is the device answering 'C'.
    if (this.calibrating && msg.kind !== 'control') this.calibRespCount += 1;

    if (msg.kind === 'control') {
      this.lastRaw = JSON.stringify(msg.control);
      if (/calib/i.test(msg.control.type) && this.calibrating) {
        this.calibRespCount += 1;
        this.calibDeviceQuality = typeof msg.control.quality === 'number' ? msg.control.quality : 100;
        log('calibration acknowledged by device, quality', this.calibDeviceQuality);
      }
      this.emit();
      return;
    }

    // Bare status word notified on TX. The firmware sends exactly:
    //   CALIB_DONE · NOT_CALIBRATED · RECORDING_START · STOPPED
    if (msg.kind === 'text') {
      this.lastRaw = msg.text;
      this.pushLog(`← ${msg.text}`);
      const w = msg.text.trim().toUpperCase();

      if (this.calibrating) {
        const done = w === TX_STATUS.CALIB_DONE
          || /(complet|zero\s*set|calibrated)/i.test(msg.text);
        if (done) {
          if (this.calibDeviceQuality === null) this.calibDeviceQuality = 100;
          log('calibration confirmed by device:', msg.text);
          this.finishCalibration(this.calibHoldMs);      // short-circuit the hold
          return;
        }
        this.emit();
        return;
      }

      if (w === TX_STATUS.NOT_CALIBRATED) {
        // device refused S1/S2 — its calibration was lost (e.g. it rebooted)
        this.stopClock();
        this.baseline = null;
        this.conn = { ...this.conn, error: 'KneeFit not calibrated — calibrate again.' };
        this.set('CONNECTED');
        return;
      }
      if (w === TX_STATUS.RECORDING_START) {
        if (this.state !== 'RECORDING') this.set('RECORDING');
        this.emit();
        return;
      }
      if (w === TX_STATUS.STOPPED) {
        // our 'X' abort was acknowledged (or the device stopped itself)
        if (this.state === 'RECORDING') {
          this.stopClock();
          this.set(this.baseline ? 'READY' : 'CONNECTED');
        }
        this.emit();
        return;
      }
      this.emit();
      return;
    }

    if (msg.kind === 'frame') {
      this.lastRaw = JSON.stringify(msg.frame);
      const f = msg.frame.frame;
      if (this.calibrating) {
        this.calibFrameCount += 1;
        const a = pickAngle(f);
        if (a !== null) this.calibSamples.push(a);
        return;
      }
      if (this.state === 'RECORDING') {
        for (const [k, v] of Object.entries(f)) {
          this.live[k] = RELATIVE_FIELDS.has(k) && this.baseline ? v - this.baseline.angle : v;
        }
        this.emit();
      }
      return;
    }

    if (msg.kind === 'result') {
      this.lastRaw = JSON.stringify(msg.reading.raw);
      if (this.calibrating) {
        // a result-shaped stream after 'C' — seed the baseline from any angle
        const m = msg.reading.metrics;
        for (const k of ['angle', 'kneeAngle', 'maxAngle', 'minAngle']) {
          if (typeof m[k] === 'number') { this.calibSamples.push(m[k]); break; }
        }
        return;
      }
      // adopt the device's own metrics for the CURRENT attempt only.
      // NON-cumulative: a fresh result REPLACES, never adds.
      const typeMatches =
        !this.testType || !msg.reading.testType ||
        this.testType === msg.reading.testType;
      const noMetricsYet = Object.keys(this.metrics).length === 0;
      // a result that lands AFTER the app auto-finished with no data — still
      // ours as long as no metrics have been shown and the movement matches
      const lateOk =
        this.state === 'FINISHED' && noMetricsYet && Date.now() < this.lateResultUntil;
      const inAttempt =
        typeMatches && (
          this.state === 'RECORDING' ||
          (this.state === 'FINISHED' && Date.now() < this.freezeAt) ||
          lateOk
        );

      if (inAttempt) {
        if (!this.testType) this.testType = msg.reading.testType;
        this.metrics = { ...msg.reading.metrics };
        this.crepitusAngles = [...msg.reading.crepitusAngles];
        this.lateResultUntil = 0;
        log('result folded in', msg.reading.testType, this.metrics);
        const keys = Object.keys(msg.reading.metrics);
        this.pushLog(`← result ${msg.reading.testType} · ${keys.length} field${keys.length === 1 ? '' : 's'}`);
        // the firmware-timed test finished on its own → freeze it
        if (this.state === 'RECORDING') {
          this.stopClock();
          this.freezeAt = Date.now() + KneeFitSession.FINISH_GRACE_MS;
          this.pushLog('✓ test complete — result received');
          this.set('FINISHED');
        } else {
          this.pushLog('✓ result received (late)');
          this.emit();
        }
      } else {
        log('result ignored — no active attempt (state', this.state,
            '· type', msg.reading.testType, 'vs', this.testType + ')');
      }
      return;
    }

    // unknown — keep for the debug panel, never act on it
    this.lastRaw = JSON.stringify(msg.value);
    this.pushLog(`← (unrecognised) ${this.lastRaw.slice(0, 60)}`);
    if (import.meta.env.DEV) console.warn('[KneeFit] unknown TX message', msg.value);
    this.emit();
  }

  // ───────────────────────── helpers ──────────────────────────────

  /** Wipe everything scoped to one attempt. Does NOT touch baseline/link. */
  private wipeAttempt(): void {
    this.live = {};
    this.metrics = {};
    this.crepitusAngles = [];
    this.calibSamples = [];
    this.calibFrameCount = 0;
  }

  private startClock(): void {
    this.stopClock();
    this.tickTimer = setInterval(() => {
      // defensive: the clock only ever runs during RECORDING. If state left
      // RECORDING by any path, stop — never let the timer climb forever.
      if (this.state !== 'RECORDING') { this.stopClock(); return; }

      this.elapsed = (Date.now() - this.t0) / 1000;

      if (this.testType) {
        const overMs = this.elapsed * 1000 - this.testDurationMs();

        if (!this.warnedNoResult && overMs > 300 && !this.hasMetrics()) {
          this.warnedNoResult = true;
          this.pushLog('… test duration reached — finishing');
        }
        // Safety net: the firmware-timed test is over and no result has landed
        // (lost notification / BLE fragmentation). Finish the movement ourselves
        // so the operator gets the same "test over" behaviour as movement 1 —
        // with real metrics if any arrived, or an honest "no data" otherwise.
        // Never fabricates values.
        if (overMs > this.autoFinishGraceMs) {
          this.autoFinish();
          return;
        }
      }
      this.emit();
    }, 200);
  }

  private autoFinish(): void {
    this.stopClock();
    this.freezeAt = Date.now() + KneeFitSession.FINISH_GRACE_MS;
    if (this.hasMetrics()) {
      this.lateResultUntil = 0;
      this.pushLog('✓ test complete (auto — result arrived late)');
    } else {
      // no data yet — the firmware result may still be arriving over a slow /
      // fragmented BLE link. Stop the countdown now, but keep listening.
      this.lateResultUntil = Date.now() + KneeFitSession.LATE_RESULT_MS;
      this.pushLog('… duration over — waiting for the result (will fill in when it lands)');
    }
    this.set('FINISHED');
  }

  private stopClock(): void {
    if (this.tickTimer) { clearInterval(this.tickTimer); this.tickTimer = null; }
  }

  dispose(): void {
    this.stopClock();
    this.clearCalibrateTimer();
    this.listeners.clear();
  }
}

// ─────────────────────────── pure utils ────────────────────────────

function pickAngle(frame: Record<string, number>): number | null {
  for (const k of ['angle', 'kneeAngle', 'relAngle', 'theta']) {
    if (typeof frame[k] === 'number' && Number.isFinite(frame[k])) return frame[k];
  }
  return null;
}

function median(xs: number[]): number {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function clamp(n: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, n));
}

function shorten(msg: string): string {
  const m = msg.replace(/^Error:\s*/, '').trim();
  return m.length > 90 ? m.slice(0, 88) + '…' : m;
}
