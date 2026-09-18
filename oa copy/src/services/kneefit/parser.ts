/**
 * Robust BLE frame parser for KneeFit TX notifications.
 *
 * A BLE notification carries ~20–244 bytes, so one JSON object arrives across
 * several notifications; several small objects can also arrive in one. The
 * KneeFit is documented to send newline-delimited JSON, but the messages
 * captured from the real device had no trailing newline shown, so this parser
 * does NOT rely on newlines: it scans a rolling buffer and extracts every
 * brace-balanced top-level `{...}` (respecting strings/escapes), then also
 * accepts newline framing as a fast path.
 *
 * Guarantees:
 *   • never throws on malformed / partial / non-JSON input
 *   • malformed fragments are dropped once a later valid object parses past them
 *   • the buffer is bounded (a device that streams garbage can't OOM the tab)
 *   • dev-mode logs malformed payloads; production stays quiet
 */
import {
  isControlMsg, isFrameMsg, isResultMsg, toReading,
} from './protocol';
import type { KneeFitControlMsg, KneeFitInbound } from './types';

const MAX_BUFFER = 16 * 1024;
/** No KneeFit JSON nests anywhere near this deep. */
const MAX_DEPTH = 64;

export class KneeFitParser {
  private buf = '';
  private readonly dev = import.meta.env.DEV;

  /** Feed one raw BLE chunk; get back zero or more classified messages. */
  push(chunk: ArrayBuffer | DataView | Uint8Array | string): KneeFitInbound[] {
    this.buf += typeof chunk === 'string' ? chunk : decode(chunk);
    if (this.buf.length > MAX_BUFFER) {
      // No KneeFit JSON is anywhere near 16 KB, so a buffer this large is a
      // device streaming garbage or dropping delimiters. Drop it whole and
      // resynchronise on the next complete object rather than keeping an
      // unbalanced tail that would poison brace matching.
      this.buf = '';
      if (this.dev) console.warn('[KneeFit] parser buffer overflow — buffer dropped, resyncing');
      return [];
    }
    const out: KneeFitInbound[] = [];

    // A "status word" is a KneeFit TX token like CALIB_DONE / STOPPED /
    // RECORDING_START — letters, digits, underscores, spaces, dots, dashes
    // only. Anything with JSON punctuation (`{}[]":,`) is a JSON fragment.
    const STATUS = /^[\w .-]+$/;

    // 1. Newline-delimited status lines (consume; leave JSON for the brace scan).
    let nl: number;
    while ((nl = this.buf.search(/[\r\n]/)) >= 0) {
      const line = this.buf.slice(0, nl);
      if (/[{}[\]":,]/.test(line)) break;                 // JSON ahead
      this.buf = this.buf.slice(nl + 1);
      const t = line.trim();
      if (t) out.push({ kind: 'text', text: t });
    }

    // 2. Brace-balanced JSON objects.
    for (const jsonText of this.drainObjects()) {
      const msg = this.classify(jsonText);
      if (msg) out.push(msg);
    }

    // 3. A leftover packet that is a clean bare token (no newline, no JSON
    //    punctuation) is a complete status word — one BLE notify == one message.
    //    A JSON fragment (which contains `:` `,` `"` …) is left buffered for
    //    the next chunk instead.
    if (this.buf.length > 0 && this.buf.length <= 40 && STATUS.test(this.buf.trim())) {
      const t = this.buf.trim();
      this.buf = '';
      if (t) out.push({ kind: 'text', text: t });
    }

    return out;
  }

  /** Drop any partial buffered data (used when a session boundary changes). */
  reset(): void {
    this.buf = '';
  }

  // Pull every complete top-level {...} out of the buffer, leaving any
  // trailing partial object behind for the next chunk.
  private drainObjects(): string[] {
    const objs: string[] = [];
    let depth = 0;
    let start = -1;
    let inStr = false;
    let esc = false;
    let consumedTo = 0;

    for (let i = 0; i < this.buf.length; i++) {
      const c = this.buf[i];
      if (inStr) {
        if (esc) esc = false;
        else if (c === '\\') esc = true;
        else if (c === '"') inStr = false;
        continue;
      }
      if (c === '"') { inStr = true; continue; }
      if (c === '{') {
        if (depth === 0) start = i;
        depth++;
        if (depth > MAX_DEPTH) {
          // absurd nesting — the stream lost its framing. Abandon the buffer
          // and resync on the next top-level object.
          if (this.dev) console.warn('[KneeFit] parser depth overflow — buffer dropped, resyncing');
          this.buf = '';
          return objs;
        }
      } else if (c === '}') {
        if (depth > 0) {
          depth--;
          if (depth === 0 && start >= 0) {
            objs.push(this.buf.slice(start, i + 1));
            consumedTo = i + 1;
            start = -1;
          }
        }
      }
    }
    // keep an unterminated object (depth>0) or trailing junk after the last
    // complete object for the next push
    this.buf = depth > 0 && start >= 0 ? this.buf.slice(start) : this.buf.slice(consumedTo);
    return objs;
  }

  private classify(text: string): KneeFitInbound | null {
    let value: unknown;
    try {
      value = JSON.parse(text);
    } catch {
      if (this.dev) console.warn('[KneeFit] TX drop (not JSON):', text.slice(0, 120));
      return null;
    }
    if (typeof value !== 'object' || value === null || Array.isArray(value)) {
      if (this.dev) console.warn('[KneeFit] TX drop (not an object):', text.slice(0, 120));
      return null;
    }
    const o = value as Record<string, unknown>;
    if (isResultMsg(o)) return { kind: 'result', reading: toReading(o) };
    if (isFrameMsg(o)) {
      return {
        kind: 'frame',
        frame: {
          t: num(o.t, Date.now()),
          movement: typeof o.movement === 'string' ? o.movement : '',
          frame: onlyNumbers(o.frame as Record<string, unknown>),
        },
      };
    }
    if (isControlMsg(o)) return { kind: 'control', control: o as unknown as KneeFitControlMsg };
    return { kind: 'unknown', value: o };
  }
}

function decode(chunk: ArrayBuffer | DataView | Uint8Array): string {
  const view =
    chunk instanceof Uint8Array ? chunk :
    chunk instanceof DataView ? new Uint8Array(chunk.buffer, chunk.byteOffset, chunk.byteLength) :
    new Uint8Array(chunk);
  return new TextDecoder().decode(view);
}

function num(v: unknown, fallback: number): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : fallback;
}

function onlyNumbers(o: Record<string, unknown>): Record<string, number> {
  const out: Record<string, number> = {};
  for (const [k, v] of Object.entries(o ?? {})) {
    if (typeof v === 'number' && Number.isFinite(v)) out[k] = v;
  }
  return out;
}
