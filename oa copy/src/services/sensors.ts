/**
 * A browser cannot read I2C / SPI / ADC. The rig computes its own metrics
 * and reports one summary object per test; a transport may additionally
 * stream frames for the live readout, but is not required to.
 *
 * Everything above this file talks to the interface, never the transport —
 * which is what lets the mock, BLE, and the Pi's WebSocket be swapped
 * without touching a screen.
 */

export interface SensorFrame {
  t: number;
  movement: string;
  frame: Record<string, number>;
}

/** One test's result, exactly as the rig reports it. */
export interface DeviceResult {
  testType: string;
  [field: string]: number | number[] | string;
}

export interface CaptureResult {
  /** device field name -> value, verbatim */
  metrics: Record<string, number>;
  /** knee angles at which crepitus was detected */
  angles: number[];
  quality: number;
  samples: number;
}

export interface SensorSource {
  readonly name: string;
  calibrate(movementId: string, ms: number): Promise<number>;
  start(movementId: string, onFrame: (f: SensorFrame) => void): void;
  stop(): CaptureResult;
  dispose(): void;
}

export const EMPTY_CAPTURE: CaptureResult = { metrics: {}, angles: [], quality: 0, samples: 0 };

/**
 * Split a rig result payload into scalars and the crepitus angle list.
 * Unknown fields are kept: a firmware that starts reporting something new
 * should show up on screen, not be silently dropped.
 */
export function parseDeviceResult(res: DeviceResult): { metrics: Record<string, number>; angles: number[] } {
  const metrics: Record<string, number> = {};
  let angles: number[] = [];
  for (const [k, v] of Object.entries(res)) {
    if (k === 'testType') continue;
    if (k === 'crepitusAngles' && Array.isArray(v)) {
      angles = v.filter((n): n is number => typeof n === 'number');
      continue;
    }
    if (typeof v === 'number' && Number.isFinite(v)) metrics[k] = v;
  }
  return { metrics, angles };
}

const rnd = (a: number, b: number) => a + Math.random() * (b - a);
const r1 = (n: number) => Number(n.toFixed(1));
const r2 = (n: number) => Number(n.toFixed(2));

/**
 * Produces the same field names the rig does, so the summary screen, the
 * report and the risk fusion cannot behave differently on demo data.
 */
export function mockResult(movementId: string): { metrics: Record<string, number>; angles: number[] } {
  if (movementId === 'flexion') {
    const maxAngle = r1(rnd(112, 134));
    const minAngle = r1(rnd(2, 9));
    const angles = Math.random() < 0.6 ? [r1(rnd(68, 82)), r1(rnd(88, 99))] : [];
    return {
      metrics: {
        maxROM: r1(maxAngle - minAngle),
        maxAngle, minAngle,
        avgFlexVel: r1(rnd(42, 68)),
        avgExtVel: r1(-rnd(38, 60)),
        hesitation: Math.random() < 0.4 ? 1 : 0,
        crepitusCount: angles.length,
        temperatureDifferenceC: r1(rnd(0.4, 3.8)),
      },
      angles,
    };
  }
  if (movementId === 'sit_to_stand') {
    const first = r2(rnd(1.4, 2.0));
    const last = r2(first + rnd(0.05, 0.6));
    const angles = Math.random() < 0.35 ? [r1(rnd(60, 85))] : [];
    return {
      metrics: {
        repCount: 5,
        avgRepTime: r2((first + last) / 2),
        firstRepTime: first,
        lastRepTime: last,
        fatigueDelta: r2(last - first),
        avgRiseVel: r1(rnd(30, 52)),
        avgDescentVel: r1(-rnd(26, 44)),
        hesitation: Math.random() < 0.3 ? 1 : 0,
        crepitusCount: angles.length,
        temperatureDifferenceC: r1(rnd(0.4, 3.8)),
      },
      angles,
    };
  }
  return {
    metrics: {
      cadence: Math.round(rnd(94, 112)),
      strideLen: Math.round(rnd(104, 132)),
      stepSymmetry: Math.round(rnd(86, 99)),
    },
    angles: [],
  };
}

/** Replays a plausible test. Ships in production as the demo fallback. */
export class MockSensorSource implements SensorSource {
  readonly name = 'mock';
  private timer: number | null = null;
  private t0 = 0;
  private count = 0;
  private movement = '';
  private quality = 0;

  async calibrate(_movementId: string, ms: number): Promise<number> {
    await new Promise((r) => setTimeout(r, ms));
    this.quality = Math.round(rnd(72, 97));
    return this.quality;
  }

  start(movementId: string, onFrame: (f: SensorFrame) => void): void {
    this.movement = movementId;
    this.count = 0;
    this.t0 = Date.now();
    this.timer = window.setInterval(() => {
      const el = (Date.now() - this.t0) / 1000;
      let frame: Record<string, number>;
      if (movementId === 'flexion') {
        const phase = (Math.sin((el / 3) * Math.PI * 2 - Math.PI / 2) + 1) / 2;
        frame = { angle: 8 + phase * 122, velocity: Math.abs(Math.cos((el / 3) * Math.PI * 2) * 62) };
      } else if (movementId === 'sit_to_stand') {
        frame = { elapsed: el };
      } else {
        frame = { cadence: 100 + Math.sin(el / 2) * 6 };
      }
      this.count++;
      onFrame({ t: Date.now(), movement: movementId, frame });
    }, 100);
  }

  stop(): CaptureResult {
    if (this.timer !== null) { window.clearInterval(this.timer); this.timer = null; }
    const { metrics, angles } = mockResult(this.movement);
    return { metrics, angles, quality: this.quality, samples: this.count };
  }

  dispose(): void {
    if (this.timer !== null) { window.clearInterval(this.timer); this.timer = null; }
  }
}

/** Talks to sensord.py on the Pi. Same interface, different transport. */
export class SocketSensorSource implements SensorSource {
  readonly name = 'socket';
  private ws: WebSocket | null = null;
  private samples = 0;
  private quality = 0;
  private result: { metrics: Record<string, number>; angles: number[] } | null = null;
  constructor(private url = 'ws://localhost:8765') {}

  private connect(): Promise<WebSocket> {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) return Promise.resolve(this.ws);
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(this.url);
      ws.onopen = () => { this.ws = ws; resolve(ws); };
      ws.onerror = () => reject(new Error('sensord unreachable'));
    });
  }

  async calibrate(movementId: string, ms: number): Promise<number> {
    const ws = await this.connect();
    ws.send(JSON.stringify({ cmd: 'calibrate', movement: movementId, ms }));
    return new Promise((resolve) => {
      const done = setTimeout(() => resolve(this.quality), ms + 2000);
      const onMsg = (e: MessageEvent) => {
        const m = JSON.parse(e.data as string);
        if (m.type === 'calibrated') {
          clearTimeout(done);
          ws.removeEventListener('message', onMsg);
          this.quality = m.quality ?? 0;
          resolve(this.quality);
        }
      };
      ws.addEventListener('message', onMsg);
    });
  }

  start(movementId: string, onFrame: (f: SensorFrame) => void): void {
    this.samples = 0;
    this.result = null;
    void this.connect().then((ws) => {
      ws.send(JSON.stringify({ cmd: 'start', movement: movementId }));
      ws.onmessage = (e) => {
        const m = JSON.parse(e.data as string);
        if (m.testType) { this.result = parseDeviceResult(m as DeviceResult); return; }
        if (m.frame) { this.samples++; onFrame(m as SensorFrame); }
      };
    });
  }

  stop(): CaptureResult {
    this.ws?.send(JSON.stringify({ cmd: 'stop' }));
    return {
      metrics: this.result?.metrics ?? {},
      angles: this.result?.angles ?? [],
      quality: this.quality,
      samples: this.samples,
    };
  }

  dispose(): void { this.ws?.close(); this.ws = null; }
}

let active: SensorSource | null = null;
export function getSensorSource(): SensorSource {
  if (!active) active = new MockSensorSource();
  return active;
}
export function setSensorSource(s: SensorSource): void {
  active?.dispose();
  active = s;
}
