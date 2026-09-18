/**
 * Bluetooth Low Energy transport for the sensor rig.
 *
 * Uses the Nordic UART Service (NUS), the de-facto standard for streaming
 * text over BLE from an ESP32 / nRF board. Frames arrive on the TX
 * characteristic as newline-delimited JSON in the same shape sensord.py
 * sends over the WebSocket, so nothing above this file changes.
 *
 * WHERE THIS WORKS
 *   Pi kiosk (Chromium)  yes   — this is the target
 *   Desktop Chrome/Edge  yes
 *   Android WebView      NO    — Web Bluetooth is not exposed to a WebView,
 *                               so the Capacitor build needs
 *                               @capacitor-community/bluetooth-le behind
 *                               this same SensorSource interface.
 *   iOS Safari           NO
 *
 * It also requires a secure context: https, or localhost. Serving the kiosk
 * over plain http://192.168.x.x will make navigator.bluetooth undefined.
 */
import {
  parseDeviceResult,
  type CaptureResult,
  type DeviceResult,
  type SensorFrame,
  type SensorSource,
} from './sensors';

export const NUS = {
  service: '6e400001-b5a3-f393-e0a9-e50e24dcca9e',
  tx: '6e400003-b5a3-f393-e0a9-e50e24dcca9e', // device -> kiosk, notify
  rx: '6e400002-b5a3-f393-e0a9-e50e24dcca9e', // kiosk -> device, write
} as const;

/** Nordic UART Service, both UUID variants seen on this hardware:
 *  the standard `b5a3` and the KneeFit device card's `65a3`. */
const NUS_ALL = [
  NUS,
  {
    service: '6e400001-65a3-f393-e0a9-e50e24dcca9e',
    tx: '6e400003-65a3-f393-e0a9-e50e24dcca9e',
    rx: '6e400002-65a3-f393-e0a9-e50e24dcca9e',
  },
] as const;

export type BleAvailability = 'ok' | 'no-api' | 'insecure-context';

export function bleAvailability(): BleAvailability {
  if (!window.isSecureContext) return 'insecure-context';
  if (!('bluetooth' in navigator)) return 'no-api';
  return 'ok';
}

interface ControlMessage { type: string; quality?: number }

export class BleSensorSource implements SensorSource {
  readonly name = 'ble';
  readonly deviceName: string;

  private buf = '';
  private frames: SensorFrame[] = [];
  private movement = '';
  private quality = 0;
  /** The rig computes its own metrics and sends one summary per test. */
  private result: { metrics: Record<string, number>; angles: number[] } | null = null;
  private onFrame: ((f: SensorFrame) => void) | null = null;
  private onControl: ((m: ControlMessage) => void) | null = null;

  constructor(
    private device: BluetoothDevice,
    private tx: BluetoothRemoteGATTCharacteristic,
    private rx: BluetoothRemoteGATTCharacteristic | null,
  ) {
    this.deviceName = device.name || 'sensor';
    this.tx.addEventListener('characteristicvaluechanged', this.handle);
  }

  /** Prompts the chooser. Must be called from a user gesture. */
  static async connect(onDisconnect?: () => void): Promise<BleSensorSource> {
    const allServices = NUS_ALL.map((v) => v.service);
    // Show every nearby BLE device: the rig's advertising packet is not known
    // to carry the 128-bit service UUID, so a service filter can hide it.
    const device = await navigator.bluetooth.requestDevice({
      acceptAllDevices: true,
      optionalServices: allServices,
    });
    const server = await device.gatt!.connect();

    let tx: BluetoothRemoteGATTCharacteristic | null = null;
    let rx: BluetoothRemoteGATTCharacteristic | null = null;

    // 1. exact Nordic UART Service — b5a3, then the KneeFit card's 65a3
    for (const v of NUS_ALL) {
      try {
        const service = await server.getPrimaryService(v.service);
        tx = await service.getCharacteristic(v.tx);
        try { rx = await service.getCharacteristic(v.rx); } catch { rx = null; }
        break;
      } catch { /* try next variant */ }
    }
    // 2. generic fallback — first service with a notify characteristic
    if (!tx) {
      try {
        for (const service of await server.getPrimaryServices()) {
          let chars: BluetoothRemoteGATTCharacteristic[] = [];
          try { chars = await service.getCharacteristics(); } catch { continue; }
          const notify = chars.find((c) => c.properties.notify || c.properties.indicate);
          if (!notify) continue;
          tx = notify;
          rx = chars.find((c) => c.properties.write || c.properties.writeWithoutResponse) ?? null;
          break;
        }
      } catch { /* fall through to the error below */ }
    }
    if (!tx) {
      server.disconnect();
      throw new Error(`No UART service on "${device.name || 'device'}"`);
    }

    await tx.startNotifications();
    if (onDisconnect) device.addEventListener('gattserverdisconnected', onDisconnect);
    return new BleSensorSource(device, tx, rx);
  }

  private handle = (e: Event) => {
    const v = (e.target as BluetoothRemoteGATTCharacteristic).value;
    if (!v) return;
    // BLE hands you ~20-244 byte chunks, so a JSON frame arrives in pieces.
    // Buffer until a newline before attempting to parse.
    this.buf += new TextDecoder().decode(v);
    let nl: number;
    while ((nl = this.buf.indexOf('\n')) >= 0) {
      const line = this.buf.slice(0, nl).trim();
      this.buf = this.buf.slice(nl + 1);
      if (!line) continue;
      try {
        const msg = JSON.parse(line) as SensorFrame & ControlMessage & DeviceResult;
        if (msg.testType) {
          this.result = parseDeviceResult(msg);
        } else if (msg.frame) {
          this.frames.push(msg);
          this.onFrame?.(msg);
        } else if (msg.type) {
          this.onControl?.(msg);
        }
      } catch {
        // A malformed line is not worth aborting a capture over.
      }
    }
    // Never let a device that forgets newlines grow the buffer without bound.
    if (this.buf.length > 8192) this.buf = '';
  };

  private send(obj: unknown): void {
    const rx = this.rx;
    if (!rx) return;
    const buf = new TextEncoder().encode(JSON.stringify(obj) + '\n');
    const data = new Uint8Array(buf.length);
    data.set(buf);
    // Use the method the characteristic supports. A WRITE-only characteristic
    // rejects writeValueWithoutResponse with NotSupportedError.
    const p = rx.properties;
    const primary = p.write
      ? rx.writeValue(data.buffer)
      : rx.writeValueWithoutResponse(data.buffer);
    void primary.catch(() =>
      (p.write ? rx.writeValueWithoutResponse(data.buffer) : rx.writeValue(data.buffer))
        .catch(() => undefined),
    );
  }

  async calibrate(movementId: string, ms: number): Promise<number> {
    this.movement = movementId;
    this.frames = [];
    let reported: number | null = null;
    this.onControl = (m) => {
      if (m.type === 'calibrated' && typeof m.quality === 'number') reported = m.quality;
    };
    this.send({ cmd: 'calibrate', movement: movementId, ms });

    const before = this.frames.length;
    await new Promise((r) => setTimeout(r, ms));
    this.onControl = null;

    if (reported !== null) { this.quality = reported; return this.quality; }

    // No firmware-reported figure: fall back to delivery rate over the still
    // window, which is an honest proxy for link quality. A baseline the
    // health worker cannot see is worse than no baseline at all.
    const got = this.frames.length - before;
    const expected = Math.max(1, Math.round((ms / 1000) * 10));
    this.quality = Math.max(0, Math.min(100, Math.round((got / expected) * 100)));
    this.frames = [];
    return this.quality;
  }

  start(movementId: string, onFrame: (f: SensorFrame) => void): void {
    this.movement = movementId;
    this.frames = [];
    this.result = null;
    this.buf = '';
    this.onFrame = onFrame;
    this.send({ cmd: 'start', movement: movementId });
  }

  stop(): CaptureResult {
    this.send({ cmd: 'stop' });
    this.onFrame = null;
    // The result may already have arrived (rigs that send it as the test
    // ends) or may still be in flight; either way the metrics on screen are
    // the rig's own numbers, never anything derived here.
    const res: CaptureResult = {
      metrics: this.result?.metrics ?? {},
      angles: this.result?.angles ?? [],
      quality: this.quality,
      samples: this.frames.length,
    };
    this.frames = [];
    return res;
  }

  dispose(): void {
    try {
      this.tx.removeEventListener('characteristicvaluechanged', this.handle);
      void this.tx.stopNotifications().catch(() => undefined);
      this.device.gatt?.disconnect();
    } catch {
      // already gone
    }
  }
}
