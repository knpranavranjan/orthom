/**
 * KneeFit Web Bluetooth transport — connection + raw byte I/O only.
 *
 * No session logic, no parsing, no React. Just: pair (from a user gesture),
 * discover the Nordic UART Service (trying both UUID variants), subscribe to
 * TX notifications, expose `write()` on RX, and surface disconnects.
 *
 * Works in Chromium on the Raspberry Pi (the target) and desktop Chrome/Edge.
 * Requires a secure context — https or http://localhost. Over the SSH reverse
 * tunnel the Pi reaches the frontend as http://localhost:5173, which counts
 * as secure, so Web Bluetooth stays available.
 */
import { NUS_VARIANTS } from './protocol';

export type BleAvailability = 'ok' | 'no-api' | 'insecure-context';

export function bleAvailability(): BleAvailability {
  if (typeof window === 'undefined') return 'no-api';
  if (!window.isSecureContext) return 'insecure-context';
  if (!('bluetooth' in navigator)) return 'no-api';
  return 'ok';
}

type VariantId = (typeof NUS_VARIANTS)[number]['id'] | 'generic';

interface Bound {
  device: BluetoothDevice;
  server: BluetoothRemoteGATTServer;
  rx: BluetoothRemoteGATTCharacteristic | null;
  tx: BluetoothRemoteGATTCharacteristic;
  variant: VariantId;
}

const log = (...a: unknown[]) => { if (import.meta.env.DEV) console.log('[KneeFit]', ...a); };

export class KneeFitBluetooth {
  private bound: Bound | null = null;
  private onData: ((chunk: DataView) => void) | null = null;
  private onDrop: (() => void) | null = null;
  private handleTx = (e: Event) => {
    const v = (e.target as BluetoothRemoteGATTCharacteristic).value;
    if (v) this.onData?.(v);
  };
  private handleDisconnect = () => {
    log('gatt disconnected');
    this.onDrop?.();
  };

  get connected(): boolean {
    return !!this.bound?.server.connected;
  }
  get deviceName(): string | null {
    return this.bound?.device.name ?? null;
  }
  get uuidVariant(): VariantId | null {
    return this.bound?.variant ?? null;
  }

  /**
   * Prompt the browser chooser and connect. MUST be called from a user
   * gesture (click). Rejects if the user dismisses the chooser.
   */
  async requestAndConnect(cb: {
    onData: (chunk: DataView) => void;
    onDisconnect: () => void;
  }): Promise<void> {
    const avail = bleAvailability();
    if (avail === 'no-api') throw new Error('Bluetooth not supported in this browser.');
    if (avail === 'insecure-context') throw new Error('Bluetooth needs https or localhost.');

    this.onData = cb.onData;
    this.onDrop = cb.onDisconnect;

    const allServices = NUS_VARIANTS.map((v) => v.service);
    log('requesting device…');
    // acceptAllDevices: the KneeFit's advertising packet is not known to carry
    // the 128-bit NUS UUID (advertising payload is only 31 bytes and many nRF
    // UART peripherals advertise the name only). A service filter therefore
    // hides the device — the earlier "Scanning… / no devices" symptom. Show
    // every nearby BLE device so the operator can pick the KneeFit by name;
    // optionalServices still lets us read either NUS variant after connecting.
    const device = await navigator.bluetooth.requestDevice({
      acceptAllDevices: true,
      optionalServices: allServices,
    });
    log('device chosen:', device.name || '(unnamed)');
    await this.attach(device);
  }

  /** (Re)connect to a device object we already have (reconnect flow). */
  async reconnect(): Promise<void> {
    const device = this.bound?.device;
    if (!device) throw new Error('No device to reconnect to.');
    await this.detachListenersOnly();
    await this.attach(device);
  }

  private async attach(device: BluetoothDevice): Promise<void> {
    device.removeEventListener('gattserverdisconnected', this.handleDisconnect);
    device.addEventListener('gattserverdisconnected', this.handleDisconnect);

    const server = await device.gatt!.connect();
    log('gatt connected, discovering NUS…');

    let bound: Bound | null = null;
    let lastErr: unknown = null;

    // 1. exact Nordic UART Service — try the observed (65a3) then legacy (b5a3) UUIDs
    for (const variant of NUS_VARIANTS) {
      try {
        const service = await server.getPrimaryService(variant.service);
        const tx = await service.getCharacteristic(variant.tx);
        let rx: BluetoothRemoteGATTCharacteristic | null = null;
        try { rx = await service.getCharacteristic(variant.rx); }
        catch { rx = null; log('RX characteristic not found on', variant.id, '(notify-only device)'); }
        bound = { device, server, rx, tx, variant: variant.id };
        log('bound NUS variant:', variant.id);
        break;
      } catch (err) {
        lastErr = err;
      }
    }

    // 2. generic fallback — the firmware may expose a UART service under a
    //    different UUID. Walk every primary service and take the first with a
    //    notify characteristic (TX); pair it with a writable one (RX) if present.
    if (!bound) {
      try {
        const services = await server.getPrimaryServices();
        log('generic scan —', services.length, 'primary service(s):', services.map((s) => s.uuid));
        for (const service of services) {
          let chars: BluetoothRemoteGATTCharacteristic[] = [];
          try { chars = await service.getCharacteristics(); } catch { continue; }
          const tx = chars.find((c) => c.properties.notify || c.properties.indicate);
          if (!tx) continue;
          const rx = chars.find(
            (c) => c.properties.write || c.properties.writeWithoutResponse,
          ) ?? null;
          bound = { device, server, rx, tx, variant: 'generic' };
          log('bound generic UART: service', service.uuid, 'tx', tx.uuid, 'rx', rx?.uuid ?? '(none)');
          break;
        }
      } catch (err) {
        lastErr = err;
      }
    }

    if (!bound) {
      server.disconnect();
      const detail = lastErr instanceof Error ? lastErr.message : String(lastErr ?? '');
      throw new Error(
        `No UART service on "${device.name || 'device'}". ${detail}`.trim(),
      );
    }

    await bound.tx.startNotifications();
    bound.tx.removeEventListener('characteristicvaluechanged', this.handleTx);
    bound.tx.addEventListener('characteristicvaluechanged', this.handleTx);
    this.bound = bound;
    log('TX notifications enabled');
  }

  /**
   * Write a command to RX. Returns true if the bytes were delivered to the
   * characteristic, false if there is no RX characteristic or the GATT write
   * failed. Never throws — a lost command must not crash the app.
   */
  async write(bytes: Uint8Array): Promise<boolean> {
    const rx = this.bound?.rx;
    if (!rx) {
      log('write skipped — no RX characteristic');
      return false;
    }
    // copy into a fresh ArrayBuffer-backed buffer for the BufferSource contract
    const payload = new Uint8Array(bytes.length);
    payload.set(bytes);
    const text = new TextDecoder().decode(bytes).trim();

    // Pick the method the characteristic actually supports. The KneeFit RX is
    // plain WRITE (with response); calling writeValueWithoutResponse on it
    // throws NotSupportedError, which is why 'C' silently never landed.
    const p = rx.properties;
    const methods: Array<'resp' | 'noresp'> =
      p.write ? ['resp', 'noresp']
      : p.writeWithoutResponse ? ['noresp', 'resp']
      : ['resp', 'noresp'];

    let lastErr: unknown = null;
    for (const m of methods) {
      try {
        if (m === 'resp') await rx.writeValue(payload.buffer);
        else await rx.writeValueWithoutResponse(payload.buffer);
        log(`RX command sent (${m}):`, JSON.stringify(text));
        return true;
      } catch (err) {
        lastErr = err;
      }
    }
    log('RX write failed:', lastErr);
    return false;
  }

  private async detachListenersOnly(): Promise<void> {
    const b = this.bound;
    if (!b) return;
    try {
      b.tx.removeEventListener('characteristicvaluechanged', this.handleTx);
      await b.tx.stopNotifications().catch(() => undefined);
    } catch { /* already gone */ }
  }

  /** Fully disconnect and forget the device. */
  disconnect(): void {
    const b = this.bound;
    this.bound = null;
    if (!b) return;
    try {
      b.device.removeEventListener('gattserverdisconnected', this.handleDisconnect);
      b.tx.removeEventListener('characteristicvaluechanged', this.handleTx);
      void b.tx.stopNotifications().catch(() => undefined);
      b.server.disconnect();
    } catch { /* already gone */ }
    log('disconnected + forgot device');
  }
}
