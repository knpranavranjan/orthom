/**
 * useKneeFit — React binding for the KneeFit BLE workflow.
 *
 * The Bluetooth transport and the session manager are module-level singletons
 * so the connection and the calibration baseline survive component unmounts
 * (the three movement screens each mount a fresh page). The hook just mirrors
 * the session's snapshot into React state and exposes the control actions.
 *
 * This is additive: it does not touch `store/link.ts`, `services/sensors.ts`
 * or the X-ray / OAF pipeline.
 */
import { useCallback, useEffect, useState } from 'react';
import { KneeFitBluetooth, bleAvailability, type BleAvailability } from '../services/kneefit/bluetooth';
import { KneeFitSession } from '../services/kneefit/session';
import type { KneeFitSnapshot, KneeFitTestType } from '../services/kneefit/types';

const ble = new KneeFitBluetooth();
const session = new KneeFitSession((bytes) => ble.write(bytes));

let wired = false;
function wireOnce(): void {
  if (wired) return;
  wired = true;
  // nothing to wire globally yet; kept for symmetry / future taps
}

export interface UseKneeFit {
  snapshot: KneeFitSnapshot;
  availability: BleAvailability;
  /** convenience flags derived from the snapshot */
  isConnected: boolean;
  isRecording: boolean;
  /** open the browser chooser and connect (must be from a click) */
  addDevice: () => Promise<void>;
  reconnect: () => Promise<void>;
  disconnect: () => void;
  selectTest: (t: KneeFitTestType) => void;
  calibrate: () => Promise<void>;
  start: () => Promise<void>;
  finish: () => Promise<void>;
  retry: () => void;
}

export function useKneeFit(): UseKneeFit {
  wireOnce();
  const [snapshot, setSnapshot] = useState<KneeFitSnapshot>(() => session.snapshot());
  const availability = bleAvailability();

  useEffect(() => session.subscribe(setSnapshot), []);

  const addDevice = useCallback(async () => {
    session.onConnecting();
    try {
      await ble.requestAndConnect({
        onData: (chunk) => session.ingest(chunk),
        onDisconnect: () => session.onLinkLost(),
      });
      session.onConnected(ble.deviceName, ble.uuidVariant);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      // dismissing the chooser is a normal outcome, not an error
      if (/cancell?ed|user cancelled|no devices?/i.test(msg)) {
        session.onDisconnectByUser();
      } else {
        session.onConnectError(msg);
      }
    }
  }, []);

  const reconnect = useCallback(async () => {
    session.onConnecting();
    try {
      await ble.reconnect();
      session.onConnected(ble.deviceName, ble.uuidVariant);
    } catch (err) {
      session.onConnectError(err instanceof Error ? err.message : String(err));
    }
  }, []);

  const disconnect = useCallback(() => {
    ble.disconnect();
    session.onDisconnectByUser();
  }, []);

  const selectTest = useCallback((t: KneeFitTestType) => session.selectTest(t), []);
  const calibrate = useCallback(() => session.calibrate(), []);
  const start = useCallback(() => session.start(), []);
  const finish = useCallback(() => session.finish(), []);
  const retry = useCallback(() => session.retry(), []);

  return {
    snapshot,
    availability,
    isConnected: snapshot.connection.state === 'connected',
    isRecording: snapshot.session === 'RECORDING',
    addDevice,
    reconnect,
    disconnect,
    selectTest,
    calibrate,
    start,
    finish,
    retry,
  };
}

/** Escape hatch for non-hook code / tests. */
export const _kneefitInternals = { ble, session };
