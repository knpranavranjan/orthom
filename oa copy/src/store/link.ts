import { create } from 'zustand';
import { MockSensorSource, setSensorSource } from '../services/sensors';
import { BleSensorSource, bleAvailability, type BleAvailability } from '../services/ble';

export type LinkState = 'demo' | 'connecting' | 'connected' | 'error';

interface LinkStore {
  state: LinkState;
  deviceName: string | null;
  availability: BleAvailability;
  error: string | null;
  connect: () => Promise<void>;
  disconnect: () => void;
}

export const useLink = create<LinkStore>((set, get) => ({
  state: 'demo',
  deviceName: null,
  availability: bleAvailability(),
  error: null,

  connect: async () => {
    if (get().availability !== 'ok') return;
    set({ state: 'connecting', error: null });
    try {
      const src = await BleSensorSource.connect(() => {
        // Losing the rig mid-camp must not brick the kiosk: fall straight
        // back to the mock so the flow still completes.
        setSensorSource(new MockSensorSource());
        set({ state: 'demo', deviceName: null, error: 'disconnected' });
      });
      setSensorSource(src);
      set({ state: 'connected', deviceName: src.deviceName, error: null });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      // Dismissing the chooser is a normal outcome, not a failure.
      const cancelled = /cancell?ed|User cancelled/i.test(msg);
      set({ state: cancelled ? 'demo' : 'error', error: cancelled ? null : msg });
    }
  },

  disconnect: () => {
    setSensorSource(new MockSensorSource());
    set({ state: 'demo', deviceName: null, error: null });
  },
}));
