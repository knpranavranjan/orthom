export type CaptureState = 'idle' | 'calibrating' | 'ready' | 'recording' | 'review';
export type ControlKey = 'calibrate' | 'start' | 'stop' | 'retry' | 'next';

export interface MediaSlot {
  kind: 'svg' | 'video' | 'image';
  /** figure id for kind 'svg'; asset path for 'image' */
  src: string;
  /**
   * Ordered <source> list for kind 'video'. H.264 first so the Pi uses its
   * hardware decoder; VP9 second because plain Chromium builds ship without
   * H.264 and would otherwise sit on the poster frame forever, silently.
   */
  sources?: { src: string; type: string }[];
  /** still shown before a video's first frame decodes */
  poster?: string;
}

const asset = (f: string) => `${import.meta.env.BASE_URL}movements/${f}`;

/**
 * One field as the sensor rig reports it. Keys are the device's own field
 * names, verbatim — renaming them here is how a firmware change becomes a
 * silent display bug.
 */
export interface MetricSpec {
  key: string;
  unit: string;
  decimals: number;
  /** primary = the headline figure; qa = engineering, not clinical */
  tier: 'primary' | 'secondary' | 'qa';
  /** show magnitude only; the label already carries the direction */
  abs?: boolean;
}

export interface MovementConfig {
  id: string;
  /** i18n key stem: mv.<id>.title / .instr / .hint */
  key: string;
  media: MediaSlot;
  sensors: string[];
  /** target repetitions; 0 means a single timed pass */
  reps: number;
  /** primary live readout shown during capture, if the rig streams frames */
  live: { metric: string; unit: string };
  metrics: MetricSpec[];
  /**
   * Fields whose value makes every timing metric meaningless when zero —
   * a bench test with the rig on a table reports repCount 0 and then
   * avgRepTime 0.00, which must not read as "stood up instantly".
   */
  gateOnZero?: string;
  /** below this, the headline reading implies the rig is not on a limb */
  implausibleBelow?: number;
}

export const MOVEMENTS: MovementConfig[] = [
  {
    id: 'flexion',
    key: 'flexion',
    // Real footage of the movement. Silent by design: movement 1 listens for
    // crepitus, and a speaker feeding the acoustic sensor corrupts the capture.
    media: {
      kind: 'video',
      src: asset('knee-flexion.mp4'),
      sources: [
        { src: asset('knee-flexion.mp4'),  type: 'video/mp4' },
        { src: asset('knee-flexion.webm'), type: 'video/webm' },
      ],
      poster: asset('knee-flexion.jpg'),
    },
    sensors: ['imu_pair', 'acoustic'],
    reps: 5,
    live: { metric: 'angle', unit: '°' },
    metrics: [
      { key: 'maxROM',        unit: '°',   decimals: 1, tier: 'primary' },
      { key: 'maxAngle',      unit: '°',   decimals: 1, tier: 'secondary' },
      { key: 'minAngle',      unit: '°',   decimals: 1, tier: 'secondary' },
      { key: 'avgFlexVel',    unit: '°/s', decimals: 1, tier: 'secondary' },
      { key: 'avgExtVel',     unit: '°/s', decimals: 1, tier: 'secondary', abs: true },
      { key: 'hesitation',    unit: '',    decimals: 0, tier: 'secondary' },
      { key: 'crepitusCount', unit: '',    decimals: 0, tier: 'secondary' },
      // Promoted from 'qa' to 'secondary': a real ROC-analogue scoring input
      // now (see data/functionalThresholds.ts THERMAL_THRESHOLD), not just
      // diagnostic noise.
      { key: 'temperatureDifferenceC', unit: '°C', decimals: 1, tier: 'secondary' },
    ],
    implausibleBelow: 20,
  },
  {
    id: 'sit_to_stand',
    key: 'sit_to_stand',
    // Real footage. Hands stay clasped at the chest for the whole clip —
    // "without hands" is the test, and a demo that shows otherwise teaches
    // the wrong thing.
    media: {
      kind: 'video',
      src: asset('sit-to-stand.mp4'),
      sources: [
        { src: asset('sit-to-stand.mp4'),  type: 'video/mp4' },
        { src: asset('sit-to-stand.webm'), type: 'video/webm' },
      ],
      poster: asset('sit-to-stand.jpg'),
    },
    sensors: ['imu_pair', 'fsr_insole'],
    reps: 5,
    live: { metric: 'elapsed', unit: 's' },
    metrics: [
      { key: 'repCount',        unit: '',    decimals: 0, tier: 'primary' },
      { key: 'avgRepTime',      unit: 's',   decimals: 2, tier: 'secondary' },
      { key: 'firstRepTime',    unit: 's',   decimals: 2, tier: 'secondary' },
      { key: 'lastRepTime',     unit: 's',   decimals: 2, tier: 'secondary' },
      { key: 'fatigueDelta',    unit: 's',   decimals: 2, tier: 'secondary' },
      { key: 'avgRiseVel',      unit: '°/s', decimals: 1, tier: 'secondary' },
      { key: 'avgDescentVel',   unit: '°/s', decimals: 1, tier: 'secondary', abs: true },
      { key: 'hesitation',      unit: '',    decimals: 0, tier: 'secondary' },
      { key: 'crepitusCount',   unit: '',    decimals: 0, tier: 'secondary' },
      { key: 'temperatureDifferenceC', unit: '°C', decimals: 1, tier: 'secondary' },
    ],
    // no gateOnZero: sit-to-stand shows every field the rig sends, verbatim
    // (including 0.00 timings when repCount is 0).
  },
  {
    id: 'gait',
    key: 'gait',
    media: { kind: 'svg', src: 'gait' },
    // Camera-based (MediaPipe pose -> OA_Computer_Vision/core/api.py's
    // /api/gait/analyze), NOT the KneeFit BLE wearable — this replaces the
    // originally-planned IMU+pressure-insole gait test. See
    // services/gaitAnalysis.ts. Capture UI (live camera / pose extraction)
    // is not built yet; the field names below are the real API response
    // shape, ready for whatever capture flow lands.
    sensors: ['camera'],
    reps: 0,
    live: { metric: 'cadence_steps_per_min', unit: '/min' },
    metrics: [
      { key: 'overall_risk_score',       unit: '',     decimals: 1, tier: 'primary' },
      { key: 'rom_asymmetry_percent',    unit: '%',    decimals: 1, tier: 'secondary' },
      { key: 'cadence_steps_per_min',    unit: '/min', decimals: 0, tier: 'secondary' },
      { key: 'left_rom_deg',             unit: '°',    decimals: 1, tier: 'secondary' },
      { key: 'right_rom_deg',            unit: '°',    decimals: 1, tier: 'secondary' },
      { key: 'timing_asymmetry_percent', unit: '%',    decimals: 1, tier: 'secondary' },
    ],
  },
];

export function movementAt(index: number): MovementConfig {
  return MOVEMENTS[Math.max(0, Math.min(MOVEMENTS.length - 1, index))];
}

/**
 * Single source of truth for button enablement.
 * Stop is live during calibration and recording, and only then —
 * it means ABORT in one and FINISH in the other, so its label changes too.
 */
export const CONTROLS: Record<CaptureState, Record<ControlKey, boolean>> = {
  idle:        { calibrate: true,  start: false, stop: false, retry: false, next: false },
  calibrating: { calibrate: false, start: false, stop: true,  retry: false, next: false },
  ready:       { calibrate: true,  start: true,  stop: false, retry: false, next: false },
  recording:   { calibrate: false, start: false, stop: true,  retry: false, next: false },
  review:      { calibrate: true,  start: false, stop: false, retry: true,  next: true  },
};

export const CALIBRATION_MS = 3000;
