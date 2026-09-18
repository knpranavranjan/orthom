import Dexie, { type Table } from 'dexie';

export type Sex = 'M' | 'F' | 'O';
export type RiskBand = 'low' | 'moderate' | 'high';

export interface Patient {
  id: string; name: string; age: number; sex: Sex;
  /** Optional — used to derive BMI for the clinical score. 0 means not recorded. */
  heightCm: number; weightKg: number;
  consentAt: number; createdAt: number;
}

export type DurationBand = '<3m' | '3-6m' | '6-12m' | '>1y';

/**
 * One patient-reported clinical intake per encounter: history flags, symptom
 * duration, and a WOMAC-structured pain/stiffness/function questionnaire.
 * `instrumentVersion` is deliberately not "womac" — the item wording here is
 * written from the instrument's published topic structure, not the licensed
 * WOMAC text, so it must not be presented as the certified instrument. Swap
 * in a properly licensed item set before this leaves prototype status.
 */
export interface ClinicalAssessment {
  id: string; encounterId: string;
  /** `none` is an explicit "no history" answer, distinct from not-yet-answered. */
  history: { priorInjury: boolean; priorSurgery: boolean; priorMskProblem: boolean; none: boolean };
  durationBand: DurationBand;
  /** raw 0-4 per item, indexed exactly as WOMAC_ITEMS in src/data/womac.ts */
  womacItems: number[];
  painScore: number; stiffnessScore: number; functionScore: number; womacTotal: number;
  bmi: number | null;
  clinicalScore: number;
  clinicalBand: RiskBand;
  instrumentVersion: 'womac-unlicensed-prototype';
  completedAt: number;
}
export interface Encounter {
  id: string; patientId: string; startedAt: number;
  completedAt?: number; locale: string; synced: 0 | 1;
}
export interface Capture {
  id: string; encounterId: string; movementId: string;
  metrics: Record<string, number>;
  /** knee angles at which crepitus fired — the clinically useful part */
  angles?: number[];
  quality: number; createdAt: number;
}
export interface XrayRec {
  id: string; encounterId: string; blob: Blob; capturedAt: number;
}
export interface ReportRec {
  id: string; encounterId: string; patientId: string;
  riskBand: RiskBand; klGrade: number; klDist: number[];
  manualScore: number; createdAt: number;
  /** Grad-CAM overlay as a data URL. Not indexed, so no migration needed. */
  heatmap?: string | null;
  /** The 3-way combined score (clinical + functional + x-ray, computeOverallRisk()).
   *  Optional/not indexed, so no migration needed — a record saved before this
   *  field existed just reads back as undefined. */
  overallPercent?: number | null;
}

class OADb extends Dexie {
  patients!: Table<Patient, string>;
  encounters!: Table<Encounter, string>;
  captures!: Table<Capture, string>;
  xrays!: Table<XrayRec, string>;
  reports!: Table<ReportRec, string>;
  settings!: Table<{ key: string; value: string }, string>;
  clinicalAssessments!: Table<ClinicalAssessment, string>;

  constructor() {
    super('oa-screening');
    this.version(1).stores({
      patients:   'id, name, createdAt',
      encounters: 'id, patientId, startedAt, synced',
      captures:   'id, encounterId, movementId',
      xrays:      'id, encounterId',
      reports:    'id, encounterId, patientId, createdAt',
      settings:   'key',
    });
    // v2: clinical/WOMAC intake layer. heightCm/weightKg on `patients` need no
    // migration (not indexed); the new table needs a version bump.
    this.version(2).stores({
      clinicalAssessments: 'id, encounterId',
    });
  }
}

export const db = new OADb();

/** Short, human-readable, collision-resistant enough for a field kiosk. */
export function newPatientId(): string {
  return 'PT-' + Math.floor(10000 + Math.random() * 89999);
}
export function uid(prefix = 'id'): string {
  return prefix + '-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 7);
}

export async function getSetting(key: string): Promise<string | undefined> {
  return (await db.settings.get(key))?.value;
}
export async function setSetting(key: string, value: string): Promise<void> {
  await db.settings.put({ key, value });
}
