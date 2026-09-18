import { create } from 'zustand';
import type { ClinicalAssessment, Patient, Sex } from '../db';
import type { KLResult } from '../services/inference';

export interface CaptureRec {
  metrics: Record<string, number>;
  angles: number[];
  quality: number;
  samples: number;
}

interface SessionState {
  encounterId: string | null;
  patient: Patient | null;
  draft: { name: string; age: number; sex: Sex; heightCm: number; weightKg: number; consent: boolean };

  clinical: ClinicalAssessment | null;

  captures: Record<string, CaptureRec>;

  xrayBlob: Blob | null;
  xrayUrl: string | null;
  xrayAt: number | null;
  kl: KLResult | null;

  setDraft: (p: Partial<SessionState['draft']>) => void;
  beginEncounter: (patient: Patient, encounterId: string) => void;
  setClinical: (c: ClinicalAssessment) => void;
  setCapture: (movementId: string, rec: CaptureRec) => void;
  clearCapture: (movementId: string) => void;
  setXray: (blob: Blob | null, url: string | null) => void;
  setKL: (kl: KLResult | null) => void;
  reset: () => void;
}

const emptyDraft = { name: '', age: 55, sex: 'F' as Sex, heightCm: 165, weightKg: 65, consent: false };

export const useSession = create<SessionState>((set, get) => ({
  encounterId: null,
  patient: null,
  draft: { ...emptyDraft },
  clinical: null,
  captures: {},
  xrayBlob: null,
  xrayUrl: null,
  xrayAt: null,
  kl: null,

  setDraft: (p) => set({ draft: { ...get().draft, ...p } }),
  beginEncounter: (patient, encounterId) => set({ patient, encounterId }),
  setClinical: (clinical) => set({ clinical }),
  setCapture: (movementId, rec) => set({ captures: { ...get().captures, [movementId]: rec } }),
  clearCapture: (movementId) => {
    const c = { ...get().captures };
    delete c[movementId];
    set({ captures: c });
  },
  setXray: (blob, url) => {
    const prev = get().xrayUrl;
    if (prev && prev !== url) URL.revokeObjectURL(prev);
    set({ xrayBlob: blob, xrayUrl: url, xrayAt: blob ? Date.now() : null, kl: null });
  },
  setKL: (kl) => set({ kl }),
  reset: () => {
    const prev = get().xrayUrl;
    if (prev) URL.revokeObjectURL(prev);
    set({
      encounterId: null, patient: null, draft: { ...emptyDraft }, clinical: null,
      captures: {}, xrayBlob: null, xrayUrl: null, xrayAt: null, kl: null,
    });
  },
}));
