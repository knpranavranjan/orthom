/**
 * Camera-based gait analysis boundary. Mirrors inference.ts / demographicRisk.ts's
 * backend-swap pattern: `HttpBackend` calls the OA_Computer_Vision project's
 * own API (`core/api.py`'s POST /api/gait/analyze — a SEPARATE service from
 * OAF, on its own port, since MediaPipe/OpenCV are heavy dependencies OAF
 * deliberately doesn't carry). With no API configured, `FixtureBackend` runs
 * so the flow stays demoable offline.
 *
 * SCOPE: this replaces the KneeFit-sensor gait placeholder that used to live
 * in movements.ts (cadence/strideLen/stepSymmetry from an IMU+pressure
 * insole) — gait is camera-based now. The capture step itself (live camera
 * -> per-frame knee angles) is NOT built yet; `analyze()` takes an already-
 * extracted angle time series. Whatever eventually captures that (in-browser
 * pose estimation, or a Pi-side camera process) plugs in here.
 */
import type { RiskBand } from '../db';

export interface GaitFrame {
  time_sec: number;
  left_knee_angle: number | null;
  right_knee_angle: number | null;
}

export interface GaitDomainInputs {
  seatedLeftFlexion?: number;
  seatedRightFlexion?: number;
  seatedLeftExtension?: number;
  seatedRightExtension?: number;
  stsCompletedReps?: number;
  stsMeanAscentSec?: number;
  stsPeakTrunkLeanDeg?: number;
  standingKneeAnkleRatio?: number;
  standingBaselineExt?: number;
}

export interface GaitBiomarker {
  Domain: string;
  Biomarker: string;
  Observed: string;
  'Clinical Norm': string;
  Finding: string;
}

export interface GaitResult {
  /** flat, numeric — matches movements.ts's gait MetricSpec keys, for the
   *  same capture/report/Functional-Score plumbing flexion & sit-to-stand use */
  metrics: Record<string, number>;
  band: RiskBand;
  overallRiskScore: number;
  riskLevel: string;
  affectedSide: string;
  domainScores: Record<string, number>;
  biomarkers: GaitBiomarker[];
  recommendations: string[];
  backend: string;
}

export interface GaitAnalysisBackend {
  readonly name: string;
  analyze(frames: GaitFrame[], fps: number, domains?: GaitDomainInputs): Promise<GaitResult>;
}

function bandFromRiskCode(code: string): RiskBand {
  if (code === 'HIGH' || code === 'MODERATE') return code === 'HIGH' ? 'high' : 'moderate';
  return 'low';
}

function toMetrics(j: any): Record<string, number> {
  return {
    overall_risk_score: j.oa_risk.overall_risk_score,
    rom_asymmetry_percent: j.rom_asymmetry_percent,
    cadence_steps_per_min: j.cadence_steps_per_min,
    left_rom_deg: j.left_rom_deg,
    right_rom_deg: j.right_rom_deg,
    timing_asymmetry_percent: j.timing_asymmetry_percent,
  };
}

export class HttpBackend implements GaitAnalysisBackend {
  readonly name: string;
  private readonly base: string;

  constructor(baseUrl: string) {
    this.base = baseUrl.replace(/\/+$/, '');
    this.name = `api ${this.base}`;
  }

  async analyze(frames: GaitFrame[], fps: number, domains: GaitDomainInputs = {}): Promise<GaitResult> {
    const res = await fetch(`${this.base}/api/gait/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        frames, fps,
        seated_left_flexion: domains.seatedLeftFlexion,
        seated_right_flexion: domains.seatedRightFlexion,
        seated_left_extension: domains.seatedLeftExtension,
        seated_right_extension: domains.seatedRightExtension,
        sts_completed_reps: domains.stsCompletedReps,
        sts_mean_ascent_sec: domains.stsMeanAscentSec,
        sts_peak_trunk_lean_deg: domains.stsPeakTrunkLeanDeg,
        standing_knee_ankle_ratio: domains.standingKneeAnkleRatio,
        standing_baseline_ext: domains.standingBaselineExt,
      }),
    });
    if (!res.ok) {
      const detail = await res.text().catch(() => '');
      throw new Error(`gait API ${res.status} ${res.statusText}${detail ? ` — ${detail}` : ''}`);
    }
    const j = await res.json();
    return {
      metrics: toMetrics(j),
      band: bandFromRiskCode(j.oa_risk.risk_code),
      overallRiskScore: j.oa_risk.overall_risk_score,
      riskLevel: j.oa_risk.risk_level,
      affectedSide: j.oa_risk.affected_side,
      domainScores: j.oa_risk.domain_scores,
      biomarkers: j.oa_risk.biomarkers,
      recommendations: j.oa_risk.recommendations,
      backend: this.name,
    };
  }
}

/** Deterministic offline stand-in — does NOT reimplement gait-cycle detection,
 *  same status as inference.ts's FixtureBackend for the X-ray model. */
export class FixtureBackend implements GaitAnalysisBackend {
  readonly name = 'fixture';
  async analyze(frames: GaitFrame[], _fps: number): Promise<GaitResult> {
    await new Promise((r) => setTimeout(r, 500));
    const seed = frames.length % 5;
    const score = 20 + seed * 12; // 20, 32, 44, 56, 68 — spans low/mild/moderate
    const band: RiskBand = score >= 65 ? 'high' : score >= 40 ? 'moderate' : 'low';
    return {
      metrics: {
        overall_risk_score: score,
        rom_asymmetry_percent: 8 + seed * 3,
        cadence_steps_per_min: 90 - seed * 4,
        left_rom_deg: 95 - seed * 2,
        right_rom_deg: 100 - seed,
        timing_asymmetry_percent: 6 + seed * 2,
      },
      band,
      overallRiskScore: score,
      riskLevel: band === 'high' ? 'MODERATE OA RISK / FUNCTIONAL COMPROMISE' : band === 'moderate' ? 'MILD RISK / EARLY PREVENTIVE WINDOW' : 'OPTIMAL / LOW RISK',
      affectedSide: 'BILATERAL SYMMETRIC',
      domainScores: { gait_kinematics: score, seated_active_rom: 0, sit_to_stand_function: 0, standing_alignment: 0 },
      biomarkers: [],
      recommendations: [],
      backend: this.name,
    };
  }
}

const API_URL = (import.meta.env.VITE_GAIT_API ?? '').trim();

let backend: GaitAnalysisBackend = API_URL ? new HttpBackend(API_URL) : new FixtureBackend();
export function getGaitBackend(): GaitAnalysisBackend { return backend; }
export function setGaitBackend(b: GaitAnalysisBackend): void { backend = b; }
