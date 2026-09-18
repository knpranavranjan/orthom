/**
 * Functional layer scoring — converts KneeFit capture metrics into a 0-100
 * percentage per exercise, then a combined Functional Score, using the
 * ROC-derived thresholds in `data/functionalThresholds.ts`.
 *
 * Two kinds of parameter, scored differently, then combined the same way:
 *
 * - ROC-fitted (most parameters): 0% = at/beyond the Healthy-vs-OA cutoff on
 *   the healthy side. 100% = at/beyond the Early-vs-Advanced cutoff on the
 *   advanced side. Linear in between, clamped to [0,100] (a single extreme
 *   reading cannot be extrapolated past the two clinically-derived cutoffs).
 * - Categorical (temperatureDifferenceC only, see functionalThresholds.ts
 *   THERMAL_THRESHOLD): banded 0-4 by fixed cutoffs, then band/4 -> the same
 *   0-100% scale.
 *
 * Per exercise: weighted average of its available parameters — ROC
 * parameters weighted by their Healthy-vs-OA AUC (a better discriminator in
 * the validation study counts for more), the categorical parameter by its
 * fixed (non-empirical) weight. A parameter the device didn't send this
 * capture (e.g. `avgFlexVel` on older firmware) is skipped, not treated as 0
 * or 100 — the remaining parameters' weights are what the average is taken over.
 *
 * Combined Functional Score: equal-weighted average across whichever
 * exercises were captured. There is no combined-exercise AUC to weight one
 * over the other, so this is the simplest defensible choice — swap in a
 * fitted model later without changing the interface.
 *
 * Gait is different: its percent comes pre-computed from
 * services/gaitAnalysis.ts's `overall_risk_score` (the camera pipeline's own
 * literature-normed composite score, see OA_Computer_Vision/core/oa_risk_engine.py)
 * — it does NOT go through paramPercent/rocPercent here, because that engine
 * already is the scoring model for this modality. It's folded into the
 * equal-weighted exercise average exactly like flexion/sit-to-stand's percent is.
 */
import type { RiskBand } from '../db';
import { FLEXION_THRESHOLDS, SIT_TO_STAND_THRESHOLDS, type ThresholdParam, type RocThresholdParam, type CategoricalThresholdParam } from '../data/functionalThresholds';

export interface ParamScore {
  key: string;
  value: number;
  percent: number;
  weight: number;
}

export interface ExerciseScore {
  percent: number;
  params: ParamScore[];
  /** thresholds defined for this exercise but not present in this capture */
  missing: string[];
}

function rocPercent(value: number, p: RocThresholdParam): number {
  const { healthyVsOA: t1, earlyVsAdvanced: t2, direction } = p;
  const span = direction === 'higherIsBetter' ? t1 - t2 : t2 - t1;
  if (span <= 0) return 0;
  const progressed = direction === 'higherIsBetter' ? t1 - value : value - t1;
  return Math.max(0, Math.min(100, (progressed / span) * 100));
}

/** value -> band index (0..bounds.length) by ascending cutoffs -> percent on the 0-100 scale. */
function categoricalBand(value: number, bounds: number[]): number {
  return bounds.filter((b) => value >= b).length;
}
function categoricalPercent(value: number, p: CategoricalThresholdParam): number {
  return (categoricalBand(value, p.bounds) / p.bounds.length) * 100;
}

function round1(n: number): number {
  return Math.round(n * 10) / 10;
}

function scoreExercise(metrics: Record<string, number>, thresholds: ThresholdParam[]): ExerciseScore {
  const params: ParamScore[] = [];
  const missing: string[] = [];
  for (const p of thresholds) {
    const raw = p.derive ? p.derive(metrics) : metrics[p.key];
    if (typeof raw !== 'number' || !Number.isFinite(raw)) { missing.push(p.key); continue; }
    const percent = p.kind === 'categorical' ? categoricalPercent(raw, p) : rocPercent(raw, p);
    const weight = p.kind === 'categorical' ? p.weight : p.aucHealthyVsOA;
    params.push({ key: p.key, value: raw, percent, weight });
  }
  const totalWeight = params.reduce((s, p) => s + p.weight, 0);
  const percent = totalWeight > 0
    ? round1(params.reduce((s, p) => s + p.percent * p.weight, 0) / totalWeight)
    : 0;
  return { percent, params, missing };
}

export function scoreFlexion(metrics: Record<string, number>): ExerciseScore {
  return scoreExercise(metrics, FLEXION_THRESHOLDS);
}
export function scoreSitToStand(metrics: Record<string, number>): ExerciseScore {
  return scoreExercise(metrics, SIT_TO_STAND_THRESHOLDS);
}

export function bandFor(percent: number): RiskBand {
  return percent >= 60 ? 'high' : percent >= 30 ? 'moderate' : 'low';
}

export interface FunctionalResult {
  percent: number;
  band: RiskBand;
  flexion: ExerciseScore | null;
  sitToStand: ExerciseScore | null;
  gait: ExerciseScore | null;
}

/** gait's percent is already final (see file header) — wrap it in the same
 *  ExerciseScore shape so it slots into the combined average unchanged. */
function scoreGait(metrics: Record<string, number>): ExerciseScore {
  const raw = metrics.overall_risk_score;
  const percent = typeof raw === 'number' && Number.isFinite(raw)
    ? Math.max(0, Math.min(100, round1(raw)))
    : 0;
  const missing = typeof raw === 'number' && Number.isFinite(raw) ? [] : ['overall_risk_score'];
  return { percent, params: [{ key: 'overall_risk_score', value: raw ?? 0, percent, weight: 1 }], missing };
}

export function computeFunctionalScore(captures: {
  flexion?: Record<string, number>;
  sit_to_stand?: Record<string, number>;
  gait?: Record<string, number>;
}): FunctionalResult | null {
  const flexion = captures.flexion ? scoreFlexion(captures.flexion) : null;
  const sitToStand = captures.sit_to_stand ? scoreSitToStand(captures.sit_to_stand) : null;
  const gait = captures.gait ? scoreGait(captures.gait) : null;
  const parts = [flexion, sitToStand, gait].filter((x): x is ExerciseScore => x !== null);
  if (parts.length === 0) return null;
  const percent = round1(parts.reduce((s, e) => s + e.percent, 0) / parts.length);
  return { percent, band: bandFor(percent), flexion, sitToStand, gait };
}
