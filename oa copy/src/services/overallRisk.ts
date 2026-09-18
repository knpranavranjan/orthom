/**
 * Final combined risk — the last fusion step, folding together whichever of
 * the three evidence streams this encounter actually has:
 *   - Clinical (WOMAC + history + duration)      — clinicalScore.ts
 *   - Functional (flexion / sit-to-stand / gait)  — functionalScore.ts
 *   - Radiographic (X-ray KL grade)                — inference.ts
 *
 * Equal-weighted average of whichever streams are present — same
 * "simplest defensible choice, swap in a fitted model later without
 * changing the interface" status as computeFunctionalScore's own
 * exercise-combining step, which this builds on. There is no
 * cross-modality validation study behind this weighting (nothing in this
 * codebase claims one), so treat this exactly like the rest of the
 * placeholder scoring: a concrete, storable number, not a diagnosis.
 */
import type { RiskBand } from '../db';
import type { KLResult } from './inference';

export interface OverallRiskInput {
  clinicalScore: number | null;
  functionalPercent: number | null;
  kl: KLResult | null;
}

export interface OverallRiskResult {
  percent: number;
  band: RiskBand;
  /** each part, or null if that evidence stream wasn't captured this encounter */
  parts: { clinical: number | null; functional: number | null; xray: number | null };
}

function bandFor(percent: number): RiskBand {
  return percent >= 60 ? 'high' : percent >= 30 ? 'moderate' : 'low';
}

/** X-ray's contribution to the 0-100 scale: the calibrated OA-screen
 *  probability when the model provided one (the better signal — continuous,
 *  temperature-scaled), else the KL grade itself scaled 0-4 -> 0-100. */
function xrayPercent(kl: KLResult | null): number | null {
  if (!kl) return null;
  if (kl.oaScreen?.pOA != null) return kl.oaScreen.pOA * 100;
  return (kl.grade / 4) * 100;
}

export function computeOverallRisk(input: OverallRiskInput): OverallRiskResult | null {
  const xray = xrayPercent(input.kl);
  const parts = { clinical: input.clinicalScore, functional: input.functionalPercent, xray };
  const present = [parts.clinical, parts.functional, parts.xray].filter((v): v is number => v != null);
  if (present.length === 0) return null;
  const percent = Math.round((present.reduce((s, v) => s + v, 0) / present.length) * 10) / 10;
  return { percent, band: bandFor(percent), parts };
}
