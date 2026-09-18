/**
 * Clinical layer scoring — WOMAC subscales + BMI + age + history + duration
 * combined into one interim "clinical score".
 *
 * PLACEHOLDER FUSION: additive point thresholds, not a validated instrument
 * or fitted model. It exists so the clinical evidence stream has a concrete,
 * storable number that feeds into computeOverallRisk() (overallRisk.ts)
 * alongside the functional and radiographic streams. Do not read
 * `clinicalBand` on its own as a clinical determination.
 */
import type { DurationBand, RiskBand } from '../db';
import { WOMAC_ITEMS, type WomacSubscale } from '../data/womac';

export interface WomacScores {
  painScore: number;       // 0-20 (5 items x 0-4)
  stiffnessScore: number;  // 0-8  (2 items x 0-4)
  functionScore: number;   // 0-68 (17 items x 0-4)
  womacTotal: number;      // 0-96
  womacNormalized: number; // 0-100
}

const MAX_BY_SUBSCALE: Record<WomacSubscale, number> = { pain: 20, stiffness: 8, function: 68 };

export function computeWomacScores(items: number[]): WomacScores {
  let painScore = 0, stiffnessScore = 0, functionScore = 0;
  WOMAC_ITEMS.forEach((item, i) => {
    const v = items[i] ?? 0;
    if (item.subscale === 'pain') painScore += v;
    else if (item.subscale === 'stiffness') stiffnessScore += v;
    else functionScore += v;
  });
  const womacTotal = painScore + stiffnessScore + functionScore;
  const max = MAX_BY_SUBSCALE.pain + MAX_BY_SUBSCALE.stiffness + MAX_BY_SUBSCALE.function;
  return {
    painScore, stiffnessScore, functionScore, womacTotal,
    womacNormalized: Math.round((womacTotal / max) * 1000) / 10,
  };
}

export function computeBmi(heightCm: number, weightKg: number): number | null {
  if (!heightCm || !weightKg) return null;
  const m = heightCm / 100;
  return Math.round((weightKg / (m * m)) * 10) / 10;
}

export function bmiCategory(bmi: number | null): 'unknown' | 'normal' | 'overweight' | 'obese' {
  if (bmi === null) return 'unknown';
  if (bmi >= 30) return 'obese';
  if (bmi >= 25) return 'overweight';
  return 'normal';
}

const DURATION_POINTS: Record<DurationBand, number> = {
  '<3m': 0, '3-6m': 3, '6-12m': 5, '>1y': 8,
};

/**
 * age/BMI risk points + history flags + symptom duration + 60% weight on the
 * normalized WOMAC score, clamped to 0-100. Thresholds are indicative only.
 */
export function computeClinicalScore(input: {
  age: number;
  bmi: number | null;
  history: { priorInjury: boolean; priorSurgery: boolean; priorMskProblem: boolean };
  durationBand: DurationBand;
  womac: WomacScores;
}): { score: number; band: RiskBand } {
  let points = 0;

  if (input.age >= 75) points += 15;
  else if (input.age >= 60) points += 10;
  else if (input.age >= 45) points += 5;

  const cat = bmiCategory(input.bmi);
  if (cat === 'obese') points += 10;
  else if (cat === 'overweight') points += 5;

  if (input.history.priorInjury) points += 5;
  if (input.history.priorSurgery) points += 5;
  if (input.history.priorMskProblem) points += 5;

  points += DURATION_POINTS[input.durationBand];

  const score = Math.max(0, Math.min(100, Math.round(input.womac.womacNormalized * 0.6 + points)));
  const band: RiskBand = score >= 60 ? 'high' : score >= 30 ? 'moderate' : 'low';
  return { score, band };
}
