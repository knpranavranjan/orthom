/**
 * KneeFit functional-assessment thresholds — ROC-derived cutoffs from the
 * device validation study (`flexion_roc_thresholds.csv`,
 * `sit_to_stand_roc_thresholds.csv`): a Youden-optimal threshold per
 * parameter for Healthy-vs-OA and Early-vs-Advanced, with the AUC at the
 * Healthy-vs-OA cutoff.
 *
 * `key` is the KneeFit firmware's own field name (see
 * `services/kneefit/types.ts` / `movements.ts`) — NOT the CSV's parameter
 * name, which differs for several fields (e.g. CSV `hesitationCount` is the
 * device's `hesitation`; CSV `avgRiseVelocity` is the device's `avgRiseVel`).
 * Renaming here is how a firmware/CSV mismatch becomes a silent scoring bug.
 *
 * `direction` says which side of the threshold is healthier — inferred from
 * clinical reasoning AND confirmed by the CSV's own threshold ordering
 * (a "lowerIsBetter" parameter's Early-vs-Advanced threshold is always
 * higher than its Healthy-vs-OA threshold, and vice versa, in the source
 * data).
 */

export type Direction = 'higherIsBetter' | 'lowerIsBetter';

export interface RocThresholdParam {
  /** default when omitted — the existing ROC-cutoff parameters below never set this */
  kind?: 'roc';
  key: string;
  direction: Direction;
  /** ROC-optimal cutoff separating Healthy from OA */
  healthyVsOA: number;
  /** ROC-optimal cutoff separating Early from Advanced OA */
  earlyVsAdvanced: number;
  /** AUC at the Healthy-vs-OA cutoff — used as this parameter's weight */
  aucHealthyVsOA: number;
  /** For a field the device doesn't send directly (flexROM) — derived from
   *  fields that ARE sent. Returns undefined if the inputs are missing. */
  derive?: (metrics: Record<string, number>) => number | undefined;
}

/**
 * A sensor with no paired Healthy-vs-OA / Early-vs-Advanced ROC study behind
 * it — scored by a fixed clinical band table instead of a fitted cutoff.
 * `temperatureDifferenceC` (peri-articular thermal asymmetry, an
 * inflammation proxy) is the only one of these: value -> band 0-4 by
 * `bounds`, band/bounds.length -> the same 0-100% scale the ROC parameters
 * use. `weight` is a MANUAL figure, not an empirical AUC — fixed at 0.75,
 * deliberately below the 0.86-0.98 AUC range the fitted parameters actually
 * measured, so an unvalidated rule can't outweigh evidence that was.
 */
export interface CategoricalThresholdParam {
  kind: 'categorical';
  key: string;
  /** ascending boundaries; value < bounds[0] -> band 0, ..., value >= bounds[last] -> band bounds.length */
  bounds: number[];
  weight: number;
  derive?: (metrics: Record<string, number>) => number | undefined;
}

export type ThresholdParam = RocThresholdParam | CategoricalThresholdParam;

/** temperatureDifferenceC bands: 0 <2, 1 [2,2.5), 2 [2.5,3), 3 [3,3.5), 4 >=3.5 */
export const THERMAL_THRESHOLD: CategoricalThresholdParam = {
  kind: 'categorical',
  key: 'temperatureDifferenceC',
  bounds: [2, 2.5, 3, 3.5],
  weight: 0.75,
};

export const FLEXION_THRESHOLDS: ThresholdParam[] = [
  { key: 'maxROM', direction: 'higherIsBetter', healthyVsOA: 118.6407, earlyVsAdvanced: 100.7504, aucHealthyVsOA: 0.92889 },
  { key: 'minAngle', direction: 'lowerIsBetter', healthyVsOA: 4.5634, earlyVsAdvanced: 8.7982, aucHealthyVsOA: 0.92403 },
  { key: 'maxAngle', direction: 'higherIsBetter', healthyVsOA: 123.2604, earlyVsAdvanced: 114.7932, aucHealthyVsOA: 0.88139 },
  // Not present in every observed device payload yet (firmware-dependent) —
  // scoring tolerates its absence, see functionalScore.ts.
  { key: 'avgFlexVel', direction: 'higherIsBetter', healthyVsOA: 35.1396, earlyVsAdvanced: 26.0734, aucHealthyVsOA: 0.93125 },
  { key: 'avgExtVel', direction: 'higherIsBetter', healthyVsOA: 33.4007, earlyVsAdvanced: 22.6942, aucHealthyVsOA: 0.9075 },
  { key: 'hesitation', direction: 'lowerIsBetter', healthyVsOA: 2, earlyVsAdvanced: 4, aucHealthyVsOA: 0.86486 },
  { key: 'crepitusCount', direction: 'lowerIsBetter', healthyVsOA: 2, earlyVsAdvanced: 4, aucHealthyVsOA: 0.89146 },
  {
    key: 'flexROM', direction: 'higherIsBetter', healthyVsOA: 115.697, earlyVsAdvanced: 100.746, aucHealthyVsOA: 0.91653,
    derive: (m) => (typeof m.maxAngle === 'number' && typeof m.minAngle === 'number') ? m.maxAngle - m.minAngle : undefined,
  },
  THERMAL_THRESHOLD,
];

export const SIT_TO_STAND_THRESHOLDS: ThresholdParam[] = [
  { key: 'repCount', direction: 'higherIsBetter', healthyVsOA: 10, earlyVsAdvanced: 7, aucHealthyVsOA: 0.93153 },
  { key: 'avgRepTime', direction: 'lowerIsBetter', healthyVsOA: 2.7852, earlyVsAdvanced: 4.0555, aucHealthyVsOA: 0.92778 },
  { key: 'firstRepTime', direction: 'lowerIsBetter', healthyVsOA: 2.352, earlyVsAdvanced: 3.7769, aucHealthyVsOA: 0.9425 },
  { key: 'lastRepTime', direction: 'lowerIsBetter', healthyVsOA: 2.9181, earlyVsAdvanced: 5.5097, aucHealthyVsOA: 0.97417 },
  { key: 'fatigueDelta', direction: 'lowerIsBetter', healthyVsOA: 0.79592, earlyVsAdvanced: 1.8396, aucHealthyVsOA: 0.98264 },
  { key: 'avgRiseVel', direction: 'higherIsBetter', healthyVsOA: 71.0966, earlyVsAdvanced: 50.964, aucHealthyVsOA: 0.96069 },
  { key: 'avgDescentVel', direction: 'higherIsBetter', healthyVsOA: 61.1329, earlyVsAdvanced: 50.989, aucHealthyVsOA: 0.94736 },
  { key: 'hesitation', direction: 'lowerIsBetter', healthyVsOA: 2, earlyVsAdvanced: 4, aucHealthyVsOA: 0.8591 },
  { key: 'crepitusCount', direction: 'lowerIsBetter', healthyVsOA: 2, earlyVsAdvanced: 4, aucHealthyVsOA: 0.885 },
  THERMAL_THRESHOLD,
];
