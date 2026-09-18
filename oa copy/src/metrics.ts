import type { MetricSpec, MovementConfig } from './movements';

/** Formats one field exactly as every surface should show it. */
export function formatMetric(spec: MetricSpec, value: number): string {
  const v = spec.abs ? Math.abs(value) : value;
  return `${v.toFixed(spec.decimals)}${spec.unit}`;
}

export function specsByTier(cfg: MovementConfig, tier: MetricSpec['tier']): MetricSpec[] {
  return cfg.metrics.filter((m) => m.tier === tier);
}

export function primarySpec(cfg: MovementConfig): MetricSpec | undefined {
  return cfg.metrics.find((m) => m.tier === 'primary');
}

/**
 * A rig sitting on a bench reports repCount 0 and then avgRepTime 0.00.
 * Shown plainly that reads as "stood up instantly", which is worse than
 * showing nothing — so the timings are withheld and the reason is stated.
 */
export function timingsUnavailable(cfg: MovementConfig, metrics: Record<string, number>): boolean {
  return !!cfg.gateOnZero && (metrics[cfg.gateOnZero] ?? 0) === 0;
}

/** Headline so low the sensor was probably not on a limb. */
export function readingImplausible(cfg: MovementConfig, metrics: Record<string, number>): boolean {
  const p = primarySpec(cfg);
  if (!p || cfg.implausibleBelow === undefined) return false;
  const v = metrics[p.key];
  return typeof v === 'number' && v < cfg.implausibleBelow;
}

/** Secondary rows worth rendering, with gated timings removed. */
export function visibleSecondary(
  cfg: MovementConfig,
  metrics: Record<string, number>,
): MetricSpec[] {
  const gated = timingsUnavailable(cfg, metrics);
  return specsByTier(cfg, 'secondary').filter((m) => {
    if (m.key === 'crepitusCount') return false;      // rendered with its angles
    if (gated && /Time|Vel|fatigue/i.test(m.key)) return false;
    return typeof metrics[m.key] === 'number';
  });
}
