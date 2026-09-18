/**
 * Demographic OA-risk boundary. Mirrors inference.ts's backend-swap pattern:
 * `HttpBackend` calls OAF's `/api/risk/demographic` (the first REAL trained
 * model in the clinical-risk pipeline — see risk-model/README.md at the repo
 * root for training data, scope and why it's age+BMI+sex only). With no API
 * configured, `FixtureBackend` runs so the flow stays demoable offline.
 *
 * SCOPE: this model does not see WOMAC, KL grade, or functional/sensor data.
 * Its output is one more input alongside computeClinicalScore()
 * (clinicalScore.ts) — not a replacement for it, and NOT yet wired into
 * computeOverallRisk() (overallRisk.ts), which combines clinical + functional
 * + x-ray only. Folding this in as a 4th stream is a real option later, but
 * wasn't asked for — don't add it silently.
 */
import type { RiskBand } from '../db';

export interface DemographicRiskResult {
  /** P(knee OA), 0..1, from the trained model */
  pOA: number;
  band: RiskBand;
  backend: string;
  model?: string;
  scopeWarning?: string;
}

export interface DemographicRiskBackend {
  readonly name: string;
  predict(input: { age: number; bmiValue: number; sex: 'male' | 'female' }): Promise<DemographicRiskResult>;
}

/** SEX coding the trained model was fit on (opaque source-data code, not verified as ISO). */
function sexCode(sex: 'male' | 'female'): number {
  return sex === 'female' ? 2 : 1;
}

/**
 * Live model. Reuses VITE_XRAY_API — that's the OAF backend's base URL, not
 * X-ray-specific; both /api/xray/grade and /api/risk/demographic are served
 * from the same local process on the Pi.
 */
export class HttpBackend implements DemographicRiskBackend {
  readonly name: string;
  private readonly base: string;

  constructor(baseUrl: string) {
    this.base = baseUrl.replace(/\/+$/, '');
    this.name = `api ${this.base}`;
  }

  async predict(input: { age: number; bmiValue: number; sex: 'male' | 'female' }): Promise<DemographicRiskResult> {
    const res = await fetch(`${this.base}/api/risk/demographic`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ age: input.age, bmi: input.bmiValue, sex: sexCode(input.sex) }),
    });
    if (!res.ok) {
      const detail = await res.text().catch(() => '');
      throw new Error(`risk API ${res.status} ${res.statusText}${detail ? ` — ${detail}` : ''}`);
    }
    const j = await res.json();
    return { pOA: j.p_oa, band: j.band, backend: this.name, model: j.model, scopeWarning: j.scope_warning };
  }
}

/** Deterministic offline stand-in — same monotonic shape as the trained model, not the model itself. */
export class FixtureBackend implements DemographicRiskBackend {
  readonly name = 'fixture';
  async predict(input: { age: number; bmiValue: number; sex: 'male' | 'female' }): Promise<DemographicRiskResult> {
    await new Promise((r) => setTimeout(r, 300));
    let p = 0.15;
    if (input.age >= 75) p += 0.3; else if (input.age >= 60) p += 0.2; else if (input.age >= 45) p += 0.1;
    if (input.bmiValue >= 30) p += 0.2; else if (input.bmiValue >= 25) p += 0.1;
    p = Math.min(0.95, Math.max(0.05, p));
    const band: RiskBand = p >= 0.65 ? 'high' : p >= 0.4 ? 'moderate' : 'low';
    return { pOA: p, band, backend: this.name };
  }
}

const API_URL = (import.meta.env.VITE_XRAY_API ?? '').trim();

let backend: DemographicRiskBackend = API_URL ? new HttpBackend(API_URL) : new FixtureBackend();
export function getDemographicRiskBackend(): DemographicRiskBackend { return backend; }
export function setDemographicRiskBackend(b: DemographicRiskBackend): void { backend = b; }
