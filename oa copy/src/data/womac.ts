/**
 * WOMAC-structured questionnaire — item topics only.
 *
 * WOMAC (Western Ontario and McMaster Universities Osteoarthritis Index) is a
 * licensed clinical instrument. The wording below follows its published
 * three-subscale topic structure (pain / stiffness / physical function) for
 * prototype purposes — it is NOT the licensor's certified item text. Do not
 * present this as validated WOMAC output; see `ClinicalAssessment.instrumentVersion`
 * in db.ts. Before any real deployment, replace `text` with the licensed
 * English (and validated-translation) item wording.
 */
export type WomacSubscale = 'pain' | 'stiffness' | 'function';

export interface WomacItem {
  id: string;
  subscale: WomacSubscale;
  text: string;
}

export const SEVERITY_LABELS = ['None', 'Mild', 'Moderate', 'Severe', 'Extreme'] as const;

export const WOMAC_ITEMS: WomacItem[] = [
  // Pain (5) — during the listed activity
  { id: 'p1', subscale: 'pain', text: 'Pain walking on a flat surface' },
  { id: 'p2', subscale: 'pain', text: 'Pain going up or down stairs' },
  { id: 'p3', subscale: 'pain', text: 'Pain at night while in bed' },
  { id: 'p4', subscale: 'pain', text: 'Pain sitting or lying' },
  { id: 'p5', subscale: 'pain', text: 'Pain standing upright' },

  // Stiffness (2)
  { id: 's1', subscale: 'stiffness', text: 'Stiffness after first waking in the morning' },
  { id: 's2', subscale: 'stiffness', text: 'Stiffness later in the day, after sitting, lying or resting' },

  // Physical function (17)
  { id: 'f1', subscale: 'function', text: 'Difficulty descending stairs' },
  { id: 'f2', subscale: 'function', text: 'Difficulty ascending stairs' },
  { id: 'f3', subscale: 'function', text: 'Difficulty rising from sitting' },
  { id: 'f4', subscale: 'function', text: 'Difficulty standing' },
  { id: 'f5', subscale: 'function', text: 'Difficulty bending to the floor' },
  { id: 'f6', subscale: 'function', text: 'Difficulty walking on a flat surface' },
  { id: 'f7', subscale: 'function', text: 'Difficulty getting in or out of a car' },
  { id: 'f8', subscale: 'function', text: 'Difficulty going shopping' },
  { id: 'f9', subscale: 'function', text: 'Difficulty putting on socks or stockings' },
  { id: 'f10', subscale: 'function', text: 'Difficulty rising from bed' },
  { id: 'f11', subscale: 'function', text: 'Difficulty taking off socks or stockings' },
  { id: 'f12', subscale: 'function', text: 'Difficulty lying in bed' },
  { id: 'f13', subscale: 'function', text: 'Difficulty getting in or out of the bath' },
  { id: 'f14', subscale: 'function', text: 'Difficulty sitting' },
  { id: 'f15', subscale: 'function', text: 'Difficulty getting on or off the toilet' },
  { id: 'f16', subscale: 'function', text: 'Difficulty with heavy household duties' },
  { id: 'f17', subscale: 'function', text: 'Difficulty with light household duties' },
];

export const WOMAC_PAIN_COUNT = WOMAC_ITEMS.filter((i) => i.subscale === 'pain').length;
export const WOMAC_STIFFNESS_COUNT = WOMAC_ITEMS.filter((i) => i.subscale === 'stiffness').length;
export const WOMAC_FUNCTION_COUNT = WOMAC_ITEMS.filter((i) => i.subscale === 'function').length;
