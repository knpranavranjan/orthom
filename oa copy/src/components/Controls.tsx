import type { Sex } from '../db';

export function Segmented<T extends string>(props: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
}) {
  return (
    <div className="seg" role="group">
      {props.options.map((o) => (
        <button
          key={o.value}
          type="button"
          aria-pressed={props.value === o.value}
          onClick={() => props.onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

/** +/- buttons AND a directly-typeable number, so a keyboard/mouse user
 *  is not stuck tapping a button 40 times to go from 1 to 55. */
export function Stepper(props: {
  value: number; min: number; max: number; step?: number;
  onChange: (v: number) => void; suffix?: string;
}) {
  const step = props.step ?? 1;
  const clamp = (v: number) => Math.min(props.max, Math.max(props.min, v));
  return (
    <div className="stepper">
      <button
        type="button" aria-label="decrease"
        disabled={props.value <= props.min}
        onClick={() => props.onChange(clamp(props.value - step))}
      >−</button>
      <span className="val">
        <input
          className="val-input" type="number" inputMode="numeric"
          value={props.value} min={props.min} max={props.max} step={step}
          onChange={(e) => {
            const n = Number(e.target.value);
            if (!Number.isNaN(n)) props.onChange(clamp(n));
          }}
        />
        {props.suffix ? <span style={{ fontSize: 13 }}>{props.suffix}</span> : null}
      </span>
      <button
        type="button" aria-label="increase"
        disabled={props.value >= props.max}
        onClick={() => props.onChange(clamp(props.value + step))}
      >+</button>
    </div>
  );
}

export function Check(props: { on: boolean; label: string; onChange: (v: boolean) => void }) {
  return (
    <button
      type="button" className="check" role="checkbox" aria-checked={props.on}
      onClick={() => props.onChange(!props.on)}
      style={{ background: 'none', border: 'none', padding: 0, textAlign: 'left', width: '100%', fontFamily: 'inherit' }}
    >
      <span className={props.on ? 'box on' : 'box'}>{props.on ? '✓' : ''}</span>
      <span className="lab">{props.label}</span>
    </button>
  );
}

export const SEXES: { value: Sex; key: string }[] = [
  { value: 'M', key: 'sex.M' },
  { value: 'F', key: 'sex.F' },
  { value: 'O', key: 'sex.O' },
];
