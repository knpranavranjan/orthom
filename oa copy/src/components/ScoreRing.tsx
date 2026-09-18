/** Hand-rolled SVG progress ring — no chart library, consistent with the
 *  rest of the app's hand-rolled visuals (bars, SVG movement figures). */
export default function ScoreRing({
  percent, size = 128, stroke = 12,
}: { percent: number; size?: number; stroke?: number }) {
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const clamped = Math.max(0, Math.min(100, percent));
  const offset = c * (1 - clamped / 100);
  const color = clamped >= 60 ? 'var(--signal)' : clamped >= 30 ? 'var(--warn)' : 'var(--ok)';
  const cx = size / 2, cy = size / 2;
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="score-ring" role="img"
         aria-label={`${Math.round(clamped)} percent`}>
      <circle cx={cx} cy={cy} r={r} fill="none" stroke="var(--surface-3)" strokeWidth={stroke} />
      <circle
        cx={cx} cy={cy} r={r} fill="none" stroke={color} strokeWidth={stroke}
        strokeDasharray={c} strokeDashoffset={offset} strokeLinecap="round"
        transform={`rotate(-90 ${cx} ${cy})`}
      />
      <text x="50%" y="50%" textAnchor="middle" dominantBaseline="central" className="score-ring-num">
        {Math.round(clamped)}%
      </text>
    </svg>
  );
}
