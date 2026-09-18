import { useTranslation } from 'react-i18next';

export default function KLDistribution({ dist }: { dist: number[] }) {
  const { t } = useTranslation();
  const top = dist.indexOf(Math.max(...dist));
  return (
    <div className="kl">
      {dist.map((p, i) => (
        <div className={i === top ? 'klrow top' : 'klrow'} key={i}>
          <span className="g">{t('kl.grade')} {i}</span>
          <span className="bar"><i style={{ width: `${Math.round(p * 100)}%` }} /></span>
          <span className="p">{(p * 100).toFixed(0)}%</span>
        </div>
      ))}
    </div>
  );
}
