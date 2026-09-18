import { useTranslation } from 'react-i18next';

const ROWS = ['QWERTYUIOP', 'ASDFGHJKL', 'ZXCVBNM'];

/**
 * In-app keyboard. A resistive kiosk panel has no OS keyboard, and this
 * also works unchanged inside the Capacitor build. Latin only for now —
 * patient names are transliterated, which is what field registers do.
 */
export default function Keyboard(props: {
  onKey: (ch: string) => void;
  onBack: () => void;
  onDone: () => void;
}) {
  const { t } = useTranslation();
  return (
    <div className="kbd">
      {ROWS.map((row) => (
        <div className="krow" key={row}>
          {row.split('').map((ch) => (
            <button key={ch} type="button" onClick={() => props.onKey(ch)}>{ch}</button>
          ))}
        </div>
      ))}
      <div className="krow">
        <button type="button" className="wide" onClick={() => props.onKey(' ')}>{t('kb.space')}</button>
        <button type="button" onClick={props.onBack}>{t('kb.back')}</button>
        <button type="button" className="act" onClick={props.onDone}>{t('kb.done')}</button>
      </div>
    </div>
  );
}
