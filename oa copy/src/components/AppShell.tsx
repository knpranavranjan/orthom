import type { ReactNode } from 'react';
import { useTranslation } from 'react-i18next';
import { LANGS, applyLang } from '../i18n';

interface Props {
  title: string;
  step?: string;
  children: ReactNode;
  /** Action bar contents. Movement screens pass their own row. */
  footer?: ReactNode;
}

export default function AppShell({ title, step, children, footer }: Props) {
  const { i18n } = useTranslation();
  const next = LANGS[(LANGS.findIndex((l) => l.code === i18n.language) + 1) % LANGS.length];

  return (
    <div className="app">
      <div className="head">
        <span className="title">{title}</span>
        <span className="row gap6" style={{ flex: '0 0 auto' }}>
          {step && <span className="step">{step}</span>}
          <button
            className="langbtn"
            onClick={() => applyLang(next.code)}
            aria-label={`Switch to ${next.native}`}
          >
            {next.label}
          </button>
        </span>
      </div>

      <div className="content">{children}</div>

      {footer && <div className="actions">{footer}</div>}
    </div>
  );
}
