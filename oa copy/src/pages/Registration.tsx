import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import AppShell from '../components/AppShell';
import Keyboard from '../components/Keyboard';
import { Segmented, Stepper, Check, SEXES } from '../components/Controls';
import { useSession } from '../store/session';
import { db, newPatientId, uid, type Patient, type Sex } from '../db';

export default function Registration({ onNext, onBack }: { onNext: () => void; onBack: () => void }) {
  const { t, i18n } = useTranslation();
  const draft = useSession((s) => s.draft);
  const setDraft = useSession((s) => s.setDraft);
  const beginEncounter = useSession((s) => s.beginEncounter);
  const [kbd, setKbd] = useState(false);

  const canSubmit = draft.name.trim().length > 0 && draft.consent;

  async function submit() {
    if (!canSubmit) return;
    const now = Date.now();
    const patient: Patient = {
      id: newPatientId(),
      name: draft.name.trim(),
      age: draft.age,
      sex: draft.sex,
      heightCm: draft.heightCm,
      weightKg: draft.weightKg,
      consentAt: now,
      createdAt: now,
    };
    const encounterId = uid('enc');
    await db.patients.put(patient);
    await db.encounters.put({
      id: encounterId, patientId: patient.id, startedAt: now,
      locale: i18n.language, synced: 0,
    });
    beginEncounter(patient, encounterId);
    onNext();
  }

  return (
    <AppShell
      title={t('reg.title')}
      step="2 / 10"
      footer={
        <>
          <button className="btn grow0" onClick={onBack}>{t('nav.back')}</button>
          <button className="btn pri" disabled={!canSubmit} onClick={() => void submit()}>
            {t('nav.next')}
          </button>
        </>
      }
    >
      <div className="sheetwrap">
        <div className="pad col gap4">
          {/* Labels sit inline-left rather than above the field — keeps every
              row compact and consistent even though the content area now
              scrolls and can grow. */}
          <div className="field">
            <label>{t('reg.name')}</label>
            <div className="ctl">
              <button
                className={draft.name ? 'input' : 'input placeholder'}
                onClick={() => setKbd(true)}
                style={{ cursor: 'pointer' }}
              >
                {draft.name || t('reg.namePlaceholder')}
              </button>
            </div>
          </div>

          <div className="field">
            <label>{t('reg.age')}</label>
            <div className="ctl">
              <Stepper value={draft.age} min={1} max={110} onChange={(v) => setDraft({ age: v })} />
            </div>
          </div>

          <div className="field">
            <label>{t('reg.sex')}</label>
            <div className="ctl">
              <Segmented<Sex>
                value={draft.sex}
                options={SEXES.map((s) => ({ value: s.value, label: t(s.key) }))}
                onChange={(v) => setDraft({ sex: v })}
              />
            </div>
          </div>

          {/* Height + weight share one row (two Steppers) rather than a row
              each, keeping the form compact even though it now scrolls if
              needed on a short viewport. */}
          <div className="field">
            <label>{t('reg.heightWeight')}</label>
            <div className="ctl row gap6">
              <Stepper value={draft.heightCm} min={100} max={220} suffix="cm"
                       onChange={(v) => setDraft({ heightCm: v })} />
              <Stepper value={draft.weightKg} min={20} max={180} suffix="kg"
                       onChange={(v) => setDraft({ weightKg: v })} />
            </div>
          </div>

          {/* Consent is a latch, not a field: Next stays dark until it is ticked. */}
          <Check on={draft.consent} label={t('reg.consent')} onChange={(v) => setDraft({ consent: v })} />
        </div>

        {kbd && (
          <Keyboard
            onKey={(ch) => setDraft({ name: (draft.name + ch).slice(0, 32) })}
            onBack={() => setDraft({ name: draft.name.slice(0, -1) })}
            onDone={() => setKbd(false)}
          />
        )}
      </div>
    </AppShell>
  );
}
