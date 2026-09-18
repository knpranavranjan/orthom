import { useState } from 'react';
import Home from './pages/Home';
import Registration from './pages/Registration';
import ClinicalAssessment from './pages/ClinicalAssessment';
import MovementCapture from './pages/MovementCapture';
import SensorSummary from './pages/SensorSummary';
import XrayUpload from './pages/XrayUpload';
import XrayCompare from './pages/XrayCompare';
import Result from './pages/Result';
import Records from './pages/Records';
import PatientDetail from './pages/PatientDetail';
import { useSession } from './store/session';

/**
 * The kiosk flow is strictly linear with guards, so it is a step index
 * rather than a router — no URL semantics leaking into kiosk mode, and
 * one obvious place to see the whole page order.
 */
const FLOW = ['home', 'reg', 'clinical', 'mv0', 'mv1', 'mv2', 'summary', 'xray', 'compare', 'result'] as const;
type Step = (typeof FLOW)[number] | 'records' | 'detail';

export default function App() {
  const [step, setStep] = useState<Step>('home');
  const [detailId, setDetailId] = useState<string | null>(null);
  const reset = useSession((s) => s.reset);

  const go = (s: Step) => setStep(s);
  const idx = FLOW.indexOf(step as (typeof FLOW)[number]);
  const next = () => go(FLOW[Math.min(FLOW.length - 1, idx + 1)]);
  const back = () => go(FLOW[Math.max(0, idx - 1)]);

  function startNew() {
    reset();
    go('reg');
  }
  function finish() {
    reset();
    go('home');
  }

  switch (step) {
    case 'home':     return <Home onStart={startNew} onRecords={() => go('records')} />;
    case 'records':  return (
      <Records
        onBack={() => go('home')}
        onOpen={(id) => { setDetailId(id); go('detail'); }}
      />
    );
    case 'detail':   return detailId
      ? <PatientDetail reportId={detailId} onBack={() => go('records')} />
      : <Records onBack={() => go('home')} onOpen={(id) => { setDetailId(id); go('detail'); }} />;
    case 'reg':      return <Registration onNext={next} onBack={() => go('home')} />;
    case 'clinical': return <ClinicalAssessment onNext={next} onBack={back} />;
    case 'mv0':      return <MovementCapture index={0} onNext={next} onBack={back} />;
    case 'mv1':      return <MovementCapture index={1} onNext={next} onBack={back} />;
    case 'mv2':      return <MovementCapture index={2} onNext={next} onBack={back} />;
    case 'summary':  return <SensorSummary onNext={next} onBack={back} />;
    case 'xray':     return <XrayUpload onNext={next} onBack={back} />;
    case 'compare':  return <XrayCompare onNext={next} onBack={back} />;
    case 'result':   return <Result onDone={finish} onBack={back} />;
    default:         return <Home onStart={startNew} onRecords={() => go('records')} />;
  }
}
