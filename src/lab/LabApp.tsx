import React, { useCallback, useEffect, useRef, useState } from 'react';
import { NeuralNetwork } from '../ml/neuralNetwork';
import { AlphanumericStudio } from './alphanumeric/AlphanumericStudio';
import { DigitLab } from './digit/DigitLab';
import { LabHeader, LabTab } from './LabHeader';

function tabFromHash(): LabTab {
  return window.location.hash === '#alphanumeric' ? 'alphanumeric' : 'digits';
}

/** The lab for looking inside the networks: train them, see their neurons, weights and metrics. */
export default function LabApp() {
  const [tab, setTab] = useState<LabTab>(tabFromHash);
  const [, setTick] = useState(0);

  // The digit network lives here so that it survives switching tabs
  const networkRef = useRef<NeuralNetwork>(new NeuralNetwork({ hiddenSize1: 64, hiddenSize2: 32 }));
  const handleNetworkUpdate = useCallback(() => setTick((t) => t + 1), []);

  useEffect(() => {
    const onHash = () => setTab(tabFromHash());
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  const select = (next: LabTab) => {
    window.location.hash = next === 'digits' ? '' : next;
    setTab(next);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans">
      <LabHeader tab={tab} onSelect={select} />
      <main>
        {tab === 'digits' && <DigitLab network={networkRef.current} onNetworkUpdate={handleNetworkUpdate} />}
        {tab === 'alphanumeric' && <AlphanumericStudio />}
      </main>
    </div>
  );
}
