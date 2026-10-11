import React from 'react';
import { FlaskConical } from 'lucide-react';
import { MathCanvas } from './MathCanvas';

/** The handwriting app for people: one infinite page, nothing else on screen. */
export default function App() {
  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-slate-50 text-slate-900 font-sans">
      <header className="flex items-center justify-between px-5 py-2.5 border-b border-slate-200 bg-white shrink-0 select-none z-30">
        <div className="flex items-center gap-2">
          <span className="text-xl font-bold tracking-tight text-slate-900 font-serif">NeuroScribe</span>
          <span className="hidden sm:inline text-xs text-slate-400">/ handwritten math</span>
        </div>
        <a
          href={`${import.meta.env.BASE_URL}lab.html`}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-600 bg-slate-50 hover:bg-slate-100 border border-slate-200 rounded-md transition-colors"
          title="Train and inspect the neural networks behind the app"
        >
          <FlaskConical className="w-3.5 h-3.5 text-slate-500" />
          <span>Neural network lab</span>
        </a>
      </header>
      <main className="flex-1 min-h-0 relative">
        <MathCanvas />
      </main>
    </div>
  );
}
