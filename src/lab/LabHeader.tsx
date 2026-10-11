import React from 'react';
import { ArrowLeft } from 'lucide-react';

export type LabTab = 'digits' | 'alphanumeric';

const TABS: Array<{ id: LabTab; label: string }> = [
  { id: 'digits', label: '01. Digit Neural Lab' },
  { id: 'alphanumeric', label: '02. Alphanumeric Lab' },
];

interface LabHeaderProps {
  tab: LabTab;
  onSelect: (tab: LabTab) => void;
}

export const LabHeader: React.FC<LabHeaderProps> = ({ tab, onSelect }) => (
  <header className="sticky top-0 flex items-center justify-between gap-4 px-6 py-3 border-b border-slate-200 bg-white/95 backdrop-blur shadow-xs shrink-0 select-none z-30">
    <div className="flex items-center gap-2">
      <span className="text-xl font-bold tracking-tight text-slate-900 font-serif">NeuroScribe</span>
      <span className="hidden sm:inline text-xs text-slate-400">/ neural network lab</span>
    </div>

    <nav className="flex items-center p-1 bg-slate-100 rounded-lg gap-1" aria-label="Labs">
      {TABS.map((t) => (
        <button
          key={t.id}
          onClick={() => onSelect(t.id)}
          aria-current={tab === t.id ? 'page' : undefined}
          className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
            tab === t.id ? 'bg-white text-slate-900 shadow-xs font-semibold' : 'text-slate-600 hover:text-slate-900'
          }`}
        >
          {t.label}
        </button>
      ))}
    </nav>

    <a
      href={import.meta.env.BASE_URL}
      className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-slate-50 hover:bg-slate-100 border border-slate-200 rounded-md transition-colors whitespace-nowrap"
    >
      <ArrowLeft className="w-3.5 h-3.5 text-slate-500" />
      <span>Handwriting app</span>
    </a>
  </header>
);
