import React from 'react';
import { AppMode } from '../types';
import { BookOpen, Sparkles, RefreshCw } from 'lucide-react';

interface HeaderProps {
  currentMode: AppMode;
  onSelectMode: (mode: AppMode) => void;
  onOpenSymbolLibrary: () => void;
  onResetModel?: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  currentMode,
  onSelectMode,
  onOpenSymbolLibrary,
  onResetModel,
}) => {
  return (
    <header className="flex items-center justify-between px-6 py-3.5 border-b border-slate-200 bg-white shadow-xs shrink-0 select-none z-30">
      {/* Zone 1: Single text element wordmark */}
      <div className="flex items-center gap-2">
        <span className="text-xl font-bold tracking-tight text-slate-900 font-serif">
          NeuroScribe
        </span>
        <span className="hidden sm:inline text-xs text-slate-400 font-normal">
          / Math & Recognition Studio
        </span>
      </div>

      {/* Zone 2: Navigation Links */}
      <nav className="flex items-center p-1 bg-slate-100 rounded-lg gap-1">
        <button
          onClick={() => onSelectMode('digit-lab')}
          className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
            currentMode === 'digit-lab'
              ? 'bg-white text-slate-900 shadow-xs font-semibold'
              : 'text-slate-600 hover:text-slate-900'
          }`}
        >
          01. Digit Neural Lab
        </button>
        <button
          onClick={() => onSelectMode('alphanumeric')}
          className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
            currentMode === 'alphanumeric'
              ? 'bg-white text-slate-900 shadow-xs font-semibold'
              : 'text-slate-600 hover:text-slate-900'
          }`}
        >
          02. Alphanumeric Studio
        </button>
        <button
          onClick={() => onSelectMode('infinite-canvas')}
          className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
            currentMode === 'infinite-canvas'
              ? 'bg-white text-slate-900 shadow-xs font-semibold'
              : 'text-slate-600 hover:text-slate-900'
          }`}
        >
          03. Infinite OneNote Math
        </button>
      </nav>

      {/* Zone 3: 1-2 primary actions */}
      <div className="flex items-center gap-2">
        <button
          onClick={onOpenSymbolLibrary}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-slate-50 hover:bg-slate-100 border border-slate-200 rounded-md transition-colors whitespace-nowrap"
          title="LaTeX symbol reference (Greek letters, calculus, logic)"
        >
          <BookOpen className="w-3.5 h-3.5 text-slate-500" />
          <span>Symbol Library</span>
        </button>
      </div>
    </header>
  );
};
