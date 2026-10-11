import React, { useEffect, useRef, useState } from 'react';
import katex from 'katex';
import { AlertTriangle, BrainCircuit, Check, ChevronDown, ChevronRight, Copy, Loader2, PlusCircle, RotateCcw, X } from 'lucide-react';
import { LessonSummary } from './lesson';
import { LineResult } from './recognition';
import { SymbolTile } from './SymbolTile';

export const Latex: React.FC<{ source: string; display?: boolean; className?: string }> = ({ source, display = true, className }) => {
  const ref = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (!ref.current) return;
    try {
      katex.render(source, ref.current, { displayMode: display, throwOnError: false });
    } catch {
      ref.current.textContent = source;
    }
  }, [source, display]);
  return <div ref={ref} className={className} />;
};

interface RecognitionPanelProps {
  lines: LineResult[];
  /** The drawing changed after this was read */
  stale: boolean;
  activeSymbol: { line: number; symbol: number } | null;
  onSelectSymbol: (line: number, symbol: number) => void;
  onInsertCard: (latex: string) => void;
  onClose: () => void;
  onRecognizeAgain: () => void;
  learning: LessonSummary;
}

export const RecognitionPanel: React.FC<RecognitionPanelProps> = ({
  lines,
  stale,
  activeSymbol,
  onSelectSymbol,
  onInsertCard,
  onClose,
  onRecognizeAgain,
  learning,
}) => {
  const [copied, setCopied] = useState<string | null>(null);
  const [open, setOpen] = useState<Record<string, boolean>>({});

  const copy = (key: string, text: string) => {
    navigator.clipboard?.writeText(text);
    setCopied(key);
    setTimeout(() => setCopied(null), 1500);
  };

  return (
    <aside className="absolute right-0 top-0 bottom-0 z-30 w-[372px] max-w-full bg-white/98 backdrop-blur border-l border-slate-200 shadow-xl flex flex-col" aria-label="Recognition result">
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-200">
        <div>
          <h2 className="text-sm font-semibold text-slate-900">What the network read</h2>
          <p className="text-[11px] text-slate-500">Click any symbol to correct it. The network learns from every correction.</p>
        </div>
        <button onClick={onClose} className="p-1.5 text-slate-400 hover:text-slate-700 rounded-md hover:bg-slate-100" aria-label="Close panel (Esc)" title="Close (Esc)">
          <X className="w-4 h-4" />
        </button>
      </div>

      {stale && (
        <div className="mx-4 mt-3 px-3 py-2 text-xs rounded-lg bg-amber-50 border border-amber-200 text-amber-900 flex items-center justify-between gap-2">
          <span>The drawing changed since this was read.</span>
          <button onClick={onRecognizeAgain} className="font-semibold underline underline-offset-2 whitespace-nowrap">
            Read again
          </button>
        </div>
      )}

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {lines.length === 0 && <div className="text-xs text-slate-400 text-center py-10">Nothing was recognized.</div>}
        {lines.map((line, li) => {
          const { solved } = line;
          const failed = solved.kind === 'error';
          const showLatex = failed ? line.readLatex : solved.latex;
          const stepsOpen = open[line.key] ?? false;
          return (
            <section key={line.key} className="rounded-xl border border-slate-200 overflow-hidden">
              <div className="px-3 py-1.5 bg-slate-50 border-b border-slate-200 text-[10px] font-semibold uppercase tracking-wider text-slate-500 flex justify-between">
                <span>{lines.length > 1 ? `Line ${li + 1}` : 'Line'}</span>
                <span className="font-mono normal-case tracking-normal text-slate-400">{solved.kind}</span>
              </div>

              <div className="p-3 space-y-3">
                <Latex source={showLatex || '\;'} className="overflow-x-auto text-slate-900 text-lg min-h-10 flex items-center justify-center" />

                {solved.solution && (
                  <div className="px-3 py-2 rounded-lg bg-emerald-50 border border-emerald-200 text-sm font-semibold text-emerald-950">{solved.solution}</div>
                )}
                {solved.kind === 'function' && <div className="text-xs text-slate-600">A graph of this function was added to the canvas.</div>}
                {solved.kind === 'symbolic' && (
                  <div className="px-3 py-2 rounded-lg bg-sky-50 border border-sky-200 text-xs text-sky-900">{solved.steps[0]}</div>
                )}
                {failed && (
                  <div className="px-3 py-2 rounded-lg bg-amber-50 border border-amber-200 text-xs text-amber-900 flex gap-2">
                    <AlertTriangle className="w-4 h-4 shrink-0 text-amber-600" />
                    <div>
                      {solved.error}
                      <div className="text-amber-700 mt-0.5">A wrongly read symbol usually explains it: click it below and pick the right one.</div>
                    </div>
                  </div>
                )}

                {solved.steps.length > 1 && solved.kind !== 'symbolic' && (
                  <div>
                    <button onClick={() => setOpen((o) => ({ ...o, [line.key]: !stepsOpen }))} className="flex items-center gap-1 text-[11px] font-semibold uppercase tracking-wider text-slate-500 hover:text-slate-800">
                      {stepsOpen ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />} Step by step
                    </button>
                    {stepsOpen && (
                      <ol className="mt-2 space-y-1.5 p-2.5 bg-slate-50 rounded-lg border border-slate-200">
                        {solved.steps.map((step, i) => (
                          <li key={i} className="flex items-start gap-2 text-xs text-slate-700">
                            <span className="w-4 h-4 rounded-full bg-slate-200 text-[10px] font-mono flex items-center justify-center shrink-0 mt-0.5">{i + 1}</span>
                            <span className="leading-relaxed break-words min-w-0">{step}</span>
                          </li>
                        ))}
                      </ol>
                    )}
                  </div>
                )}

                <div>
                  <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-400 mb-1.5">Symbols, left to right</div>
                  <div className="flex flex-wrap gap-1.5">
                    {line.symbols.map((sym, si) => (
                      <SymbolTile key={sym.id} symbol={sym} active={activeSymbol?.line === li && activeSymbol.symbol === si} onClick={() => onSelectSymbol(li, si)} />
                    ))}
                  </div>
                </div>

                <div className="flex items-center gap-2 pt-1">
                  <button
                    onClick={() => copy(line.key, showLatex)}
                    className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium text-slate-700 bg-white border border-slate-200 rounded-md hover:bg-slate-50"
                  >
                    {copied === line.key ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5 text-slate-500" />}
                    {copied === line.key ? 'Copied' : 'Copy LaTeX'}
                  </button>
                  {showLatex && !failed && (
                    <button
                      onClick={() => onInsertCard(showLatex)}
                      className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium text-slate-700 bg-white border border-slate-200 rounded-md hover:bg-slate-50"
                    >
                      <PlusCircle className="w-3.5 h-3.5 text-slate-500" /> Typeset on page
                    </button>
                  )}
                </div>
              </div>
            </section>
          );
        })}
      </div>

      <div className="px-4 py-3 border-t border-slate-200 bg-slate-50 text-xs text-slate-600 space-y-1.5">
        <div className="flex items-center gap-2">
          {learning.learning ? <Loader2 className="w-3.5 h-3.5 animate-spin text-indigo-500" /> : <BrainCircuit className="w-3.5 h-3.5 text-indigo-500" />}
          <span className="font-medium text-slate-800">{learning.learning ? 'Learning from your correction…' : learning.headline}</span>
        </div>
        {learning.detail && <div className="text-[11px] text-slate-500 pl-5">{learning.detail}</div>}
        {learning.count > 0 && (
          <button onClick={learning.onForget} className="ml-5 flex items-center gap-1 text-[11px] text-slate-500 hover:text-rose-600">
            <RotateCcw className="w-3 h-3" /> Forget what it learned from me
          </button>
        )}
      </div>
    </aside>
  );
};
