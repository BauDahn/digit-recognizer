import React, { useMemo, useState } from 'react';
import katex from 'katex';
import { Search, X } from 'lucide-react';
import { SYMBOLS, symbolsByGroup } from '../ml/symbols';

interface SymbolLibraryModalProps {
  isOpen: boolean;
  onClose: () => void;
}

/** Everything the network can read, with its LaTeX. Search by character, name or LaTeX. */
export const SymbolLibraryModal: React.FC<SymbolLibraryModalProps> = ({ isOpen, onClose }) => {
  const [query, setQuery] = useState('');
  const groups = useMemo(() => {
    const q = query.trim().toLowerCase();
    return symbolsByGroup()
      .map((g) => ({
        ...g,
        symbols: g.symbols.filter((s) => !q || s.char === q || s.name.toLowerCase().includes(q) || s.latex.toLowerCase().includes(q.replace(/^\\/, ''))),
      }))
      .filter((g) => g.symbols.length > 0);
  }, [query]);

  if (!isOpen) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 p-4" onClick={onClose}>
      <div role="dialog" aria-label="Symbol library" className="bg-white rounded-xl shadow-2xl border border-slate-200 w-full max-w-3xl max-h-[85vh] flex flex-col overflow-hidden" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between gap-4 px-5 py-3.5 border-b border-slate-200">
          <div>
            <h2 className="text-sm font-semibold text-slate-900">Symbols the network reads</h2>
            <p className="text-[11px] text-slate-500">{SYMBOLS.length} symbols. Sums and products take limits above and below, fractions a bar, roots an overline.</p>
          </div>
          <div className="flex items-center gap-2">
            <div className="relative">
              <Search className="w-3.5 h-3.5 text-slate-400 absolute left-2.5 top-1/2 -translate-y-1/2" />
              <input
                autoFocus
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Escape' && onClose()}
                placeholder="Search…"
                className="pl-8 pr-3 py-1.5 text-xs border border-slate-200 rounded-lg w-44 focus:outline-hidden focus:border-indigo-400"
              />
            </div>
            <button onClick={onClose} className="p-1.5 text-slate-400 hover:text-slate-700 rounded-md hover:bg-slate-100" aria-label="Close">
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>
        <div className="p-5 overflow-y-auto space-y-5">
          {groups.map((g) => (
            <div key={g.group}>
              <h3 className="text-[10px] font-semibold uppercase tracking-wider text-slate-400 mb-2">{g.label}</h3>
              <div className="grid grid-cols-[repeat(auto-fill,minmax(112px,1fr))] gap-2">
                {g.symbols.map((s) => (
                  <div key={s.char} className="px-2.5 py-2 rounded-lg border border-slate-200 bg-white text-center" title={s.latex}>
                    <div className="text-xl text-slate-900 h-8 flex items-center justify-center" dangerouslySetInnerHTML={{ __html: katex.renderToString(s.latex === '\\sqrt{}' ? '\\sqrt{\\,}' : s.latex, { throwOnError: false }) }} />
                    <div className="text-[10px] text-slate-500 leading-tight mt-1">{s.name}</div>
                  </div>
                ))}
              </div>
            </div>
          ))}
          {groups.length === 0 && <div className="text-xs text-slate-400 text-center py-10">Nothing matches “{query}”.</div>}
        </div>
      </div>
    </div>
  );
};
