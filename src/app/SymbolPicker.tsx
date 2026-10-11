import React, { useEffect, useMemo, useRef, useState } from 'react';
import { X } from 'lucide-react';
import { RecognizedSymbol } from '../ml/mathLayout';
import { displayChar } from '../ml/strokeRecognizer';
import { SYMBOLS, symbolsByGroup, SymbolInfo } from '../ml/symbols';
import { PixelPreview } from './SymbolTile';

interface SymbolPickerProps {
  symbol: RecognizedSymbol;
  /** Classes the network can output; only these can be taught */
  classes: string[];
  onPick: (char: string) => void;
  onClose: () => void;
}

function matches(info: SymbolInfo, query: string): boolean {
  const q = query.trim().toLowerCase();
  if (!q) return true;
  return (
    info.char.toLowerCase() === q ||
    info.name.toLowerCase().includes(q) ||
    info.latex.toLowerCase().replace(/\\/g, '').includes(q.replace(/\\/g, '')) ||
    info.group.includes(q)
  );
}

/** "What was this symbol?": the network's other guesses and the whole alphabet to choose from. */
export const SymbolPicker: React.FC<SymbolPickerProps> = ({ symbol, classes, onPick, onClose }) => {
  const [query, setQuery] = useState('');
  const inputRef = useRef<HTMLInputElement | null>(null);
  useEffect(() => inputRef.current?.focus(), []);

  const known = useMemo(() => new Set(classes), [classes]);
  const groups = useMemo(
    () => symbolsByGroup().map((g) => ({ ...g, symbols: g.symbols.filter((s) => known.has(s.char) && matches(s, query)) })).filter((g) => g.symbols.length > 0),
    [known, query]
  );
  const first = groups[0]?.symbols[0];
  const isDot = symbol.pixels === null;

  return (
    <div
      className="absolute right-3 top-3 bottom-3 z-40 w-[340px] max-w-[calc(100%-1.5rem)] bg-white rounded-xl border border-slate-200 shadow-2xl flex flex-col overflow-hidden"
      role="dialog"
      aria-label="Correct a symbol"
      onKeyDown={(e) => {
        if (e.key === 'Escape') {
          e.stopPropagation();
          onClose();
        } else if (e.key === 'Enter' && first) {
          e.stopPropagation();
          onPick(first.char);
        }
      }}
    >
      <div className="flex items-center gap-3 px-4 py-3 border-b border-slate-200 bg-slate-50">
        {symbol.pixels && <PixelPreview pixels={symbol.pixels} className="w-14 h-14 border border-slate-200 bg-white [image-rendering:pixelated]" />}
        <div className="flex-1 min-w-0">
          <div className="text-sm font-semibold text-slate-900">What was this symbol?</div>
          <div className="text-[11px] text-slate-500">
            The network read <span className="font-serif text-base text-slate-900">{displayChar(symbol.char)}</span> ({(symbol.confidence * 100).toFixed(0)}%). Choose the right one and it will remember.
          </div>
        </div>
        <button onClick={onClose} className="p-1.5 text-slate-400 hover:text-slate-700 rounded-md hover:bg-slate-100" aria-label="Close">
          <X className="w-4 h-4" />
        </button>
      </div>

      {isDot ? (
        <div className="p-4 space-y-3">
          <p className="text-xs text-slate-500">This mark was told apart by its size. It is either a decimal point or a multiplication dot.</p>
          <div className="flex gap-2">
            {[
              { char: '.', label: 'decimal point' },
              { char: '*', label: 'multiplication dot' },
            ].map((o) => (
              <button key={o.char} onClick={() => onPick(o.char)} className="flex-1 py-2 text-xs rounded-lg border border-slate-200 hover:bg-slate-50">
                <span className="block font-serif text-lg">{o.char === '*' ? '·' : '.'}</span>
                {o.label}
              </button>
            ))}
          </div>
        </div>
      ) : (
        <>
          <div className="px-4 pt-3">
            <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-400 mb-1.5">The network's guesses</div>
            <div className="flex flex-wrap gap-1.5">
              {symbol.candidates.map((c) => (
                <button
                  key={c.char}
                  onClick={() => onPick(c.char)}
                  className={`px-2.5 py-1.5 rounded-lg border text-center min-w-[52px] hover:bg-slate-50 ${c.char === symbol.char ? 'border-slate-400 bg-slate-50' : 'border-slate-200'}`}
                >
                  <div className="font-serif text-lg leading-none">{displayChar(c.char)}</div>
                  <div className="text-[10px] text-slate-500 mt-0.5">{(c.confidence * 100).toFixed(0)}%</div>
                </button>
              ))}
            </div>
          </div>

          <div className="px-4 pt-3 pb-2">
            <input
              ref={inputRef}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search: ∂, alpha, sum, integral…"
              className="w-full px-3 py-1.5 text-xs border border-slate-200 rounded-lg focus:outline-hidden focus:border-indigo-400"
            />
          </div>

          <div className="flex-1 overflow-y-auto px-4 pb-4 space-y-3">
            {groups.map((g) => (
              <div key={g.group}>
                <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-400 mb-1">{g.label}</div>
                <div className="flex flex-wrap gap-1">
                  {g.symbols.map((s) => (
                    <button
                      key={s.char}
                      onClick={() => onPick(s.char)}
                      title={`${s.name}  (${s.latex})`}
                      className={`w-8 h-8 rounded-md border text-base font-serif hover:bg-indigo-50 hover:border-indigo-300 ${
                        s.char === symbol.char ? 'border-slate-400 bg-slate-100' : 'border-slate-200 bg-white'
                      }`}
                    >
                      {s.char}
                    </button>
                  ))}
                </div>
              </div>
            ))}
            {groups.length === 0 && <div className="text-xs text-slate-400 py-6 text-center">No symbol matches “{query}”. There are {SYMBOLS.length} in total.</div>}
          </div>
        </>
      )}
    </div>
  );
};
