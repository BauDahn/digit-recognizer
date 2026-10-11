import React from 'react';
import { X } from 'lucide-react';
import { SHORTCUTS } from './shortcuts';

export const ShortcutsDialog: React.FC<{ onClose: () => void }> = ({ onClose }) => {
  const groups = ['Edit', 'Tools', 'Recognition', 'View'] as const;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 p-4" onClick={onClose}>
      <div role="dialog" aria-label="Keyboard shortcuts" className="bg-white rounded-xl shadow-2xl border border-slate-200 w-full max-w-2xl max-h-[85vh] overflow-hidden flex flex-col" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-slate-200">
          <h2 className="text-sm font-semibold text-slate-900">Keyboard shortcuts</h2>
          <button onClick={onClose} className="p-1.5 text-slate-400 hover:text-slate-700 rounded-md hover:bg-slate-100" aria-label="Close">
            <X className="w-4 h-4" />
          </button>
        </div>
        <div className="p-5 overflow-y-auto grid grid-cols-1 sm:grid-cols-2 gap-x-8 gap-y-5">
          {groups.map((group) => (
            <div key={group}>
              <h3 className="text-[10px] font-semibold uppercase tracking-wider text-slate-400 mb-2">{group}</h3>
              <ul className="space-y-1.5">
                {SHORTCUTS.filter((s) => s.group === group).map((s) => (
                  <li key={s.keys.join('+') + s.label} className="flex items-center justify-between gap-3 text-xs text-slate-700">
                    <span>{s.label}</span>
                    <span className="flex items-center gap-1 shrink-0">
                      {s.keys.map((k, i) => (
                        <React.Fragment key={k}>
                          {i > 0 && <span className="text-slate-300">+</span>}
                          <kbd className="px-1.5 py-0.5 rounded border border-slate-300 bg-slate-50 font-mono text-[10px] text-slate-700 shadow-2xs">{k}</kbd>
                        </React.Fragment>
                      ))}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
        <div className="px-5 py-3 border-t border-slate-200 bg-slate-50 text-[11px] text-slate-500">On a Mac, use ⌘ instead of Ctrl.</div>
      </div>
    </div>
  );
};
