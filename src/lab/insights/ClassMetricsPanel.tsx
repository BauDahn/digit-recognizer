import React, { useMemo, useState } from 'react';
import { classReport, topConfusions } from '../../ml/netStats';

type SortKey = 'f1' | 'recall' | 'precision' | 'support' | 'label';

/** Per-class precision, recall and F1, the most frequent mistakes and the global scores. */
export const ClassMetricsPanel: React.FC<{ confusion: number[][]; classes: string[] }> = ({ confusion, classes }) => {
  const [sort, setSort] = useState<SortKey>('f1');
  const report = useMemo(() => classReport(confusion, classes), [confusion, classes]);
  const confusions = useMemo(() => topConfusions(confusion, classes, 8), [confusion, classes]);

  const rows = useMemo(() => {
    const list = report.perClass.filter((c) => c.support > 0);
    return [...list].sort((a, b) => (sort === 'label' ? a.label.localeCompare(b.label) : sort === 'support' ? b.support - a.support : a[sort] - b[sort]));
  }, [report, sort]);

  if (report.total === 0) {
    return (
      <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs text-xs text-slate-400">
        <span className="font-semibold uppercase tracking-wider text-slate-500">Metrics</span>
        <p className="mt-2">Evaluate on the test set (load a model or train) to see accuracy, precision, recall and F1 per class.</p>
      </div>
    );
  }

  const bar = (v: number) => (
    <div className="h-1.5 w-16 bg-slate-100 rounded-full overflow-hidden">
      <div className={`h-full ${v > 0.9 ? 'bg-emerald-500' : v > 0.7 ? 'bg-amber-400' : 'bg-rose-400'}`} style={{ width: `${v * 100}%` }} />
    </div>
  );
  const head = (key: SortKey, label: string, right = true) => (
    <th className={`py-1.5 pr-3 font-semibold ${right ? 'text-right' : ''}`}>
      <button onClick={() => setSort(key)} className={`uppercase tracking-wider ${sort === key ? 'text-slate-900' : 'hover:text-slate-700'}`}>{label}{sort === key ? ' ↑' : ''}</button>
    </th>
  );

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
      <div className="flex flex-wrap items-baseline justify-between gap-2 mb-3">
        <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">Metrics on the test set</span>
        <span className="text-[11px] text-slate-400">{report.total.toLocaleString()} images</span>
      </div>

      <div className="grid grid-cols-3 gap-3 mb-4">
        {[
          ['Accuracy', report.accuracy, 'right answers / all'],
          ['Macro F1', report.macroF1, 'every class counts the same'],
          ['Weighted F1', report.weightedF1, 'weighted by class size'],
        ].map(([label, value, hint]) => (
          <div key={label as string} className="p-3 bg-slate-50 rounded-lg border border-slate-100">
            <div className="text-[10px] uppercase tracking-wider text-slate-400">{label}</div>
            <div className="text-xl font-mono font-semibold text-slate-900">{((value as number) * 100).toFixed(1)}%</div>
            <div className="text-[10px] text-slate-400">{hint}</div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-5">
        <div className="lg:col-span-3 max-h-72 overflow-y-auto border border-slate-100 rounded-lg">
          <table className="w-full text-xs">
            <thead className="sticky top-0 bg-white text-left text-[10px] text-slate-400 border-b border-slate-100">
              <tr>
                {head('label', 'Class', false)}
                {head('precision', 'Precision')}
                {head('recall', 'Recall')}
                {head('f1', 'F1')}
                {head('support', 'Images')}
              </tr>
            </thead>
            <tbody className="font-mono text-slate-700">
              {rows.map((c) => (
                <tr key={c.index} className="border-b border-slate-50">
                  <td className="py-1 pr-3 pl-2 font-serif text-sm text-slate-900">{c.label}</td>
                  <td className="py-1 pr-3"><div className="flex items-center justify-end gap-2">{(c.precision * 100).toFixed(0)}% {bar(c.precision)}</div></td>
                  <td className="py-1 pr-3"><div className="flex items-center justify-end gap-2">{(c.recall * 100).toFixed(0)}% {bar(c.recall)}</div></td>
                  <td className="py-1 pr-3 text-right">{(c.f1 * 100).toFixed(0)}%</td>
                  <td className="py-1 pr-3 text-right text-slate-400">{c.support}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="lg:col-span-2">
          <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-400 mb-2">Most frequent mistakes</div>
          {confusions.length === 0 ? (
            <p className="text-xs text-emerald-600">No mistakes on this test set.</p>
          ) : (
            <ul className="space-y-1.5">
              {confusions.map((c) => (
                <li key={c.actual + c.predicted} className="flex items-center justify-between text-xs">
                  <span>
                    <span className="font-serif text-base text-slate-900">{c.actual}</span>
                    <span className="text-slate-400 mx-1.5">read as</span>
                    <span className="font-serif text-base text-rose-600">{c.predicted}</span>
                  </span>
                  <span className="font-mono text-slate-500">{c.count}×</span>
                </li>
              ))}
            </ul>
          )}
          <p className="text-[11px] text-slate-400 mt-3 leading-relaxed">
            Precision: of the images the network called this class, how many were. Recall: of the images that are this class, how many it found. F1 is their harmonic mean.
          </p>
        </div>
      </div>
    </div>
  );
};
