import React from 'react';

interface ConfusionMatrixProps {
  matrix: number[][]; // 10x10
}

export const ConfusionMatrix: React.FC<ConfusionMatrixProps> = ({ matrix }) => {
  // Find max value in matrix for color scaling
  let maxVal = 1;
  for (let r = 0; r < 10; r++) {
    for (let c = 0; c < 10; c++) {
      if (matrix[r]?.[c] > maxVal) {
        maxVal = matrix[r][c];
      }
    }
  }

  // Calculate per-digit accuracy (diagonal / row sum)
  const perDigitStats = Array.from({ length: 10 }, (_, d) => {
    const row = matrix[d] || Array(10).fill(0);
    const totalActual = row.reduce((a, b) => a + b, 0);
    const correct = row[d] || 0;
    const acc = totalActual > 0 ? (correct / totalActual) * 100 : 0;
    return { digit: d, correct, total: totalActual, acc };
  });

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-sm font-semibold text-slate-900">Confusion Matrix (10×10)</h3>
          <p className="text-xs text-slate-500">Rows = Actual Digit, Columns = Predicted Digit</p>
        </div>
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <span className="inline-block w-2.5 h-2.5 bg-emerald-600 rounded-xs"></span>
          <span>Correct (Diagonal)</span>
          <span className="inline-block w-2.5 h-2.5 bg-rose-500 rounded-xs ml-2"></span>
          <span>Confused Pairs</span>
        </div>
      </div>

      <div className="overflow-x-auto">
        <div className="inline-block min-w-full">
          <table className="border-collapse text-center mx-auto">
            <thead>
              <tr>
                <th className="p-1 text-[11px] font-medium text-slate-400">Act \ Pred</th>
                {Array.from({ length: 10 }, (_, c) => (
                  <th key={c} className="w-8 h-7 text-xs font-semibold text-slate-700 bg-slate-50">
                    {c}
                  </th>
                ))}
                <th className="p-1 text-[11px] font-medium text-slate-500 pl-3">Recall</th>
              </tr>
            </thead>
            <tbody>
              {Array.from({ length: 10 }, (_, r) => {
                const stat = perDigitStats[r];
                return (
                  <tr key={r}>
                    <td className="w-7 h-7 text-xs font-semibold text-slate-700 bg-slate-50">
                      {r}
                    </td>
                    {Array.from({ length: 10 }, (_, c) => {
                      const count = matrix[r]?.[c] || 0;
                      const isDiag = r === c;
                      const intensity = Math.min(1, count / maxVal);

                      let bg = 'bg-slate-50 text-slate-300';
                      if (count > 0) {
                        if (isDiag) {
                          bg = intensity > 0.6 ? 'bg-emerald-600 text-white' : 'bg-emerald-100 text-emerald-900';
                        } else {
                          bg = intensity > 0.4 ? 'bg-rose-500 text-white' : 'bg-rose-100 text-rose-800';
                        }
                      }

                      return (
                        <td
                          key={c}
                          title={`Actual: ${r}, Predicted: ${c}, Count: ${count}`}
                          className={`w-8 h-8 text-[11px] font-mono border border-white transition-colors cursor-default ${bg}`}
                        >
                          {count > 0 ? count : '-'}
                        </td>
                      );
                    })}
                    <td className="pl-3 text-xs font-mono tabular-nums text-slate-600 text-left">
                      {stat.total > 0 ? `${stat.acc.toFixed(0)}%` : '—'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
