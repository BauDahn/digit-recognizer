import React, { useEffect, useRef } from 'react';
import { MathRecognitionResult } from '../../types';
import { X, Copy, Check, Calculator, Sparkles, PlusCircle } from 'lucide-react';
import katex from 'katex';

interface MathResultModalProps {
  isOpen: boolean;
  result: MathRecognitionResult | null;
  isLoading: boolean;
  onClose: () => void;
  onInsertLatexCard?: (latex: string) => void;
}

export const MathResultModal: React.FC<MathResultModalProps> = ({
  isOpen,
  result,
  isLoading,
  onClose,
  onInsertLatexCard,
}) => {
  const katexFormulaRef = useRef<HTMLDivElement | null>(null);
  const katexSolutionRef = useRef<HTMLDivElement | null>(null);
  const [copied, setCopied] = React.useState(false);

  // Render KaTeX formula whenever result changes
  useEffect(() => {
    if (!result?.latex) return;

    if (katexFormulaRef.current) {
      try {
        katex.render(result.latex, katexFormulaRef.current, {
          displayMode: true,
          throwOnError: false,
        });
      } catch (e) {
        katexFormulaRef.current.innerText = result.latex;
      }
    }

    if (katexSolutionRef.current && result.solution) {
      try {
        katex.render(result.solution, katexSolutionRef.current, {
          displayMode: false,
          throwOnError: false,
        });
      } catch (e) {
        katexSolutionRef.current.innerText = result.solution;
      }
    }
  }, [result]);

  if (!isOpen) return null;

  const handleCopyLatex = () => {
    if (!result?.latex) return;
    navigator.clipboard.writeText(result.latex);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 backdrop-blur-xs p-4">
      <div className="bg-white rounded-xl shadow-2xl border border-slate-200 w-full max-w-xl max-h-[85vh] flex flex-col overflow-hidden animate-in fade-in zoom-in-95 duration-150">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200 bg-slate-50/50">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-blue-50 border border-blue-200 flex items-center justify-center text-blue-600">
              <Calculator className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-slate-900">
                Mathematical Handwriting Analysis
              </h2>
              <p className="text-[11px] text-slate-500">
                Formula recognition & step-by-step math solver
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 text-slate-400 hover:text-slate-600 rounded-md hover:bg-slate-100 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto space-y-5">
          {isLoading ? (
            <div className="py-12 flex flex-col items-center justify-center space-y-3">
              <div className="w-8 h-8 border-2 border-slate-900 border-t-transparent rounded-full animate-spin"></div>
              <p className="text-xs text-slate-500 font-medium">
                Analyzing handwritten strokes & symbols with Gemini Vision...
              </p>
            </div>
          ) : result ? (
            <>
              {/* LaTeX Formula View */}
              <div className="p-5 bg-slate-50 rounded-xl border border-slate-200 text-center">
                <div className="text-[11px] text-slate-400 uppercase tracking-wider font-semibold mb-2">
                  Recognized LaTeX Formula
                </div>
                <div
                  ref={katexFormulaRef}
                  className="overflow-x-auto py-2 text-slate-900 text-lg min-h-12 flex items-center justify-center font-serif"
                />
                <div className="mt-2 text-xs font-mono text-slate-500 bg-white/80 py-1 px-2 rounded border border-slate-200 inline-block max-w-full truncate">
                  {result.latex}
                </div>
              </div>

              {/* Solved Result Card */}
              {result.solution && (
                <div className="p-4 bg-emerald-50/60 rounded-xl border border-emerald-200 space-y-1">
                  <div className="flex items-center gap-1.5 text-xs font-semibold text-emerald-800">
                    <Sparkles className="w-3.5 h-3.5 text-emerald-600" />
                    <span>Evaluated Solution:</span>
                  </div>
                  <div className="text-base font-semibold text-emerald-950 font-serif">
                    <span ref={katexSolutionRef}>{result.solution}</span>
                  </div>
                </div>
              )}

              {/* Step-by-Step Explanation */}
              {result.steps && result.steps.length > 0 && (
                <div className="space-y-2">
                  <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                    Step-by-Step Solving Derivation:
                  </h4>
                  <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 space-y-2">
                    {result.steps.map((step, idx) => (
                      <div key={idx} className="flex items-start gap-2.5 text-xs text-slate-700">
                        <span className="w-4 h-4 rounded-full bg-slate-200 text-slate-700 font-mono text-[10px] flex items-center justify-center shrink-0 mt-0.5">
                          {idx + 1}
                        </span>
                        <span className="leading-relaxed">{step}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Detected Symbols Breakdown */}
              {result.symbolsDetected && result.symbolsDetected.length > 0 && (
                <div>
                  <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-2">
                    Identified Symbols & Tokens:
                  </h4>
                  <div className="flex flex-wrap gap-1.5">
                    {result.symbolsDetected.map((sym, i) => (
                      <span
                        key={i}
                        className="px-2 py-0.5 text-xs font-mono bg-white border border-slate-200 rounded text-slate-800"
                      >
                        {sym}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="py-8 text-center text-xs text-slate-500">
              No recognition result available.
            </div>
          )}
        </div>

        {/* Footer Actions */}
        <div className="px-6 py-3.5 border-t border-slate-200 bg-slate-50 flex items-center justify-between">
          <button
            onClick={handleCopyLatex}
            disabled={!result?.latex}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-white border border-slate-200 rounded-md hover:bg-slate-100 transition-colors"
          >
            {copied ? (
              <>
                <Check className="w-3.5 h-3.5 text-emerald-600" />
                <span className="text-emerald-600">Copied LaTeX</span>
              </>
            ) : (
              <>
                <Copy className="w-3.5 h-3.5 text-slate-500" />
                <span>Copy LaTeX</span>
              </>
            )}
          </button>

          <div className="flex items-center gap-2">
            {result?.latex && onInsertLatexCard && (
              <button
                onClick={() => {
                  onInsertLatexCard(result.latex);
                  onClose();
                }}
                className="flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-medium text-slate-700 bg-white border border-slate-200 rounded-md hover:bg-slate-100 transition-colors"
              >
                <PlusCircle className="w-3.5 h-3.5 text-slate-600" />
                <span>Insert Sticky Card</span>
              </button>
            )}
            <button
              onClick={onClose}
              className="px-4 py-1.5 text-xs font-medium text-white bg-slate-900 rounded-md hover:bg-slate-800 transition-colors"
            >
              Done
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
