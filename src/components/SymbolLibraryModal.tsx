import React, { useState } from 'react';
import { X, Search, Check } from 'lucide-react';

interface SymbolLibraryModalProps {
  isOpen: boolean;
  onClose: () => void;
  onInsertSymbol?: (latex: string) => void;
}

interface SymbolItem {
  name: string;
  latex: string;
  display: string;
  category: 'greek' | 'calculus' | 'algebra' | 'logic' | 'geometry';
  example: string;
}

const SYMBOL_CATALOG: SymbolItem[] = [
  // Greek
  { name: 'Alpha', latex: '\\alpha', display: 'α', category: 'greek', example: '2\\alpha + 5 = 15' },
  { name: 'Beta', latex: '\\beta', display: 'β', category: 'greek', example: '\\beta = \\frac{\\pi}{4}' },
  { name: 'Gamma', latex: '\\gamma', display: 'γ', category: 'greek', example: '\\gamma = 0.5772' },
  { name: 'Delta', latex: '\\Delta', display: 'Δ', category: 'greek', example: '\\Delta = b^2 - 4ac' },
  { name: 'Theta', latex: '\\theta', display: 'θ', category: 'greek', example: '\\sin(\\theta) = 1' },
  { name: 'Lambda', latex: '\\lambda', display: 'λ', category: 'greek', example: '\\lambda = \\frac{h}{p}' },
  { name: 'Mu', latex: '\\mu', display: 'μ', category: 'greek', example: '\\mu = 10^{-6}' },
  { name: 'Pi', latex: '\\pi', display: 'π', category: 'greek', example: 'A = \\pi r^2' },
  { name: 'Sigma', latex: '\\sigma', display: 'σ', category: 'greek', example: '\\sigma = \\sqrt{s^2}' },
  { name: 'Omega', latex: '\\omega', display: 'ω', category: 'greek', example: 'E = \\hbar \\omega' },

  // Calculus
  { name: 'Integral', latex: '\\int', display: '∫', category: 'calculus', example: '\\int x^2 dx = \\frac{x^3}{3}' },
  { name: 'Definite Integral', latex: '\\int_{a}^{b}', display: '∫ₐᵇ', category: 'calculus', example: '\\int_{0}^{1} 2x dx = 1' },
  { name: 'Derivative', latex: '\\frac{d}{dx}', display: 'd/dx', category: 'calculus', example: '\\frac{d}{dx}(x^3) = 3x^2' },
  { name: 'Partial Derivative', latex: '\\partial', display: '∂', category: 'calculus', example: '\\frac{\\partial f}{\\partial x}' },
  { name: 'Summation', latex: '\\sum', display: '∑', category: 'calculus', example: '\\sum_{i=1}^{n} i = \\frac{n(n+1)}{2}' },
  { name: 'Limit', latex: '\\lim', display: 'lim', category: 'calculus', example: '\\lim_{x \\to 0} \\frac{\\sin x}{x} = 1' },
  { name: 'Infinity', latex: '\\infty', display: '∞', category: 'calculus', example: '\\int_{0}^{\\infty} e^{-x} dx = 1' },

  // Algebra & Arithmetic
  { name: 'Square Root', latex: '\\sqrt{x}', display: '√', category: 'algebra', example: '\\sqrt{144} = 12' },
  { name: 'Fraction', latex: '\\frac{a}{b}', display: 'a/b', category: 'algebra', example: '\\frac{3}{4} + \\frac{1}{2} = \\frac{5}{4}' },
  { name: 'Exponent', latex: 'x^n', display: 'xⁿ', category: 'algebra', example: '2^5 = 32' },
  { name: 'Subscript', latex: 'x_i', display: 'xᵢ', category: 'algebra', example: 'x_1 + x_2 = 10' },
  { name: 'Plus-Minus', latex: '\\pm', display: '±', category: 'algebra', example: 'x = \\pm 4' },
  { name: 'Not Equal', latex: '\\neq', display: '≠', category: 'algebra', example: 'x \\neq 0' },
  { name: 'Less or Equal', latex: '\\leq', display: '≤', category: 'algebra', example: '2x \\leq 10' },
  { name: 'Greater or Equal', latex: '\\geq', display: '≥', category: 'algebra', example: 'y \\geq 0' },
  { name: 'Multiplication Dot', latex: '\\cdot', display: '·', category: 'algebra', example: '3 \\cdot 5 = 15' },

  // Logic & Sets
  { name: 'Element Of', latex: '\\in', display: '∈', category: 'logic', example: 'x \\in \\mathbb{R}' },
  { name: 'Subset', latex: '\\subset', display: '⊂', category: 'logic', example: 'A \\subset B' },
  { name: 'Union', latex: '\\cup', display: '∪', category: 'logic', example: 'A \\cup B' },
  { name: 'Intersection', latex: '\\cap', display: '∩', category: 'logic', example: 'A \\cap B' },
  { name: 'For All', latex: '\\forall', display: '∀', category: 'logic', example: '\\forall x > 0' },
  { name: 'Exists', latex: '\\exists', display: '∃', category: 'logic', example: '\\exists y \\in S' },
  { name: 'Implies', latex: '\\implies', display: '⇒', category: 'logic', example: 'P \\implies Q' },
];

export const SymbolLibraryModal: React.FC<SymbolLibraryModalProps> = ({
  isOpen,
  onClose,
  onInsertSymbol,
}) => {
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [copiedId, setCopiedId] = useState<string | null>(null);

  if (!isOpen) return null;

  const filteredSymbols = SYMBOL_CATALOG.filter((item) => {
    const matchesCat = selectedCategory === 'all' || item.category === selectedCategory;
    const matchesQuery =
      item.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.latex.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.display.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesCat && matchesQuery;
  });

  const handleCopy = (latex: string, name: string) => {
    navigator.clipboard.writeText(latex);
    setCopiedId(name);
    setTimeout(() => setCopiedId(null), 1500);
    if (onInsertSymbol) {
      onInsertSymbol(latex);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 backdrop-blur-xs p-4">
      <div className="bg-white rounded-xl shadow-xl border border-slate-200 w-full max-w-2xl max-h-[85vh] flex flex-col overflow-hidden animate-in fade-in zoom-in-95 duration-150">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200">
          <div>
            <h2 className="text-base font-semibold text-slate-900">
              Mathematical & Alphanumeric Symbol Catalog
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              LaTeX reference for sticky notes. The handwriting network on the canvas reads digits, + − / = ( ), x,
              dots and exponents.
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 text-slate-400 hover:text-slate-600 rounded-md hover:bg-slate-100 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Filter Bar */}
        <div className="px-6 py-3 border-b border-slate-100 bg-slate-50/50 flex flex-col sm:flex-row items-center gap-3">
          <div className="relative w-full sm:w-64">
            <Search className="w-3.5 h-3.5 absolute left-3 top-2.5 text-slate-400" />
            <input
              type="text"
              placeholder="Search symbols or LaTeX..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-8 pr-3 py-1.5 text-xs bg-white border border-slate-200 rounded-md focus:outline-hidden focus:border-slate-400 focus:ring-1 focus:ring-slate-300"
            />
          </div>

          <div className="flex items-center gap-1 overflow-x-auto w-full pb-1 sm:pb-0">
            {['all', 'greek', 'calculus', 'algebra', 'logic'].map((cat) => (
              <button
                key={cat}
                onClick={() => setSelectedCategory(cat)}
                className={`px-2.5 py-1 text-xs rounded-md capitalize font-medium transition-colors whitespace-nowrap ${
                  selectedCategory === cat
                    ? 'bg-slate-900 text-white'
                    : 'bg-white border border-slate-200 text-slate-600 hover:bg-slate-100'
                }`}
              >
                {cat}
              </button>
            ))}
          </div>
        </div>

        {/* Grid List */}
        <div className="p-6 overflow-y-auto grid grid-cols-1 sm:grid-cols-2 gap-3 max-h-[50vh]">
          {filteredSymbols.map((item) => (
            <div
              key={item.name}
              className="p-3 border border-slate-200 rounded-lg hover:border-slate-300 hover:bg-slate-50/50 transition-colors flex items-center justify-between group"
            >
              <div className="flex items-center gap-3">
                <span className="w-9 h-9 flex items-center justify-center bg-slate-100 rounded-md text-lg font-serif text-slate-900 shrink-0">
                  {item.display}
                </span>
                <div>
                  <div className="text-xs font-semibold text-slate-800">{item.name}</div>
                  <div className="text-xs text-slate-500 font-mono mt-0.5">{item.latex}</div>
                </div>
              </div>

              <button
                onClick={() => handleCopy(item.latex, item.name)}
                className="px-2.5 py-1 text-xs font-medium text-slate-600 bg-white border border-slate-200 rounded hover:bg-slate-100 transition-colors flex items-center gap-1 shrink-0"
              >
                {copiedId === item.name ? (
                  <>
                    <Check className="w-3 h-3 text-emerald-600" />
                    <span className="text-emerald-600">Copied</span>
                  </>
                ) : (
                  <span>Copy LaTeX</span>
                )}
              </button>
            </div>
          ))}
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-slate-200 bg-slate-50 flex items-center justify-between text-xs text-slate-500">
          <span>Draw any of these directly on the OneNote canvas!</span>
          <button
            onClick={onClose}
            className="px-4 py-1.5 bg-slate-900 text-white rounded-md text-xs font-medium hover:bg-slate-800"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
};
