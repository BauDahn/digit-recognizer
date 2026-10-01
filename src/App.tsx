import React, { useState, useRef, useCallback } from 'react';
import { AppMode } from './types';
import { NeuralNetwork } from './ml/neuralNetwork';
import { Header } from './components/Header';
import { DigitLab } from './components/DigitLab/DigitLab';
import { AlphanumericStudio } from './components/Alphanumeric/AlphanumericStudio';
import { InfiniteMathCanvas } from './components/InfiniteCanvas/InfiniteMathCanvas';
import { SymbolLibraryModal } from './components/SymbolLibraryModal';

export default function App() {
  const [currentMode, setCurrentMode] = useState<AppMode>('infinite-canvas');
  const [isSymbolLibraryOpen, setIsSymbolLibraryOpen] = useState(false);

  // Shared neural network instance
  const networkRef = useRef<NeuralNetwork>(new NeuralNetwork({ hiddenSize1: 64, hiddenSize2: 32 }));
  const [, setTick] = useState(0);

  const handleNetworkUpdate = useCallback(() => {
    setTick((t) => t + 1);
  }, []);

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-slate-50 text-slate-900 font-sans">
      {/* Universal Top Navigation Contract */}
      <Header
        currentMode={currentMode}
        onSelectMode={(mode) => setCurrentMode(mode)}
        onOpenSymbolLibrary={() => setIsSymbolLibraryOpen(true)}
      />

      {/* Main Content Area */}
      <main className="flex-1 overflow-y-auto">
        {currentMode === 'digit-lab' && (
          <DigitLab
            network={networkRef.current}
            onNetworkUpdate={handleNetworkUpdate}
          />
        )}

        {currentMode === 'alphanumeric' && (
          <AlphanumericStudio />
        )}

        {currentMode === 'infinite-canvas' && (
          <InfiniteMathCanvas />
        )}
      </main>

      {/* Global Mathematical & Alphanumeric Symbol Catalog */}
      <SymbolLibraryModal
        isOpen={isSymbolLibraryOpen}
        onClose={() => setIsSymbolLibraryOpen(false)}
      />
    </div>
  );
}
