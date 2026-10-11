import { useCallback, useReducer } from 'react';
import { MathCanvasCard, Stroke } from '../types';

/** Everything on the page. Undo and redo move through whole documents. */
export interface PageDocument {
  strokes: Stroke[];
  cards: MathCanvasCard[];
}

interface History {
  past: PageDocument[];
  present: PageDocument;
  future: PageDocument[];
}

type Action =
  | { type: 'commit'; next: PageDocument }
  | { type: 'undo' }
  | { type: 'redo' }
  | { type: 'reset'; next: PageDocument };

const LIMIT = 200;

function reducer(state: History, action: Action): History {
  switch (action.type) {
    case 'commit':
      if (action.next === state.present) return state;
      return { past: [...state.past, state.present].slice(-LIMIT), present: action.next, future: [] };
    case 'undo': {
      if (state.past.length === 0) return state;
      const previous = state.past[state.past.length - 1];
      return { past: state.past.slice(0, -1), present: previous, future: [state.present, ...state.future] };
    }
    case 'redo': {
      if (state.future.length === 0) return state;
      const [next, ...rest] = state.future;
      return { past: [...state.past, state.present], present: next, future: rest };
    }
    case 'reset':
      return { past: [], present: action.next, future: [] };
  }
}

export const EMPTY_DOCUMENT: PageDocument = { strokes: [], cards: [] };

export function useDocument(initial: PageDocument = EMPTY_DOCUMENT) {
  const [state, dispatch] = useReducer(reducer, { past: [], present: initial, future: [] });
  const commit = useCallback((next: PageDocument) => dispatch({ type: 'commit', next }), []);
  const undo = useCallback(() => dispatch({ type: 'undo' }), []);
  const redo = useCallback(() => dispatch({ type: 'redo' }), []);
  const reset = useCallback((next: PageDocument) => dispatch({ type: 'reset', next }), []);
  return {
    doc: state.present,
    canUndo: state.past.length > 0,
    canRedo: state.future.length > 0,
    commit,
    undo,
    redo,
    reset,
  };
}
