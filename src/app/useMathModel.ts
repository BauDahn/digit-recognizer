import { useCallback, useEffect, useRef, useState } from 'react';
import { forgetUserModel, learnCorrection, loadMathModel, saveUserModel } from '../ml/mathModel';
import { LoadedModel, loadBuiltinModel } from '../ml/modelFile';
import { getWeights, setWeights } from '../ml/trainingClient';
import { Correction, FineTuneResult } from '../ml/userModel';

export interface Lesson {
  char: string;
  result: FineTuneResult;
}

/**
 * The math network of the app: loads the shipped model plus this person's lessons, and teaches
 * it new ones. `version` changes whenever the weights change.
 */
export function useMathModel() {
  const [status, setStatus] = useState<'loading' | 'ready' | 'error'>('loading');
  const [error, setError] = useState<string | null>(null);
  const [corrections, setCorrections] = useState<Correction[]>([]);
  const [learning, setLearning] = useState(false);
  const [lesson, setLesson] = useState<Lesson | null>(null);
  const [version, setVersion] = useState(0);
  const modelRef = useRef<LoadedModel | null>(null);
  const baseRef = useRef<LoadedModel | null>(null);
  const correctionsRef = useRef<Correction[]>([]);
  const queue = useRef<Promise<unknown>>(Promise.resolve());

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [handle, base] = await Promise.all([loadMathModel(), loadBuiltinModel('math')]);
        if (cancelled) return;
        modelRef.current = handle.model;
        baseRef.current = base;
        correctionsRef.current = handle.corrections;
        setCorrections(handle.corrections);
        setStatus('ready');
      } catch (e) {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : String(e));
        setStatus('error');
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  /** Teach the network that `pixels` (what it saw) is `char`. Lessons are applied one at a time. */
  const teach = useCallback((char: string, pixels: Uint8Array): Promise<FineTuneResult> => {
    const run = async () => {
      const model = modelRef.current;
      const base = baseRef.current;
      if (!model || !base) throw new Error('The model is not loaded yet');
      const correction: Correction = {
        id: `lesson-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
        char,
        pixels,
        createdAt: Date.now(),
      };
      const all = [...correctionsRef.current, correction];
      setLearning(true);
      try {
        const { weights, result } = await learnCorrection(model.network.config, getWeights(model.network), model.classes, all, correction.id);
        setWeights(model.network, weights);
        correctionsRef.current = all;
        setCorrections(all);
        setLesson({ char, result });
        setVersion((v) => v + 1);
        await saveUserModel(model, base, all);
        return result;
      } finally {
        setLearning(false);
      }
    };
    const next = queue.current.then(run, run);
    queue.current = next.catch(() => undefined);
    return next;
  }, []);

  /** Forget every lesson and go back to the shipped model. */
  const forget = useCallback(async () => {
    const model = modelRef.current;
    const base = baseRef.current;
    if (!model || !base) return;
    model.network.copyWeightsFrom(base.network);
    correctionsRef.current = [];
    setCorrections([]);
    setLesson(null);
    setVersion((v) => v + 1);
    await forgetUserModel();
  }, []);

  return {
    status,
    error,
    model: modelRef.current,
    base: baseRef.current,
    corrections,
    learning,
    lesson,
    version,
    teach,
    forget,
  };
}
