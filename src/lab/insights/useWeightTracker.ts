import { useCallback, useRef, useState } from 'react';
import { copyWeights, weightChange } from '../../ml/netStats';
import { NeuralNetwork } from '../../ml/neuralNetwork';
import { getWeights, NetworkWeights } from '../../ml/trainingClient';
import { DynamicsPoint } from './WeightDynamics';

/**
 * Remembers the weights before every update so the lab can show what changed
 * (diagram highlights, "how the weights move" chart).
 */
export function useWeightTracker(network: NeuralNetwork) {
  const previous = useRef<NetworkWeights | null>(null);
  const last = useRef<NetworkWeights | null>(null);
  const [points, setPoints] = useState<DynamicsPoint[]>([]);
  const [version, setVersion] = useState(0);

  /** Call after the network's weights changed because of training. `at` is the position in epochs. */
  const recordUpdate = useCallback(
    (at: number) => {
      const now = copyWeights(getWeights(network));
      if (last.current) {
        previous.current = last.current;
        const changes = weightChange(last.current, now);
        setPoints((p) => [...p.slice(-399), { at, changes }]);
      }
      last.current = now;
      setVersion((v) => v + 1);
    },
    [network]
  );

  /** Call when the weights were replaced as a whole (reset, import, pre-trained): no history. */
  const reset = useCallback(() => {
    previous.current = null;
    last.current = copyWeights(getWeights(network));
    setPoints([]);
    setVersion((v) => v + 1);
  }, [network]);

  /** Take the current weights as the starting point of the next update, keeping the history. */
  const rebase = useCallback(() => {
    last.current = copyWeights(getWeights(network));
  }, [network]);

  return { previous: previous.current, version, points, recordUpdate, reset, rebase };
}

export type WeightTracker = ReturnType<typeof useWeightTracker>;
