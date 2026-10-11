import React from 'react';
import { NeuralNetwork } from '../../ml/neuralNetwork';
import { ClassMetricsPanel } from './ClassMetricsPanel';
import { NetworkDiagram } from './NetworkDiagram';
import { ParameterPanel } from './ParameterPanel';
import { WeightTracker } from './useWeightTracker';
import { WeightDynamics } from './WeightDynamics';

interface Props {
  network: NeuralNetwork;
  classes: string[];
  /** The drawing the network is looking at (28x28) */
  pixels: Float32Array | null;
  tracker: WeightTracker;
  confusion: number[][];
  training: boolean;
  hyper?: { learningRate: number; momentum: number; batchSize: number; epochs?: number; augmentation?: boolean };
}

/** Everything about the inside of a network: the graph, the parameters, how they move and the metrics. */
export const LabInsights: React.FC<Props> = ({ network, classes, pixels, tracker, confusion, training, hyper }) => (
  <section className="space-y-6" aria-label="Inside the network">
    <NetworkDiagram network={network} classes={classes} pixels={pixels} previous={tracker.previous} version={tracker.version} training={training} />
    <div className="grid grid-cols-1 xl:grid-cols-5 gap-6">
      <div className="xl:col-span-3">
        <ParameterPanel network={network} version={tracker.version} hyper={hyper} />
      </div>
      <div className="xl:col-span-2">
        <WeightDynamics points={tracker.points} />
      </div>
    </div>
    <ClassMetricsPanel confusion={confusion} classes={classes} />
  </section>
);
