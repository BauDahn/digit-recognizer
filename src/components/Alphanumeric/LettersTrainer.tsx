import React, { useEffect, useRef, useState } from 'react';
import { Play, Square, RotateCcw } from 'lucide-react';
import { EMNIST_BALANCED_CLASSES } from '../../ml/classes';
import { datasetUrl, fetchDataset, toTrainingSamples } from '../../ml/datasets';
import { LoadedModel, removeModelFromStorage, saveModelToStorage, toModelFile, loadBuiltinModel } from '../../ml/modelFile';
import { NeuralNetwork } from '../../ml/neuralNetwork';
import { createTrainerWorker, packSamples, setWeights, TrainRequest, WorkerMessage } from '../../ml/trainingClient';
import { TrainingEpochMetric } from '../../types';
import { TrainingCharts } from '../DigitLab/TrainingCharts';

/**
 * Train a letters + digits network from scratch on EMNIST Balanced, in a Web Worker.
 * The browser subset has 14,100 training images (300 per class) and 2,820 test images.
 */
export const LettersTrainer: React.FC<{ onModelTrained: (model: LoadedModel) => void }> = ({ onModelTrained }) => {
  const [epochs, setEpochs] = useState(8);
  const [status, setStatus] = useState<'idle' | 'loading' | 'training'>('idle');
  const [metrics, setMetrics] = useState<TrainingEpochMetric[]>([]);
  const [error, setError] = useState<string | null>(null);
  const workerRef = useRef<Worker | null>(null);

  useEffect(() => () => workerRef.current?.terminate(), []);

  const start = async () => {
    setError(null);
    setMetrics([]);
    setStatus('loading');
    try {
      const [train, test] = await Promise.all([
        fetchDataset(datasetUrl('emnist-balanced-train')),
        fetchDataset(datasetUrl('emnist-balanced-test')),
      ]);
      const network = new NeuralNetwork({ hiddenSize1: 128, hiddenSize2: 64, outputSize: EMNIST_BALANCED_CLASSES.length });
      const worker = createTrainerWorker();
      workerRef.current = worker;
      setStatus('training');

      worker.onmessage = (event: MessageEvent<WorkerMessage>) => {
        const msg = event.data;
        if (msg.type === 'epoch') {
          setWeights(network, msg.weights);
          setMetrics((prev) => [
            ...prev,
            { epoch: msg.epoch, loss: msg.loss, accuracy: msg.testAccuracy, trainAccuracy: msg.trainAccuracy, seconds: msg.seconds },
          ]);
          const file = toModelFile('letters', EMNIST_BALANCED_CLASSES, network, {
            trainedOn: 'EMNIST Balanced subset (trained in your browser)',
            testAccuracy: msg.testAccuracy,
            epochs: msg.epoch,
            createdAt: new Date().toISOString().slice(0, 10),
          });
          saveModelToStorage('letters', file);
          onModelTrained({ name: 'letters', classes: EMNIST_BALANCED_CLASSES, network, meta: file.meta });
        } else {
          if (msg.type === 'error') setError(msg.message);
          worker.terminate();
          workerRef.current = null;
          setStatus('idle');
        }
      };

      const request: TrainRequest = {
        type: 'train',
        config: network.config,
        initialWeights: null,
        train: packSamples(toTrainingSamples(train, 'emnist-train')),
        test: packSamples(toTrainingSamples(test, 'emnist-test')),
        epochs,
        augmentProbability: 0.3,
      };
      worker.postMessage(request);
    } catch (e: any) {
      setError(String(e?.message || e));
      setStatus('idle');
    }
  };

  const stop = () => {
    workerRef.current?.terminate();
    workerRef.current = null;
    setStatus('idle');
  };

  const restorePretrained = async () => {
    stop();
    removeModelFromStorage('letters');
    setMetrics([]);
    onModelTrained(await loadBuiltinModel('letters'));
  };

  const last = metrics[metrics.length - 1];

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs grid grid-cols-1 lg:grid-cols-12 gap-6">
      <div className="lg:col-span-5 space-y-3">
        <div>
          <h3 className="text-sm font-semibold text-slate-900">Train your own letters network</h3>
          <p className="text-xs text-slate-500">
            Same code as the digit network, but with 47 output neurons and wider hidden layers (784-128-64-47). Training
            starts from random weights on a subset of EMNIST Balanced; each epoch takes a few seconds. Your model is
            saved in this browser and replaces the pre-trained one.
          </p>
        </div>
        <div className="flex items-center gap-3 text-xs">
          <label className="flex items-center gap-2 text-slate-600">
            Epochs
            <input
              type="range"
              min={1}
              max={20}
              value={epochs}
              disabled={status !== 'idle'}
              onChange={(e) => setEpochs(Number(e.target.value))}
            />
            <span className="font-mono font-semibold w-5">{epochs}</span>
          </label>
        </div>
        <div className="flex gap-2">
          {status === 'idle' ? (
            <button
              onClick={start}
              className="flex items-center gap-1.5 py-2 px-3 bg-slate-900 text-white rounded-lg text-xs font-medium hover:bg-slate-800"
            >
              <Play className="w-3.5 h-3.5 fill-current" /> Train from scratch
            </button>
          ) : (
            <button
              onClick={stop}
              disabled={status === 'loading'}
              className="flex items-center gap-1.5 py-2 px-3 bg-amber-600 text-white rounded-lg text-xs font-medium hover:bg-amber-700 disabled:opacity-60"
            >
              <Square className="w-3.5 h-3.5 fill-current" /> {status === 'loading' ? 'Loading EMNIST…' : 'Stop'}
            </button>
          )}
          <button
            onClick={restorePretrained}
            className="flex items-center gap-1.5 py-2 px-3 border border-slate-200 rounded-lg text-xs text-slate-700 hover:bg-slate-50"
            title="Forget your model and use the pre-trained one"
          >
            <RotateCcw className="w-3.5 h-3.5" /> Use pre-trained
          </button>
        </div>
        {last && (
          <div className="text-xs text-slate-600 font-mono">
            epoch {last.epoch}/{epochs} · test accuracy {(last.accuracy * 100).toFixed(1)}%
          </div>
        )}
        {error && <div className="text-xs text-rose-600">{error}</div>}
      </div>
      <div className="lg:col-span-7">
        {metrics.length > 0 ? (
          <TrainingCharts metrics={metrics} />
        ) : (
          <div className="h-full min-h-24 flex items-center justify-center text-xs text-slate-400 bg-slate-50 rounded-lg">
            Learning curves appear here while training.
          </div>
        )}
      </div>
    </div>
  );
};
