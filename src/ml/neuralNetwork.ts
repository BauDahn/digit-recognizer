import { NetworkConfig, PredictionResult, TrainingEpochMetric, TrainingSample } from '../types';

export class NeuralNetwork {
  config: NetworkConfig;

  // Layer dimensions
  // 784 -> h1 -> h2 -> 10
  w1: Float32Array; // [h1 x 784]
  b1: Float32Array; // [h1]
  w2: Float32Array; // [h2 x h1]
  b2: Float32Array; // [h2]
  w3: Float32Array; // [10 x h2]
  b3: Float32Array; // [10]

  // Momentum velocity buffers
  vw1: Float32Array;
  vb1: Float32Array;
  vw2: Float32Array;
  vb2: Float32Array;
  vw3: Float32Array;
  vb3: Float32Array;

  isTrained = false;

  constructor(config?: Partial<NetworkConfig>) {
    this.config = {
      inputSize: 784,
      hiddenSize1: config?.hiddenSize1 ?? 64,
      hiddenSize2: config?.hiddenSize2 ?? 32,
      outputSize: 10,
      learningRate: config?.learningRate ?? 0.04,
      momentum: config?.momentum ?? 0.85,
      batchSize: config?.batchSize ?? 16,
      epochs: config?.epochs ?? 15,
    };

    const { inputSize, hiddenSize1, hiddenSize2, outputSize } = this.config;

    this.w1 = new Float32Array(hiddenSize1 * inputSize);
    this.b1 = new Float32Array(hiddenSize1);
    this.w2 = new Float32Array(hiddenSize2 * hiddenSize1);
    this.b2 = new Float32Array(hiddenSize2);
    this.w3 = new Float32Array(outputSize * hiddenSize2);
    this.b3 = new Float32Array(outputSize);

    this.vw1 = new Float32Array(this.w1.length);
    this.vb1 = new Float32Array(this.b1.length);
    this.vw2 = new Float32Array(this.w2.length);
    this.vb2 = new Float32Array(this.b2.length);
    this.vw3 = new Float32Array(this.w3.length);
    this.vb3 = new Float32Array(this.b3.length);

    this.initWeightsHe();
  }

  /**
   * He (Kaiming) weight initialization for ReLU / LeakyReLU activations
   */
  initWeightsHe() {
    const initLayer = (weights: Float32Array, fanIn: number) => {
      const std = Math.sqrt(2.0 / fanIn);
      for (let i = 0; i < weights.length; i++) {
        // Box-Muller normal distribution
        const u1 = Math.max(1e-7, Math.random());
        const u2 = Math.random();
        const randStd = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
        weights[i] = randStd * std;
      }
    };

    initLayer(this.w1, this.config.inputSize);
    initLayer(this.w2, this.config.hiddenSize1);
    initLayer(this.w3, this.config.hiddenSize2);

    this.b1.fill(0.01);
    this.b2.fill(0.01);
    this.b3.fill(0.01);

    this.vw1.fill(0);
    this.vb1.fill(0);
    this.vw2.fill(0);
    this.vb2.fill(0);
    this.vw3.fill(0);
    this.vb3.fill(0);
    this.isTrained = false;
  }

  // Leaky ReLU activation
  private activate(x: number): number {
    return x > 0 ? x : 0.01 * x;
  }

  private activateDeriv(x: number): number {
    return x > 0 ? 1 : 0.01;
  }

  /**
   * Forward pass: computes all intermediate activations
   */
  forward(x: Float32Array): {
    h1Raw: Float32Array;
    h1Act: Float32Array;
    h2Raw: Float32Array;
    h2Act: Float32Array;
    outRaw: Float32Array;
    probabilities: Float32Array;
  } {
    const { inputSize, hiddenSize1, hiddenSize2, outputSize } = this.config;

    // Layer 1
    const h1Raw = new Float32Array(hiddenSize1);
    const h1Act = new Float32Array(hiddenSize1);
    for (let i = 0; i < hiddenSize1; i++) {
      let sum = this.b1[i];
      const rowOffset = i * inputSize;
      for (let j = 0; j < inputSize; j++) {
        sum += this.w1[rowOffset + j] * x[j];
      }
      h1Raw[i] = sum;
      h1Act[i] = this.activate(sum);
    }

    // Layer 2
    const h2Raw = new Float32Array(hiddenSize2);
    const h2Act = new Float32Array(hiddenSize2);
    for (let i = 0; i < hiddenSize2; i++) {
      let sum = this.b2[i];
      const rowOffset = i * hiddenSize1;
      for (let j = 0; j < hiddenSize1; j++) {
        sum += this.w2[rowOffset + j] * h1Act[j];
      }
      h2Raw[i] = sum;
      h2Act[i] = this.activate(sum);
    }

    // Layer 3 (Output logits)
    const outRaw = new Float32Array(outputSize);
    for (let i = 0; i < outputSize; i++) {
      let sum = this.b3[i];
      const rowOffset = i * hiddenSize2;
      for (let j = 0; j < hiddenSize2; j++) {
        sum += this.w3[rowOffset + j] * h2Act[j];
      }
      outRaw[i] = sum;
    }

    // Softmax with numerical stability (subtract max)
    let maxLogit = -Infinity;
    for (let i = 0; i < outputSize; i++) {
      if (outRaw[i] > maxLogit) maxLogit = outRaw[i];
    }

    const probabilities = new Float32Array(outputSize);
    let sumExp = 0;
    for (let i = 0; i < outputSize; i++) {
      const expVal = Math.exp(outRaw[i] - maxLogit);
      probabilities[i] = expVal;
      sumExp += expVal;
    }

    for (let i = 0; i < outputSize; i++) {
      probabilities[i] /= sumExp;
    }

    return { h1Raw, h1Act, h2Raw, h2Act, outRaw, probabilities };
  }

  /**
   * Predict single digit with confidence
   */
  predict(x: Float32Array): PredictionResult {
    const { probabilities, h1Act } = this.forward(x);
    let bestDigit = 0;
    let maxProb = -1;

    const probList: number[] = [];
    for (let i = 0; i < probabilities.length; i++) {
      probList.push(probabilities[i]);
      if (probabilities[i] > maxProb) {
        maxProb = probabilities[i];
        bestDigit = i;
      }
    }

    return {
      digit: bestDigit,
      probabilities: probList,
      confidence: maxProb,
      rawActivations: Array.from(h1Act),
    };
  }

  /**
   * Train for one epoch over given dataset with mini-batches & momentum
   */
  trainEpoch(dataset: TrainingSample[]): { loss: number; accuracy: number } {
    const { inputSize, hiddenSize1, hiddenSize2, outputSize, learningRate, momentum, batchSize } = this.config;

    // Gradient accumulators for mini-batch
    const gw1 = new Float32Array(this.w1.length);
    const gb1 = new Float32Array(this.b1.length);
    const gw2 = new Float32Array(this.w2.length);
    const gb2 = new Float32Array(this.b2.length);
    const gw3 = new Float32Array(this.w3.length);
    const gb3 = new Float32Array(this.b3.length);

    let totalLoss = 0;
    let correctCount = 0;

    // Shuffle samples for this epoch
    const indices = Array.from({ length: dataset.length }, (_, i) => i);
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }

    let batchCount = 0;

    for (let b = 0; b < dataset.length; b += batchSize) {
      gw1.fill(0);
      gb1.fill(0);
      gw2.fill(0);
      gb2.fill(0);
      gw3.fill(0);
      gb3.fill(0);

      const actualBatchSize = Math.min(batchSize, dataset.length - b);

      for (let s = 0; s < actualBatchSize; s++) {
        const sample = dataset[indices[b + s]];
        const x = sample.pixels;
        const target = sample.label;

        // Forward
        const { h1Raw, h1Act, h2Raw, h2Act, probabilities } = this.forward(x);

        // Loss (Cross-Entropy: -log(p_target))
        const pTarget = Math.max(1e-15, probabilities[target]);
        totalLoss += -Math.log(pTarget);

        // Prediction accuracy
        let predDigit = 0;
        let highestProb = -1;
        for (let i = 0; i < 10; i++) {
          if (probabilities[i] > highestProb) {
            highestProb = probabilities[i];
            predDigit = i;
          }
        }
        if (predDigit === target) correctCount++;

        // Backpropagation
        // Delta for output layer: (p - y_one_hot)
        const dOut = new Float32Array(outputSize);
        for (let i = 0; i < outputSize; i++) {
          dOut[i] = probabilities[i] - (i === target ? 1 : 0);
        }

        // Gradients for Layer 3 (w3, b3)
        for (let i = 0; i < outputSize; i++) {
          const d = dOut[i];
          gb3[i] += d;
          const rowOffset = i * hiddenSize2;
          for (let j = 0; j < hiddenSize2; j++) {
            gw3[rowOffset + j] += d * h2Act[j];
          }
        }

        // Delta for Hidden Layer 2: (W3^T * dOut) * activateDeriv(h2Raw)
        const dH2 = new Float32Array(hiddenSize2);
        for (let j = 0; j < hiddenSize2; j++) {
          let sum = 0;
          for (let i = 0; i < outputSize; i++) {
            sum += this.w3[i * hiddenSize2 + j] * dOut[i];
          }
          dH2[j] = sum * this.activateDeriv(h2Raw[j]);
        }

        // Gradients for Layer 2 (w2, b2)
        for (let i = 0; i < hiddenSize2; i++) {
          const d = dH2[i];
          gb2[i] += d;
          const rowOffset = i * hiddenSize1;
          for (let j = 0; j < hiddenSize1; j++) {
            gw2[rowOffset + j] += d * h1Act[j];
          }
        }

        // Delta for Hidden Layer 1: (W2^T * dH2) * activateDeriv(h1Raw)
        const dH1 = new Float32Array(hiddenSize1);
        for (let j = 0; j < hiddenSize1; j++) {
          let sum = 0;
          for (let i = 0; i < hiddenSize2; i++) {
            sum += this.w2[i * hiddenSize1 + j] * dH2[i];
          }
          dH1[j] = sum * this.activateDeriv(h1Raw[j]);
        }

        // Gradients for Layer 1 (w1, b1)
        for (let i = 0; i < hiddenSize1; i++) {
          const d = dH1[i];
          gb1[i] += d;
          const rowOffset = i * inputSize;
          for (let j = 0; j < inputSize; j++) {
            gw1[rowOffset + j] += d * x[j];
          }
        }
      }

      // Update weights with Momentum SGD
      const lrNorm = learningRate / actualBatchSize;

      const updateLayer = (
        w: Float32Array,
        b: Float32Array,
        vw: Float32Array,
        vb: Float32Array,
        gw: Float32Array,
        gb: Float32Array
      ) => {
        for (let i = 0; i < w.length; i++) {
          vw[i] = momentum * vw[i] + lrNorm * gw[i];
          w[i] -= vw[i];
        }
        for (let i = 0; i < b.length; i++) {
          vb[i] = momentum * vb[i] + lrNorm * gb[i];
          b[i] -= vb[i];
        }
      };

      updateLayer(this.w1, this.b1, this.vw1, this.vb1, gw1, gb1);
      updateLayer(this.w2, this.b2, this.vw2, this.vb2, gw2, gb2);
      updateLayer(this.w3, this.b3, this.vw3, this.vb3, gw3, gb3);

      batchCount++;
    }

    this.isTrained = true;
    return {
      loss: totalLoss / dataset.length,
      accuracy: correctCount / dataset.length,
    };
  }

  /**
   * Evaluate against a test dataset and produce confusion matrix [actual][predicted]
   */
  evaluate(testSet: TrainingSample[]): {
    accuracy: number;
    loss: number;
    confusionMatrix: number[][]; // 10 x 10
  } {
    const cm: number[][] = Array.from({ length: 10 }, () => Array(10).fill(0));
    let correct = 0;
    let totalLoss = 0;

    for (const sample of testSet) {
      const { probabilities } = this.forward(sample.pixels);
      let pred = 0;
      let maxP = -1;
      for (let i = 0; i < 10; i++) {
        if (probabilities[i] > maxP) {
          maxP = probabilities[i];
          pred = i;
        }
      }

      cm[sample.label][pred]++;
      if (pred === sample.label) correct++;
      totalLoss += -Math.log(Math.max(1e-15, probabilities[sample.label]));
    }

    return {
      accuracy: correct / Math.max(1, testSet.length),
      loss: totalLoss / Math.max(1, testSet.length),
      confusionMatrix: cm,
    };
  }

  /**
   * Export weights for state preservation
   */
  exportWeights(): string {
    return JSON.stringify({
      w1: Array.from(this.w1),
      b1: Array.from(this.b1),
      w2: Array.from(this.w2),
      b2: Array.from(this.b2),
      w3: Array.from(this.w3),
      b3: Array.from(this.b3),
    });
  }

  /**
   * Import weights
   */
  importWeights(jsonStr: string) {
    try {
      const data = JSON.parse(jsonStr);
      this.w1.set(data.w1);
      this.b1.set(data.b1);
      this.w2.set(data.w2);
      this.b2.set(data.b2);
      this.w3.set(data.w3);
      this.b3.set(data.b3);
      this.isTrained = true;
    } catch (e) {
      console.error('Failed to import weights', e);
    }
  }
}
