export interface Point {
  x: number;
  y: number;
  pressure?: number;
  time?: number;
}

export interface Stroke {
  id: string;
  points: Point[];
  color: string;
  width: number;
  tool: 'pen' | 'highlighter' | 'eraser';
  /** Set on the handwritten answers the app adds after '=': the id of the line they answer. */
  answerFor?: string;
}

export interface TrainingSample {
  id: string;
  label: number;
  pixels: Float32Array; // 784 normalized values (0 to 1)
  isCustom?: boolean;
}

export interface NetworkConfig {
  inputSize: number; // 784 (28x28)
  hiddenSize1: number; // 32, 64, or 128
  hiddenSize2: number; // 16, 32
  outputSize: number; // number of classes: 10 digits, 47 EMNIST, ...
  learningRate: number; // e.g. 0.05
  momentum: number; // e.g. 0.9
  batchSize: number; // e.g. 16
  epochs: number; // e.g. 15
}

export interface TrainingEpochMetric {
  epoch: number;
  loss: number; // training loss
  accuracy: number; // test (validation) accuracy
  trainAccuracy?: number;
  seconds?: number;
}

export interface PredictionResult {
  digit: number;
  probabilities: number[];
  confidence: number;
  rawActivations?: number[];
}

export interface AlphanumericFeatureBreakdown {
  loopCount: number; // Euler holes
  endpointCount: number;
  crossIntersectionCount: number;
  aspectRatio: number;
  horizontalSymmetry: number;
  verticalSymmetry: number;
  densityTop: number;
  densityBottom: number;
  densityLeft: number;
  densityRight: number;
}

export interface SegmentedCharacter {
  id: string;
  box: { x: number; y: number; width: number; height: number };
  pixels: Float32Array; // 28x28 normalized
  features: AlphanumericFeatureBreakdown;
  predictedChar: string; // heuristic (feature rules) prediction
  candidates: Array<{ char: string; confidence: number }>;
  nnPixels?: Float32Array; // MNIST-style 28x28 input fed to the network
  nnChar?: string; // neural network prediction
  nnCandidates?: Array<{ char: string; confidence: number }>;
}

export interface MathCanvasCard {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  type: 'math-solution' | 'graph-plot' | 'image';
  /** Set on cards the app adds as the result of a line: the id of that line */
  answerFor?: string;
  title?: string;
  latex?: string;
  solution?: string;
  autoAnswer?: string;
  steps?: string[];
  graphExpr?: string;
  /** type 'image': a pasted exercise, as a data URL; it is drawn under the ink */
  src?: string;
}
