export type AppMode = 'digit-lab' | 'alphanumeric' | 'infinite-canvas';

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

export interface MathRecognitionResult {
  latex: string;
  readableText: string;
  symbolsDetected: string[];
  isSolvable: boolean;
  solution?: string;
  autoAnswer?: string;
  steps?: string[];
  isGraphable?: boolean;
  graphExpr?: string; // expression in x, parsed by src/ml/mathParser (never evaluated as JS)
  graphTitle?: string;
  error?: string;
  symbols?: Array<{
    char: string;
    confidence: number;
    candidates: Array<{ char: string; confidence: number }>;
    pixels: Float32Array | null;
    isSuperscript: boolean;
  }>;
  rawBoundingBox?: { x: number; y: number; width: number; height: number };
}

export interface MathCanvasCard {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  type: 'math-solution' | 'graph-plot';
  title?: string;
  latex?: string;
  solution?: string;
  autoAnswer?: string;
  steps?: string[];
  graphExpr?: string;
}
