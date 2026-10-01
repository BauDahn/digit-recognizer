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
  inputSize: number; // 784
  hiddenSize1: number; // 32, 64, or 128
  hiddenSize2: number; // 16, 32
  outputSize: number; // 10
  learningRate: number; // e.g. 0.05
  momentum: number; // e.g. 0.9
  batchSize: number; // e.g. 16
  epochs: number; // e.g. 15
}

export interface TrainingEpochMetric {
  epoch: number;
  loss: number;
  accuracy: number;
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
  predictedChar: string;
  candidates: Array<{ char: string; confidence: number }>;
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
  graphJsExpr?: string;
  graphTitle?: string;
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
  graphJsExpr?: string;
}
