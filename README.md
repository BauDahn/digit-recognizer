# NeuroScribe

An educational handwriting-recognition studio. A small neural network (a multi-layer perceptron), written from scratch in TypeScript, is trained on real **MNIST** images and runs **entirely in your browser**. No cloud APIs, no ML libraries: you can read every line of the forward pass and of backpropagation in [`src/ml/neuralNetwork.ts`](src/ml/neuralNetwork.ts).

## The three modes

| Mode | What it shows |
|---|---|
| **01. Digit Lab** | Train a 784-64-32-10 network on 8,000 MNIST digits in a Web Worker and watch the learning curves, the confusion matrix, the first-layer weights and the activation of every neuron for what you draw. Load the pre-trained model (97.9% on the full MNIST test set), export or import weights, or teach the model your own handwriting. |
| **02. Alphanumeric Studio** | Several characters at once: segmentation into separate characters, a 784-128-64-47 network trained on **EMNIST Balanced** (digits and letters), and side by side, a classifier made of hand-written rules (loops, symmetry, aspect ratio) so you can compare both approaches. You can also train the letters network from scratch in the browser. |
| **03. Infinite Math Canvas** | Write math by hand: `24+16=`, `2x+3=11`, `x²-4`. Each symbol is cut out, turned into a 28×28 MNIST-style image and classified by a network trained on MNIST digits + synthetic operators. A small parser (no `eval`) evaluates the expression step by step, solves linear and quadratic equations, writes the answer after the `=` and plots functions of x. |

## Getting started

```bash
npm install
npm run dev        # http://localhost:5173
```

Other scripts:

```bash
npm test           # unit tests (Vitest)
npm run lint       # type checking
npm run build      # production build in dist/

npm run build:datasets   # re-create public/datasets/*.bin.gz from MNIST / EMNIST
npm run train:models     # re-train public/models/*.json (digits, symbols, letters)
```

The two last scripts download the original datasets into `.cache/datasets/` (MNIST ≈ 11 MB, EMNIST ≈ 560 MB; EMNIST needs the `unzip` command). The results are committed, so you only need them to change the datasets or retrain.

## How it works

```
drawing ──► 28×28 image ──► neural network ──► probabilities ──► answer
```

1. **Preprocessing** (`src/ml/imageOps.ts`): exactly what was done to build MNIST. The ink is cropped, scaled to fit a 20×20 box keeping its proportions, and placed in a 28×28 image so that its center of mass is in the middle. Strokes from the canvas are rasterized with a fixed pen width so a symbol looks the same whether you wrote it small or large.
2. **The network** (`src/ml/neuralNetwork.ts`): two hidden layers with Leaky ReLU, a softmax output, cross-entropy loss and mini-batch SGD with momentum. Weights start with He initialization.
3. **Training** (`src/ml/trainer.worker.ts`): runs in a Web Worker so the page keeps responding. With *data augmentation* on, half of the images are randomly rotated, scaled, shifted or made bolder every epoch.
4. **Datasets** (`src/ml/datasets.ts`): compact gzip-compressed binary files in `public/datasets/`, decompressed in the browser with `DecompressionStream`.
5. **Math** (`src/ml/strokeRecognizer.ts`, `src/ml/mathParser.ts`): strokes that overlap horizontally form one symbol (`=`, `+`, `4`…), tiny marks are dots, raised and smaller symbols are exponents. The resulting text is parsed into a syntax tree that is evaluated, simplified one operation at a time or solved.

## Pre-trained models

| File | Architecture | Trained on | Test accuracy |
|---|---|---|---|
| `public/models/digits.json` | 784-64-32-10 | MNIST (60,000 images) | 97.9% |
| `public/models/symbols.json` | 784-128-64-17 | MNIST + synthetic `+ - / = ( )` + `x` from EMNIST | 97.5% |
| `public/models/letters.json` | 784-128-64-47 | EMNIST Balanced | 81.9% |

Models trained in the browser are saved in `localStorage` and can be exported as JSON in the same format.

## Project structure

```
src/
  ml/            neural network, preprocessing, datasets, math parser (no React, unit-tested)
  components/    one folder per mode
scripts/         dataset builder and offline training (Node, same code as the browser)
public/          datasets and pre-trained models served as static files
```
