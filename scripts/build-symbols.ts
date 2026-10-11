/**
 * Builds the dataset of the math network (every class of src/ml/symbols.ts):
 *
 *   .cache/symbols/train.bin, test.bin     everything we have (used by `npm run train:models math`)
 *   public/datasets/symbols-*.bin.gz       small balanced subsets for the browser
 *                                          (symbols-train / symbols-test for the lab,
 *                                           symbols-replay for the "teach the network" fine-tuning)
 *
 * Needs MNIST + EMNIST (downloaded automatically, see scripts/lib/idx.ts) and HASYv2:
 *   mkdir -p .cache/hasy && cd .cache/hasy
 *   curl -L -o HASYv2.tar.bz2 https://zenodo.org/records/259444/files/HASYv2.tar.bz2 && tar xjf HASYv2.tar.bz2
 *
 * Usage: npm run build:symbols
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { gzipSync } from 'node:zlib';
import { centerInk28, IMG_PIXELS, seededRandom } from '../src/ml/imageOps';
import { encodeDataset } from '../src/ml/datasets';
import { MATH_CLASSES } from '../src/ml/symbols';
import { generateOperatorImage } from '../src/ml/syntheticSymbols';
import { decodePng } from './lib/png';
import { loadEmnist, loadMnist } from './lib/idx';
import { DRAWN_SOURCES, emnistByClassToChar, HASY_SOURCES } from './lib/symbolSources';

const ROOT = path.resolve(import.meta.dirname, '..');
const HASY_DIR = path.join(ROOT, '.cache/hasy');
const OUT_CACHE = path.join(ROOT, '.cache/symbols');
const OUT_PUBLIC = path.join(ROOT, 'public/datasets');

const CAP = {
  mnist: { train: 4500, test: 400 },
  emnist: { train: 1400, test: 140 }, // per class (merged cases share it)
  hasy: { train: 1600, test: 160 },
  drawn: { train: 1800, test: 200 },
};

interface Sample {
  label: number;
  pixels: Uint8Array;
}

const rand = seededRandom(2024);
const classIndex = new Map(MATH_CLASSES.map((c, i) => [c, i]));
const labelOf = (char: string) => {
  const i = classIndex.get(char);
  if (i === undefined) throw new Error(`Unknown class ${char}`);
  return i;
};

const toBytes = (pixels: Float32Array): Uint8Array => Uint8Array.from(pixels, (v) => Math.round(Math.min(1, v) * 255));

/** HASY strokes are about half as thick as MNIST's: grow them by roughly a pixel. */
function thicken(ink: Float32Array, size: number): Float32Array {
  const out = new Float32Array(ink.length);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let m = 0;
      if (x > 0) m = Math.max(m, ink[y * size + x - 1]);
      if (x < size - 1) m = Math.max(m, ink[y * size + x + 1]);
      if (y > 0) m = Math.max(m, ink[(y - 1) * size + x]);
      if (y < size - 1) m = Math.max(m, ink[(y + 1) * size + x]);
      out[y * size + x] = Math.max(ink[y * size + x], 0.7 * m);
    }
  }
  return out;
}

function shuffle<T>(items: T[]): T[] {
  for (let i = items.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [items[i], items[j]] = [items[j], items[i]];
  }
  return items;
}

function readHasyFold(file: string): Map<string, string[]> {
  const byLatex = new Map<string, string[]>();
  for (const line of readFileSync(path.join(HASY_DIR, 'classification-task/fold-1', file), 'utf8').split('\n').slice(1)) {
    if (!line.trim()) continue;
    const [imgPath, , latex] = line.split(',');
    // Some LaTeX names contain commas inside quotes; none of the ones we use do.
    if (!byLatex.has(latex)) byLatex.set(latex, []);
    byLatex.get(latex)!.push(path.resolve(HASY_DIR, 'classification-task/fold-1', imgPath));
  }
  return byLatex;
}

function hasySamples(split: 'train' | 'test'): Sample[] {
  const files = readHasyFold(split === 'train' ? 'train.csv' : 'test.csv');
  const sources: Record<string, string[]> = { ...HASY_SOURCES };
  for (let d = 0; d <= 9; d++) sources[String(d)] = [String(d)];

  const out: Sample[] = [];
  for (const [char, names] of Object.entries(sources)) {
    const paths = shuffle(names.flatMap((n) => files.get(n) ?? [])).slice(0, CAP.hasy[split]);
    if (paths.length === 0) console.warn(`  no HASY images for ${char} (${names.join(', ')})`);
    for (const p of paths) {
      const img = decodePng(readFileSync(p));
      const ink = thicken(Float32Array.from(img.gray, (g) => 1 - g / 255), img.width);
      out.push({ label: labelOf(char), pixels: toBytes(centerInk28(ink, img.width, img.height).pixels) });
    }
  }
  return out;
}

async function mnistSamples(split: 'train' | 'test'): Promise<Sample[]> {
  const ds = await loadMnist(split);
  const counts = new Array(10).fill(0);
  const out: Sample[] = [];
  for (let i = 0; i < ds.labels.length; i++) {
    const d = ds.labels[i];
    if (counts[d]++ >= CAP.mnist[split]) continue;
    out.push({ label: labelOf(String(d)), pixels: ds.images.slice(i * IMG_PIXELS, (i + 1) * IMG_PIXELS) });
  }
  return out;
}

async function emnistLetterSamples(split: 'train' | 'test'): Promise<Sample[]> {
  const ds = await loadEmnist('byclass', split);
  // Classes made of two cases split the budget between them
  const perLabel = new Map<number, number>();
  const labelsOfChar = new Map<string, number[]>();
  for (let label = 10; label < 62; label++) {
    const char = emnistByClassToChar(label);
    if (!char) continue;
    labelsOfChar.set(char, [...(labelsOfChar.get(char) ?? []), label]);
  }
  for (const labels of labelsOfChar.values()) {
    for (const l of labels) perLabel.set(l, Math.ceil(CAP.emnist[split] / labels.length));
  }

  const out: Sample[] = [];
  const counts = new Map<number, number>();
  for (let i = 0; i < ds.labels.length; i++) {
    const l = ds.labels[i];
    const budget = perLabel.get(l);
    if (!budget) continue;
    const n = counts.get(l) ?? 0;
    if (n >= budget) continue;
    counts.set(l, n + 1);
    out.push({
      label: labelOf(emnistByClassToChar(l)!),
      pixels: ds.images.slice(i * IMG_PIXELS, (i + 1) * IMG_PIXELS),
    });
  }
  return out;
}

function drawnSamples(split: 'train' | 'test'): Sample[] {
  const out: Sample[] = [];
  for (const [char, op] of Object.entries(DRAWN_SOURCES)) {
    for (let i = 0; i < CAP.drawn[split]; i++) {
      out.push({ label: labelOf(char), pixels: toBytes(generateOperatorImage(op, rand)) });
    }
  }
  return out;
}

function pack(samples: Sample[]) {
  const labels = new Uint8Array(samples.length);
  const images = new Uint8Array(samples.length * IMG_PIXELS);
  samples.forEach((s, i) => {
    labels[i] = s.label;
    images.set(s.pixels, i * IMG_PIXELS);
  });
  return { labels, images };
}

/** The first `perClass` samples of every class, after shuffling (keeps the subsets balanced). */
function balancedSubset(samples: Sample[], perClass: number, skipPerClass = 0): Sample[] {
  const byClass = new Map<number, Sample[]>();
  for (const s of samples) {
    if (!byClass.has(s.label)) byClass.set(s.label, []);
    byClass.get(s.label)!.push(s);
  }
  return [...byClass.values()].flatMap((list) => list.slice(skipPerClass, skipPerClass + perClass));
}

function writeGz(name: string, samples: Sample[]) {
  const { labels, images } = pack(samples);
  const gz = gzipSync(encodeDataset(labels, images), { level: 9 });
  writeFileSync(path.join(OUT_PUBLIC, `${name}.bin.gz`), gz);
  console.log(`  public/datasets/${name}.bin.gz  ${samples.length} samples, ${(gz.length / 1024).toFixed(0)} KB`);
}

if (!existsSync(path.join(HASY_DIR, 'classification-task'))) {
  throw new Error('HASYv2 not found in .cache/hasy, see the instructions at the top of this file.');
}
mkdirSync(OUT_CACHE, { recursive: true });
mkdirSync(OUT_PUBLIC, { recursive: true });
console.log(`${MATH_CLASSES.length} classes`);

const sets: Record<'train' | 'test', Sample[]> = { train: [], test: [] };
for (const split of ['train', 'test'] as const) {
  console.log(`[${split}] MNIST ...`);
  const mnist = await mnistSamples(split);
  console.log(`[${split}] EMNIST ByClass ...`);
  const letters = await emnistLetterSamples(split);
  console.log(`[${split}] HASY ...`);
  const hasy = hasySamples(split);
  console.log(`[${split}] drawn symbols ...`);
  const drawn = drawnSamples(split);
  sets[split] = shuffle([...mnist, ...letters, ...hasy, ...drawn]);
  console.log(`[${split}] ${sets[split].length} samples (mnist ${mnist.length}, emnist ${letters.length}, hasy ${hasy.length}, drawn ${drawn.length})`);

  const { labels, images } = pack(sets[split]);
  writeFileSync(path.join(OUT_CACHE, `${split}.bin`), encodeDataset(labels, images));
}

const counts = new Array(MATH_CLASSES.length).fill(0);
for (const s of sets.train) counts[s.label]++;
console.log(
  'train samples per class:\n' +
    MATH_CLASSES.map((c, i) => `${c}:${counts[i]}`).join('  ')
);

console.log('Browser subsets:');
writeGz('symbols-train', balancedSubset(sets.train, 120));
writeGz('symbols-test', balancedSubset(sets.test, 40));
writeGz('symbols-replay', balancedSubset(sets.train, 24, 120));
