/**
 * Builds the compact datasets served from public/datasets/:
 *   - mnist-train / mnist-test             (balanced subset of MNIST)
 *   - emnist-balanced-train / -test        (balanced subset of EMNIST Balanced, 47 classes)
 *
 * Usage: npm run build:datasets
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { gzipSync } from 'node:zlib';
import { DIGIT_CLASSES, EMNIST_BALANCED_CLASSES } from '../src/ml/classes';
import { encodeDataset } from '../src/ml/datasets';
import { IdxDataset, loadEmnistBalanced, loadMnist, stratifiedSubset } from './lib/idx';

const OUT_DIR = path.resolve(import.meta.dirname, '../public/datasets');

function write(name: string, ds: IdxDataset) {
  const gz = gzipSync(encodeDataset(ds.labels, ds.images), { level: 9 });
  writeFileSync(path.join(OUT_DIR, `${name}.bin.gz`), gz);
  console.log(`${name}: ${ds.labels.length} samples, ${(gz.length / 1024).toFixed(0)} KB`);
}

mkdirSync(OUT_DIR, { recursive: true });

write('mnist-train', stratifiedSubset(await loadMnist('train'), 800, DIGIT_CLASSES.length));
write('mnist-test', stratifiedSubset(await loadMnist('test'), 200, DIGIT_CLASSES.length));
write('emnist-balanced-train', stratifiedSubset(await loadEmnistBalanced('train'), 300, EMNIST_BALANCED_CLASSES.length));
write('emnist-balanced-test', stratifiedSubset(await loadEmnistBalanced('test'), 60, EMNIST_BALANCED_CLASSES.length));
