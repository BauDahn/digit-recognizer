import { NeuralNetwork, SerializedNetwork } from './neuralNetwork';

/**
 * A trained model on disk (public/models/*.json) or in localStorage:
 * the network weights plus what its outputs mean and how well it did.
 */
export interface ModelFile extends SerializedNetwork {
  name: string;
  classes: string[];
  meta?: {
    trainedOn?: string;
    testAccuracy?: number;
    epochs?: number;
    createdAt?: string;
  };
}

export interface LoadedModel {
  name: string;
  classes: string[];
  network: NeuralNetwork;
  meta: ModelFile['meta'];
}

export function toModelFile(name: string, classes: string[], network: NeuralNetwork, meta?: ModelFile['meta']): ModelFile {
  return { ...network.serialize(), name, classes, meta };
}

export function fromModelFile(file: ModelFile): LoadedModel {
  const network = NeuralNetwork.deserialize(file);
  if (file.classes.length !== network.config.outputSize) {
    throw new Error(`Model ${file.name}: ${file.classes.length} classes but ${network.config.outputSize} outputs`);
  }
  return { name: file.name, classes: file.classes, network, meta: file.meta };
}

export type BuiltinModel = 'digits' | 'letters' | 'math';

const cache = new Map<BuiltinModel, Promise<LoadedModel>>();

/** Load one of the pre-trained models shipped in public/models (browser only, cached). */
export function loadBuiltinModel(name: BuiltinModel): Promise<LoadedModel> {
  let promise = cache.get(name);
  if (!promise) {
    promise = fetch(`${import.meta.env.BASE_URL}models/${name}.json`)
      .then((res) => {
        if (!res.ok) throw new Error(`Could not load model ${name}: ${res.status}`);
        return res.json();
      })
      .then((file: ModelFile) => fromModelFile(file));
    promise.catch(() => cache.delete(name));
    cache.set(name, promise);
  }
  return promise;
}

const STORAGE_PREFIX = 'neuroscribe:model:';

/** Persist a model the user trained in the browser. Returns false if storage is unavailable/full. */
export function saveModelToStorage(key: string, file: ModelFile): boolean {
  try {
    localStorage.setItem(STORAGE_PREFIX + key, JSON.stringify(file));
    return true;
  } catch {
    return false;
  }
}

export function loadModelFromStorage(key: string): ModelFile | null {
  try {
    const raw = localStorage.getItem(STORAGE_PREFIX + key);
    return raw ? (JSON.parse(raw) as ModelFile) : null;
  } catch {
    return null;
  }
}

export function removeModelFromStorage(key: string) {
  try {
    localStorage.removeItem(STORAGE_PREFIX + key);
  } catch {
    // ignore
  }
}
