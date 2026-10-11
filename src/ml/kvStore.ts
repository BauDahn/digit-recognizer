/**
 * A tiny key-value store on IndexedDB (localStorage is too small for network weights).
 * Falls back to memory when IndexedDB is not available (private windows, tests).
 */
const DB_NAME = 'neuroscribe';
const STORE = 'kv';

const memory = new Map<string, unknown>();
let dbPromise: Promise<IDBDatabase | null> | null = null;

function openDb(): Promise<IDBDatabase | null> {
  if (!dbPromise) {
    dbPromise = new Promise((resolve) => {
      if (typeof indexedDB === 'undefined') return resolve(null);
      try {
        const request = indexedDB.open(DB_NAME, 1);
        request.onupgradeneeded = () => request.result.createObjectStore(STORE);
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => resolve(null);
        request.onblocked = () => resolve(null);
      } catch {
        resolve(null);
      }
    });
  }
  return dbPromise;
}

function run<T>(db: IDBDatabase, mode: IDBTransactionMode, action: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    const request = action(db.transaction(STORE, mode).objectStore(STORE));
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export async function kvGet<T>(key: string): Promise<T | undefined> {
  const db = await openDb();
  if (!db) return memory.get(key) as T | undefined;
  try {
    return (await run(db, 'readonly', (s) => s.get(key))) as T | undefined;
  } catch {
    return memory.get(key) as T | undefined;
  }
}

export async function kvSet(key: string, value: unknown): Promise<boolean> {
  const db = await openDb();
  if (!db) {
    memory.set(key, value);
    return false;
  }
  try {
    await run(db, 'readwrite', (s) => s.put(value, key));
    return true;
  } catch {
    memory.set(key, value);
    return false;
  }
}

export async function kvDelete(key: string): Promise<void> {
  memory.delete(key);
  const db = await openDb();
  if (!db) return;
  try {
    await run(db, 'readwrite', (s) => s.delete(key));
  } catch {
    // ignore
  }
}
