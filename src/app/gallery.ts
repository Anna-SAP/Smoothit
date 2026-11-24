// On-device gallery: the last shots are kept in IndexedDB so they survive reloads. Nothing is
// uploaded anywhere.

export interface Shot {
  id: number;
  createdAt: number;
  look: string;
  width: number;
  height: number;
  image: Blob;
  /** The same frame and crop without any processing, for before/after comparison. */
  original: Blob | null;
  thumb: Blob;
}

const DB_NAME = 'smoothit';
const STORE = 'shots';
const MAX_SHOTS = 40;

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE, { keyPath: 'id' });
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function run<T>(mode: IDBTransactionMode, fn: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  return openDb().then(
    (db) =>
      new Promise<T>((resolve, reject) => {
        const tx = db.transaction(STORE, mode);
        const req = fn(tx.objectStore(STORE));
        tx.oncomplete = () => { db.close(); resolve(req.result); };
        tx.onerror = tx.onabort = () => { db.close(); reject(tx.error); };
      }),
  );
}

/** Newest first. Storage failures (private mode, quota) degrade to an in-memory list. */
export class Gallery {
  private memory: Shot[] = [];
  private persistent = true;

  async list(): Promise<Shot[]> {
    if (this.persistent) {
      try {
        const all = await run<Shot[]>('readonly', (s) => s.getAll());
        return all.sort((a, b) => b.id - a.id);
      } catch {
        this.persistent = false;
      }
    }
    return [...this.memory];
  }

  async add(shot: Shot): Promise<void> {
    this.memory.unshift(shot);
    this.memory.length = Math.min(this.memory.length, MAX_SHOTS);
    if (!this.persistent) return;
    try {
      await run('readwrite', (s) => s.put(shot));
      const all = await this.list();
      for (const old of all.slice(MAX_SHOTS)) await run('readwrite', (s) => s.delete(old.id));
    } catch {
      this.persistent = false;
    }
  }

  async remove(id: number): Promise<void> {
    this.memory = this.memory.filter((s) => s.id !== id);
    if (!this.persistent) return;
    try {
      await run('readwrite', (s) => s.delete(id));
    } catch {
      this.persistent = false;
    }
  }
}
