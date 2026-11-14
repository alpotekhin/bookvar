export const bpeDefaults = { training: 'aaabdaaabac', probe: 'aaabac', limit: 3 };
export const bpeMaxMerges = 12;
export const bpeMaxInputLength = 96;

type Pair = { left: number; right: number; count: number };
type Merge = Pair & { id: number; replacements: number };

const encoder = new TextEncoder();
const decoder = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true });

function adjacentPairs(ids: readonly number[]): Pair[] {
  const counts = new Map<number, Pair>();
  for (let i = 1; i < ids.length; i++) {
    const left = ids[i - 1];
    const right = ids[i];
    const key = left * (256 + bpeMaxMerges) + right;
    const existing = counts.get(key);
    if (existing) existing.count++;
    else counts.set(key, { left, right, count: 1 });
  }
  // Stable sorting keeps first appearance when several pairs have the same count.
  return [...counts.values()].sort((a, b) => b.count - a.count);
}

function applyMerge(ids: readonly number[], { left, right, id }: Merge): number[] {
  const result: number[] = [];
  for (let i = 0; i < ids.length; i++) {
    if (ids[i] === left && ids[i + 1] === right) {
      result.push(id);
      i++;
    } else {
      result.push(ids[i]);
    }
  }
  return result;
}

export function trainBpeLab(text: string, limit: number) {
  if (!Number.isInteger(limit) || limit < 0 || limit > bpeMaxMerges)
    throw new RangeError('Invalid BPE merge limit');
  const vocab = Array.from({ length: 256 }, (_, id) => [id]);
  const first = [...encoder.encode(text)];
  const frames = [{ ids: first, pairs: adjacentPairs(first) }];
  const merges: Merge[] = [];
  for (let i = 0; i < limit; i++) {
    const { ids, pairs } = frames[i];
    if (!pairs.length) break;
    const best = pairs[0];
    const id = vocab.length;
    const rule = { ...best, id, replacements: 0 };
    const after = applyMerge(ids, rule);
    rule.replacements = ids.length - after.length;
    vocab.push([...vocab[best.left], ...vocab[best.right]]);
    merges.push(rule);
    frames.push({ ids: after, pairs: adjacentPairs(after) });
  }
  return { vocab, merges, frames };
}

export function encodeBpeLab(text: string, merges: readonly Merge[]): number[] {
  let ids = [...encoder.encode(text)];
  for (const rule of merges) ids = applyMerge(ids, rule);
  return ids;
}

export function decodeBpeLab(ids: readonly number[], vocab: readonly number[][]): string {
  const bytes = ids.flatMap(id => {
    const token = vocab[id];
    if (!token) throw new RangeError('Unknown BPE token: ' + id);
    return token;
  });
  return decoder.decode(Uint8Array.from(bytes));
}

export function showBpeBytes(bytes: readonly number[]): string {
  const hex = () => bytes.map(b => b.toString(16).toUpperCase().padStart(2, '0')).join(' ');
  try {
    const text = decoder.decode(Uint8Array.from(bytes));
    if (/\p{C}/u.test(text)) return '⟦' + hex() + '⟧';
    return text.replaceAll(' ', '␠');
  } catch {
    return '⟦' + hex() + '⟧';
  }
}
