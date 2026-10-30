export const attentionTokens = ['Она', 'читает', 'книгу', '.'] as const;

export interface AttentionSettings {
  pastScore: number;
  futureScore: number;
  causal: boolean;
}

export function attentionWeights(settings: AttentionSettings): number[][] {
  return attentionTokens.map((_, row) => {
    const scores = [1, 2, .5, -.5];
    if (row === 1) {
      scores[0] = settings.pastScore; // «читает» attends to «Она».
      scores[2] = settings.futureScore; // «книгу» is in the future.
    }
    const allowed = scores.map((value, column) =>
      settings.causal && column > row ? -Infinity : value);
    const maximum = Math.max(...allowed);
    const exponentials = allowed.map(value => Math.exp(value - maximum));
    const total = exponentials.reduce((sum, value) => sum + value, 0);
    return exponentials.map(value => value / total);
  });
}

export function checkAttention(settings: AttentionSettings): Array<[string, boolean]> {
  const rows = attentionWeights(settings);
  const changed = attentionWeights({ ...settings, futureScore: settings.futureScore + 10 });
  return [
    ['Форма матрицы равна 4 × 4', rows.length === 4 && rows.every(row => row.length === 4)],
    ['Каждая строка суммируется в единицу', rows.every(row =>
      Math.abs(row.reduce((sum, value) => sum + value, 0) - 1) < 1e-9)],
    ['Все будущие позиции имеют нулевой вес', rows.every((row, i) =>
      row.every((value, j) => j <= i || value === 0))],
    ['Первый токен видит только себя', rows[0][0] === 1 && rows[0].slice(1).every(value => value === 0)],
    ['Изменение будущей оценки не влияет на веса прошлой позиции',
      rows[1].every((value, j) => Math.abs(value - changed[1][j]) < 1e-9)],
  ];
}
