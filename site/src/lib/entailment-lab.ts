export const stages = ['gpt1', 'gpt2', 'gpt3', 'llama'] as const;
export type Stage = typeof stages[number];

const input = 'Premise: A cat sleeps.\nHypothesis: An animal rests.\nRelation:';
const instruction = 'Choose entailment, contradiction, or neutral.';
const examples = [
  'Premise: A robin is singing.\nHypothesis: A bird is singing.\nRelation: entailment',
  'Premise: The door is open.\nHypothesis: The door is shut.\nRelation: contradiction',
];

export const classLogits = [1, 0, -1] as const;
export const labels = ['entailment', 'neutral', 'contradiction'] as const;
const classWeights = classLogits.map(Math.exp);
const classTotal = classWeights.reduce((sum, weight) => sum + weight, 0);
export const classProbabilities = classWeights.map(weight => weight / classTotal);

export function entailmentInput(stage: Stage, shots: number) {
  if (stage === 'gpt1') return '[Start] A cat sleeps [Delim] An animal rests [Extract]';
  if (stage === 'gpt2') return input;
  if (shots !== 0 && shots !== 1 && shots !== 2) throw new RangeError('Expected 0, 1 or 2 demonstrations');
  return [instruction, ...examples.slice(2 - shots), input].join('\n\n');
}

export function continuationScores(stage: Exclude<Stage, 'gpt1'>, secondMilli: number) {
  if (!Number.isInteger(secondMilli) || secondMilli < 0 || secondMilli > 1000)
    throw new RangeError('Expected a probability in thousandths');
  const factors = stage === 'llama' ? [400, secondMilli, 800] : [400, secondMilli];
  // Integer arithmetic makes the exact ties at p₂ = .625 (GPT) unambiguous.
  const scale = 1000 ** factors.length;
  const scores = [factors.reduce((product, f) => product * f, 1), scale / 4, scale / 10];
  const top = Math.max(...scores);
  return {
    factors: factors.map(f => f / 1000),
    scores: scores.map(s => s / scale),
    winners: labels.filter((_, i) => scores[i] === top),
  };
}
