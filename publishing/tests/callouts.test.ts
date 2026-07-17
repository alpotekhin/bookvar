import { describe, expect, it } from 'vitest';
import { convertCallouts } from '../adapter/callouts.js';

describe('convertCallouts', () => {
  it('converts an Obsidian warning callout to a titled caution directive', () => {
    const source = [
      '> [!warning] Ограничение',
      '> Этот вывод зависит от предположения.'
    ].join('\n');

    expect(convertCallouts(source)).toBe([
      ':::caution[Ограничение]',
      'Этот вывод зависит от предположения.',
      ':::'
    ].join('\n'));
    expect(source).toBe([
      '> [!warning] Ограничение',
      '> Этот вывод зависит от предположения.'
    ].join('\n'));
  });

  it.each([
    ['note', 'note'],
    ['info', 'note'],
    ['abstract', 'note'],
    ['tip', 'tip'],
    ['caution', 'caution'],
    ['danger', 'danger'],
    ['failure', 'danger']
  ])('maps %s callouts to %s directives', (sourceType, targetType) => {
    expect(convertCallouts(`> [!${sourceType}]\n> Текст`)).toBe(
      `:::${targetType}\nТекст\n:::`
    );
  });

  it('preserves Markdown outside converted callouts', () => {
    const source = 'До\n\nОбычный текст.\n\nПосле';

    expect(convertCallouts(source)).toBe(source);
  });
});
