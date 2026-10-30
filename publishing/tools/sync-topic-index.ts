import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import matter from 'gray-matter';
import { loadManifest } from '../adapter/manifest.js';
import { readerTitle } from '../adapter/reading-paths.js';
import { parseMarkdownHeadings } from '../adapter/build.js';
import { referenceTerms } from '../adapter/reference-terms.js';

const root = resolve(import.meta.dirname, '../..');
const manifest = loadManifest(resolve(root, 'publishing/navigation.yml'));
const textbook = manifest.sections.find(section => section.id === 'textbook')!;
const englishContents = readFileSync(resolve(root, 'en/00 Textbook/_index.md'), 'utf8');
const englishLabels = new Map([...englishContents.matchAll(/\[\[([^\]|]+)\|([^\]]+)\]\]/g)]
  .map(match => [match[1].replace(/^02 Areas\/ML & DL\//, '') + '.md', match[2]]));
const groups: Array<[string, string, string[]]> = [
  ['Математика и основы обучения', 'Mathematics and learning foundations', ['textbook/ml-foundations/']],
  ['Классическое машинное обучение', 'Classical machine learning', ['textbook/classical-ml/']],
  ['Нейронные сети и генеративные модели', 'Neural networks and generative models', ['textbook/neural-networks/']],
  ['Представление текста и языковое моделирование', 'Text representation and language modeling', ['textbook/text-representation/', 'textbook/language-modeling/', 'textbook/recurrent-networks/']],
  ['Transformer и первые языковые модели', 'The Transformer and early language models', ['textbook/transformer/', 'textbook/models/']],
  ['Современная LLM, внимание и MoE', 'Modern LLMs, attention, and MoE', ['textbook/modern-llm/', 'textbook/moe/']],
  ['Вычислительные основы ML-систем', 'Computational foundations of ML systems', ['textbook/ml-systems/']],
  ['Данные, предобучение и распределённое обучение', 'Data, pre-training, and distributed training', ['textbook/pretraining/', 'textbook/pre-training/', 'textbook/distributed-training/']],
  ['Post-training и рассуждения', 'Post-training and reasoning', ['textbook/post-training/', 'textbook/reasoning/']],
  ['Инференс и обслуживание запросов', 'Inference and serving', ['textbook/inference/']],
  ['Поиск, embeddings и RAG', 'Retrieval, embeddings, and RAG', ['textbook/retrieval/', 'textbook/rag/']],
  ['Мультимодальные модели', 'Multimodal models', ['textbook/multimodal/']],
  ['Инструменты и агенты', 'Tools and agents', ['textbook/agents/']],
  ['Оценивание и эксплуатация', 'Evaluation and operations', ['textbook/evaluation/', 'textbook/mlops/', 'textbook/deployment/']],
  ['Рекомендательные системы', 'Recommender systems', ['textbook/recommender-systems/']],
];
const pages = textbook.pages.filter(page => !['textbook/index', 'textbook/how-to-use'].includes(page.route));
const covered = new Set<string>();
function render(english: boolean) {
  const title = english ? 'Topic reference' : 'Справочник';
  const lines = [
    '---', 'title: ' + title, 'type: concept', 'status: canonical',
    'last_updated: 2026-10-30',
    ...(english ? ['locale: en', 'translation_of: "01 Справочник/_index.md"'] : []),
    '---', '', '# ' + title, '',
    ...(english ? [
      'Find a subject below to open its full explanation, including derivations, examples, figures, and sources.',
      'These are the same chapters used by the [[00 Учебник/_index|textbook]], arranged by subject rather than as a reading sequence.',
      'EN marks a reviewed English edition; EN original marks a chapter written in English. Other links open the canonical-language text.', '',
      '## Models and comparisons', '',
      '- [[02 Атлас моделей/_index|Model families and releases]]',
      '- [[02 Атлас моделей/Сравнения/Сравнение семейств|Family comparisons]]',
      '- [[02 Атлас моделей/Timeline|Timeline of models and methods]]',
      '- [[03 Исследовательские линии/_index|Research comparisons and overviews]]', '',
      '## Topics', '',
    ] : [
      'Найдите тему ниже и откройте полное объяснение с выводами, примерами, рисунками и источниками.',
      'Это те же главы, что читаются последовательно в [[00 Учебник/_index|учебнике]];',
      'здесь они сгруппированы по предмету, а внутри группы расположены по названию.', '',
      '## Модели и сравнения', '',
      '- [[02 Атлас моделей/_index|Семейства и релизы моделей]]',
      '- [[02 Атлас моделей/Сравнения/Сравнение семейств|Сравнение семейств]]',
      '- [[02 Атлас моделей/Timeline|Хронология моделей и технологий]]',
      '- [[03 Исследовательские линии/_index|Сравнительные обзоры направлений]]', '',
      '## Темы', '',
    ]),
  ];
  for (const [ru, en, prefixes] of groups) {
    const members = pages.filter(page => prefixes.some(prefix => page.route.startsWith(prefix)));
    if (!members.length) continue;
    lines.push('### ' + (english ? en : ru), '');
    const entries = members.map(page => {
      covered.add(page.route);
      const metadata = matter(readFileSync(resolve(root, page.source), 'utf8')).data;
      const label = english ? englishLabels.get(page.source) ?? readerTitle(page.source, metadata.title)
        : readerTitle(page.source, metadata.title);
      const anchors = parseMarkdownHeadings(matter(readFileSync(resolve(root, page.source), 'utf8')).content, true);
      const terms = referenceTerms(page.source, metadata.reference_terms, anchors);
      return { page, label, terms, nativeEnglish: metadata.source_language === 'en' };
    }).sort((a, b) => a.label.localeCompare(b.label, english ? 'en' : 'ru', { numeric: true }));
    for (const { page, label, terms, nativeEnglish } of entries) {
      lines.push('- [[' + page.source.replace(/\.md$/, '') + '|' + label + ']]'
        + (english && page.sourceEn ? ' · **EN**' : english && nativeEnglish ? ' · **EN original**' : ''));
      for (const term of terms)
        lines.push('  - [[' + page.source.replace(/\.md$/, '') + '#' + term.anchor + '|'
          + (english ? term.label_en : term.label) + ']]');
    }
    lines.push('');
  }
  const standalone = manifest.sections.flatMap(section => section.pages)
    .filter(page => page.route.startsWith('reference/') && page.route !== 'reference/index')
    .map(page => ({ page, meta: matter(readFileSync(resolve(root, page.source), 'utf8')).data }))
    .filter(({ meta }) => !['redirect', 'legacy'].includes(meta.status));
  if (standalone.length) {
    lines.push(english ? '## Cross-topic reference notes' : '## Межтематические справки', '');
    for (const { page, meta } of standalone)
      lines.push('- [[' + page.source.replace(/\.md$/, '') + '|' + meta.title + ']]');
    lines.push('');
  }
  lines.push(...(english ? [
    '## Implementation studies', '',
    '- [[05 Источники/Implementation studies/Inside vLLM|Inside vLLM]]',
    '- [[05 Источники/Implementation studies/SGLang and RadixAttention|SGLang and RadixAttention]]',
    '- [[05 Источники/Implementation studies/Triton GPU kernels|Triton GPU kernels]]', '',
    '## Original material and practice', '',
    '- [[05 Источники/Курсы|Course library: original lectures, slides, and assignments]]',
    '- [[06 Практика/_index|Practice and interactive examples]]',
    '- [[04 Вопросы/_index|Questions linked to explanations]]', '',
  ] : [
    '## Разборы реализаций', '',
    '- [[05 Источники/Implementation studies/Inside vLLM|Inside vLLM]]',
    '- [[05 Источники/Implementation studies/SGLang and RadixAttention|SGLang и RadixAttention]]',
    '- [[05 Источники/Implementation studies/Triton GPU kernels|Triton GPU kernels]]', '',
    '## Оригиналы и практика', '',
    '- [[05 Источники/Курсы|Библиотека курсов: оригинальные лекции, слайды и задания]]',
    '- [[06 Практика/_index|Практика и интерактивные примеры]]',
    '- [[04 Вопросы/_index|Вопросы со ссылками на объяснения]]', '',
  ]));
  return lines.join('\n');
}
const outputs = [
  ['01 Справочник/_index.md', render(false)],
  ['en/01 Reference/_index.md', render(true)],
] as const;
const missing = pages.filter(page => !covered.has(page.route));
if (missing.length) throw new Error('Unclassified textbook topics: ' + missing.map(page => page.route).join(', '));
for (const [path, text] of outputs) {
  const absolute = resolve(root, path);
  if (process.argv.includes('--check')) {
    const stored = readFileSync(absolute, 'utf8');
    if (stored !== text) {
      const oldLines = stored.split('\n'), expected = text.split('\n');
      const at = Math.min(oldLines.length, expected.length,
        expected.findIndex((line, i) => line !== oldLines[i]) >>> 0);
      throw new Error('Topic index is out of date: ' + path + ':' + (at + 1)
        + '\nStored: ' + JSON.stringify(oldLines.slice(at, at + 3))
        + '\nExpected: ' + JSON.stringify(expected.slice(at, at + 3)));
    }
  } else {
    mkdirSync(dirname(absolute), { recursive: true });
    writeFileSync(absolute, text);
  }
}
console.log(JSON.stringify({ mode: process.argv.includes('--check') ? 'check' : 'write', topics: covered.size, indexes: outputs.map(([path]) => path) }));
