import { readFile, writeFile } from 'node:fs/promises';
import { basename } from 'node:path';

const [inputPath, rootId, locale, outputPath, indexPath, imageListPath] = process.argv.slice(2);
if (!inputPath || !rootId || !locale || !outputPath || !indexPath || !imageListPath) {
  throw new Error(
    'usage: notion-questions-to-markdown.mjs <tree.json> <root-id> <ru|en> '
    + '<answers.md> <index.md> <images.tsv>'
  );
}

const sourceUrls = {
  ru: 'https://dynamic-epoch-4bb.notion.site/100-questions-about-NLP-549ccde0d81a4689b5635888b9d0d7e6',
  en: 'https://dynamic-epoch-4bb.notion.site/100-questions-NLP-english-337ac246920c4afd9c54af825f5076f1'
};
const sourceUrl = sourceUrls[locale];
if (!sourceUrl) throw new Error(`unsupported locale: ${locale}`);

const tree = JSON.parse(await readFile(inputPath, 'utf8'));
const wrappedBlocks = tree.recordMap?.block ?? {};
const blocks = new Map(Object.entries(wrappedBlocks).map(([id, wrapper]) => {
  const outer = wrapper.value;
  return [id, outer?.value ?? outer];
}));
const root = blocks.get(rootId);
if (!root) throw new Error(`root block not found: ${rootId}`);

function plain(property = []) {
  return property.map((segment) => segment?.[0] ?? '').join('');
}

function escapeMarkdown(text) {
  return text.replaceAll('\\', '\\\\').replace(/([*_`[\]])/g, '\\$1');
}

function latexSafe(text) {
  return text
    .replaceAll('стандартно - первой-категории', '\\text{standard | first batch}')
    .replaceAll('стандартное', '\\text{standard}')
    .replaceAll('первой- категории', '\\text{first batch}')
    .replace(/[А-Яа-яЁё][А-Яа-яЁё\s-]*/g, (match) => `\\text{${match.trim()}}`);
}

function rich(property = []) {
  return property.map((segment) => {
    const raw = segment?.[0] ?? '';
    const annotations = segment?.[1] ?? [];
    const equation = annotations.find(([kind]) => kind === 'e');
    let value = equation ? `$${latexSafe(equation[1])}$` : escapeMarkdown(raw);
    for (const [kind, argument] of annotations) {
      if (kind === 'a') {
        const href = argument.startsWith('/')
          ? new URL(argument, 'https://dynamic-epoch-4bb.notion.site').href
          : argument;
        value = `[${value}](${href})`;
      }
      else if (kind === 'b') value = `**${value}**`;
      else if (kind === 'i') value = `*${value}*`;
      else if (kind === 's') value = `~~${value}~~`;
      else if (kind === 'c') value = `\`${raw.replaceAll('`', '\\`')}\``;
    }
    return value;
  }).join('');
}

function imageExtension(block) {
  const source = plain(block.properties?.source);
  const title = plain(block.properties?.title);
  const candidate = basename(new URL(source).pathname) || title;
  const extension = candidate.match(/\.(png|jpe?g|gif|webp|svg)$/i)?.[1]?.toLowerCase();
  return extension === 'jpeg' ? 'jpg' : (extension ?? 'png');
}

function imagePath(block) {
  return `Assets/Sources/100 questions about NLP/${block.id}.${imageExtension(block)}`;
}

const images = [...blocks.values()].filter((block) => block?.type === 'image');
const imageRows = images.map((block) => {
  const rawSource = plain(block.properties?.source);
  const encoded = encodeURIComponent(rawSource);
  const proxy = `https://dynamic-epoch-4bb.notion.site/image/${encoded}?table=block&id=${block.id}&cache=v2`;
  return `${block.id}\t${proxy}\t${imagePath(block)}`;
});
await writeFile(imageListPath, `${imageRows.join('\n')}\n`);

function isQuestion(text) {
  return /^\s*(?:[1-9]|[1-9]\d|100)\.\s+/.test(text);
}

function isAnswerLabel(text) {
  return /^(?:ответ|answer)\s*$/i.test(text.trim());
}

function heading(level, text) {
  return `${'#'.repeat(level)} ${text.trim()}\n\n`;
}

function renderChildren(block, depth, context) {
  return (block.content ?? []).map((id) => renderBlock(blocks.get(id), depth, context)).join('');
}

function renderList(block, depth, context, marker) {
  const text = rich(block.properties?.title).trim();
  if (block.type === 'to_do' && isQuestion(plain(block.properties?.title))) {
    const question = heading(3, escapeMarkdown(plain(block.properties?.title)));
    const children = renderChildren(block, depth + 1, { ...context, inQuestion: true });
    if (children.trim()) return question + children;
    const missing = locale === 'ru'
      ? '> [!todo] Ответ пока не добавлен в исходный конспект.\n\n'
      : '> [!todo] The source notes do not contain an answer yet.\n\n';
    return question + missing;
  }
  const indent = '  '.repeat(Math.max(0, depth));
  const line = `${indent}${marker} ${text}\n`;
  const children = renderChildren(block, depth + 1, context);
  return children ? `${line}${children}\n` : line;
}

function renderBlock(block, depth = 0, context = {}) {
  if (!block || block.alive === false) return '';
  const text = rich(block.properties?.title).trim();
  const rawText = plain(block.properties?.title).trim();
  switch (block.type) {
    case 'page':
      return block.id === rootId ? renderChildren(block, depth, context) : '';
    case 'header':
      return heading(2, text);
    case 'sub_header':
      return heading(2, text);
    case 'sub_sub_header':
      return heading(3, text);
    case 'toggle': {
      if (isAnswerLabel(rawText)) return renderChildren(block, depth, { ...context, inAnswer: true });
      if (isQuestion(rawText)) {
        const question = heading(3, escapeMarkdown(rawText));
        const children = renderChildren(block, depth + 1, { ...context, inQuestion: true });
        return question + (children.trim()
          ? children
          : (locale === 'ru'
            ? '> [!todo] Ответ пока не добавлен в исходный конспект.\n\n'
            : '> [!todo] The source notes do not contain an answer yet.\n\n'));
      }
      return heading(2, text) + renderChildren(block, depth, context);
    }
    case 'to_do':
      return renderList(block, depth, context, '-');
    case 'bulleted_list':
      return renderList(block, depth, context, '-');
    case 'numbered_list':
      return renderList(block, depth, context, '1.');
    case 'text': {
      const body = text ? `${text}\n\n` : '';
      return body + renderChildren(block, depth, context);
    }
    case 'quote':
      return `${text.split('\n').map((line) => `> ${line}`).join('\n')}\n\n`
        + renderChildren(block, depth, context);
    case 'equation':
      return `$$\n${latexSafe(plain(block.properties?.title).trim())}\n$$\n\n`;
    case 'code': {
      const language = plain(block.properties?.language).toLowerCase() || 'text';
      const code = plain(block.properties?.title).replace(/\n+$/, '');
      return `\`\`\`${language}\n${code}\n\`\`\`\n\n`;
    }
    case 'divider':
      return '---\n\n';
    case 'image': {
      const title = plain(block.properties?.title).trim() || 'Illustration';
      const label = locale === 'ru' ? 'Источник изображения' : 'Image source';
      return `![[${imagePath(block)}|${title}]]\n\n*${label}: [Notion](${sourceUrl}).*\n\n`;
    }
    case 'file': {
      const url = plain(block.properties?.source);
      const title = plain(block.properties?.title) || 'Attachment';
      return url ? `[${escapeMarkdown(title)}](${url})\n\n` : '';
    }
    case 'table':
    case 'table_row':
      return renderChildren(block, depth, context);
    case 'transclusion_container':
    case 'transclusion_reference':
      return renderChildren(block, depth, context);
    default:
      return renderChildren(block, depth, context);
  }
}

const sourceTitle = locale === 'ru'
  ? '100 вопросов по NLP — исходные ответы'
  : '100 questions about NLP — original answers';
const intro = locale === 'ru'
  ? `Это редакционный перенос [публичного конспекта в Notion](${sourceUrl}). `
    + 'Формулировки ответов, примеры, код, изображения и исходные ссылки сохранены; '
    + 'исправления и расширения учебника следует делать в канонических главах, а не скрыто переписывать этот источник.'
  : `This is an editorial import of the [public Notion notes](${sourceUrl}). `
    + 'The answers, examples, code, figures, and original links are preserved. '
    + 'Corrections and extensions belong in Bookvar’s canonical chapters rather than in a silent rewrite of this source.'
const answersFrontmatter = `---\n`
  + `title: ${sourceTitle}\n`
  + `type: source-note\n`
  + `status: active\n`
  + (locale === 'en'
    ? `locale: en\ntranslation_of: "04 Вопросы/100 вопросов по NLP — исходные ответы.md"\n`
    : '')
  + `last_updated: 2026-07-23\n`
  + `last_verified: 2026-07-23\n`
  + `source_url: "${sourceUrl}"\n`
  + `---\n\n# ${sourceTitle}\n\n${intro}\n\n`;
const answers = `${answersFrontmatter}${renderChildren(root).replace(/\n{3,}/g, '\n\n')}`;
await writeFile(outputPath, answers);

const questionBlocks = [...blocks.values()]
  .filter((block) => block?.type === 'to_do')
  .map((block) => ({
    block,
    text: plain(block.properties?.title).trim(),
    number: Number.parseInt(plain(block.properties?.title), 10)
  }))
  .filter(({ text, number }) => isQuestion(text) && number >= 1 && number <= 100)
  .filter(({ text }) => locale === 'ru' ? /[А-Яа-яЁё]/.test(text) : !/[А-Яа-яЁё]/.test(text))
  .sort((left, right) => left.number - right.number);

if (questionBlocks.length !== 100) {
  throw new Error(`expected 100 ${locale} questions, found ${questionBlocks.length}`);
}

function hasSubstantiveContent(block) {
  if (!block) return false;
  if (['image', 'equation', 'code', 'file'].includes(block.type)) return true;
  const text = plain(block.properties?.title).trim();
  if (text && !isAnswerLabel(text)) return true;
  return (block.content ?? []).some((id) => hasSubstantiveContent(blocks.get(id)));
}

function hasAnswer(block) {
  return (block.content ?? []).some((id) => hasSubstantiveContent(blocks.get(id)));
}

const answeredCount = questionBlocks.filter(({ block }) => hasAnswer(block)).length;

const topicRanges = [
  [1, 8, locale === 'ru' ? 'Классические методы и TF–IDF' : 'Classical methods and TF–IDF',
    [['00 Учебник/02 Представление текста и токенизация/01 Представление текста числами',
      locale === 'ru' ? 'Представление текста числами' : 'Representing text numerically']]],
  [9, 15, locale === 'ru' ? 'Метрики' : 'Metrics',
    [['00 Учебник/18 Evaluation и методология/59 Оценивание моделей и контаминация',
      locale === 'ru' ? 'Оценивание моделей' : 'Model evaluation']]],
  [16, 24, 'Word2Vec and embeddings',
    [['00 Учебник/02 Представление текста и токенизация/01 От слов к embeddings',
      locale === 'ru' ? 'От слов к векторам' : 'From words to embeddings']]],
  [25, 31, 'RNN and CNN',
    [['00 Учебник/04 RNN, LSTM и Seq2Seq/01 RNN и BPTT', 'RNN and BPTT'],
      ['00 Учебник/01 Основы нейронных сетей/07 CNN — от свёртки до ResNet',
        locale === 'ru' ? 'Свёрточные сети' : 'Convolutional networks']]],
  [32, 46, 'Attention and Transformer',
    [['00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V', 'Self-attention'],
      ['00 Учебник/05 Attention и Transformer/03 Полный Transformer',
        locale === 'ru' ? 'Полный Transformer' : 'The complete Transformer']]],
  [47, 53, locale === 'ru' ? 'Семейства Transformer-моделей' : 'Transformer model families',
    [['00 Учебник/06 Encoder, Decoder и Encoder-Decoder/01 Три архитектурных паттерна',
      locale === 'ru' ? 'Encoder, decoder и encoder–decoder' : 'Encoder, decoder, and encoder–decoder'],
      ['00 Учебник/06 Encoder, Decoder и Encoder-Decoder/04 GPT-1 — генеративное предобучение', 'GPT']]],
  [54, 59, locale === 'ru' ? 'Позиционная информация' : 'Positional information',
    [['00 Учебник/05 Attention и Transformer/04 Позиционная информация',
      locale === 'ru' ? 'Позиционная информация' : 'Positional information'],
      ['00 Учебник/07 Анатомия современной LLM/04 RoPE.md', 'RoPE']]],
  [60, 63, 'Pre-training',
    [['00 Учебник/11 Pre-training и Scaling/42 Next-token prediction', 'Next-token prediction'],
      ['00 Учебник/11 Pre-training и Scaling/41 Сбор, очистка и смеси данных',
        locale === 'ru' ? 'Данные для предобучения' : 'Pre-training data']]],
  [64, 72, locale === 'ru' ? 'Токенизация' : 'Tokenization',
    [['00 Учебник/02 Представление текста и токенизация/02 BPE, WordPiece и Unigram',
      locale === 'ru' ? 'BPE, WordPiece и Unigram' : 'BPE, WordPiece, and Unigram']]],
  [73, 87, locale === 'ru' ? 'Обучение и адаптация' : 'Training and adaptation',
    [['00 Учебник/01 Основы нейронных сетей/02 Оптимизация и стабильность обучения',
      locale === 'ru' ? 'Оптимизация и устойчивость обучения' : 'Optimization and training stability'],
      ['00 Учебник/12 Post-training и Alignment/01 SFT и instruction data',
        locale === 'ru' ? 'SFT и instruction data' : 'SFT and instruction data']]],
  [88, 89, locale === 'ru' ? 'Декодирование' : 'Decoding',
    [['00 Учебник/14 Inference и оптимизация/54 Декодирование и выбор следующего токена',
      locale === 'ru' ? 'Декодирование и выбор токена' : 'Decoding and next-token selection']]],
  [90, 100, 'Large language models',
    [['00 Учебник/15 Embeddings, Retrieval и RAG/63 RAG — полный конвейер', 'RAG'],
      ['00 Учебник/14 Inference и оптимизация/57 Квантизация языковых моделей',
        locale === 'ru' ? 'Квантизация' : 'Quantization'],
      ['00 Учебник/09 Dense FFN и Mixture of Experts/02 Mixture of Experts — routing, capacity и serving',
        'Mixture of Experts']]]
];

const answerSource = '04 Вопросы/100 вопросов по NLP — исходные ответы';
let indexBody = '';
for (const [start, end, title, chapters] of topicRanges) {
  indexBody += `## ${title}\n\n`;
  const textbookLabel = locale === 'ru' ? 'Главы Bookvar' : 'Bookvar chapters';
  indexBody += `**${textbookLabel}:** ${chapters.map(([target, label]) => `[[${target}|${label}]]`).join(' · ')}\n\n`;
  for (const { block, text, number } of questionBlocks.filter((item) => item.number >= start && item.number <= end)) {
    const answerLabel = locale === 'ru' ? 'исходный ответ' : 'original answer';
    const missingLabel = locale === 'ru' ? 'ответ ещё не заполнен' : 'answer not yet supplied';
    indexBody += `- **${escapeMarkdown(text)}** — `;
    indexBody += hasAnswer(block)
      ? `[[${answerSource}#${text}|${answerLabel}]]`
      : missingLabel;
    indexBody += '\n';
  }
  indexBody += '\n';
}

const indexTitle = locale === 'ru' ? '100 вопросов по NLP' : '100 questions about NLP';
const indexIntro = locale === 'ru'
  ? `Список перенесён из [публичной страницы Notion](${sourceUrl}). `
    + 'Ссылка «исходный ответ» ведёт к сохранённому авторскому конспекту. '
    + `В текущем дереве Notion содержательные ответы найдены у ${answeredCount} из 100 вопросов; `
    + 'счётчик «49/100» внутри исходной страницы, по-видимому, не обновлялся после последних дополнений. '
    + 'Связи с каноническими главами Bookvar добавляются отдельно: исходный ответ не заменяет главу учебника.'
  : `This list was imported from the [public Notion page](${sourceUrl}). `
    + '“Original answer” links to the preserved source notes. '
    + `The English source currently contains substantive answers to ${answeredCount} of the 100 questions; `
    + 'it trails the newer Russian page, so missing translations are marked explicitly. '
    + 'Links to canonical Bookvar chapters are maintained separately: a source answer is not a substitute for a textbook chapter.'
const indexFrontmatter = `---\n`
  + `title: ${indexTitle}\n`
  + `type: question-index\n`
  + `status: active\n`
  + (locale === 'en'
    ? `locale: en\ntranslation_of: "04 Вопросы/100 вопросов по NLP.md"\n`
    : '')
  + `last_updated: 2026-07-23\n`
  + `last_verified: 2026-07-23\n`
  + `---\n\n# ${indexTitle}\n\n${indexIntro}\n\n`;
await writeFile(indexPath, `${indexFrontmatter}${indexBody}`);
