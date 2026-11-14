import { readFileSync } from 'node:fs';

type Figure = {
  description: string;
  blob: string;
  sourceSha256: string;
  sha256: string;
  stylesheetOmitted: boolean;
  staticViewport?: boolean;
};

export const genaiMedia = JSON.parse(readFileSync(
  new URL('./genai-course-figures.json', import.meta.url), 'utf8',
)) as { revision: string; license: { blob: string; sha256: string }; figures: Record<string, Figure> };

export const genaiSource = 'https://github.com/NirDiamant/GenAI_Agents/blob/' + genaiMedia.revision + '/';
export const genaiAssetDirectory = 'Assets/Sources/GenAI Agents/repository-figures';
const raw = 'https://github.com/NirDiamant/GenAI_Agents/raw/' + genaiMedia.revision + '/images/';
const remote = /^(?:https?:)?\/\//i;
const image = /!\[([^\]\n]*)\]\(((?:https?:)?\/\/[^)\s]+)\)/gi;

function altLabel(alt: string) {
  return alt.trim() && alt !== 'image' ? alt.trim() : 'Figure';
}

/** Work on reader prose only. The upstream notebook and its code stay intact. */
export function genaiReaderMedia(prose: string): string {
  function figureOrLink(url: string, alt: string) {
    const file = url.startsWith(raw) ? url.slice(raw.length) : '';
    const original = file ? genaiSource + 'images/' + file : url;
    const entry = Object.hasOwn(genaiMedia.figures, file) ? genaiMedia.figures[file] : undefined;
    if (!entry) return `[${altLabel(alt)}: open original image](${original})`;
    const modification = entry.staticViewport
      ? 'External stylesheet and pan controls omitted. The viewport now fits the whole graph.'
      : entry.stylesheetOmitted
        ? 'External stylesheet reference omitted; drawing and labels unchanged.' : 'Unmodified image.';
    return [
      '',
      `![[${genaiAssetDirectory}/${file}|${entry.description}]]`,
      `*Nir Diamant and GenAI Agents contributors, [original figure](${original}), [custom non-commercial license](${genaiSource}LICENSE). ${modification}*`,
      '',
    ].join('\n\n');
  }
  // Keep the destination of an image badge, including a Colab notebook,
  // but replace the badge with its text. Do not link to the badge service.
  return prose.replace(
    /\[!\[([^\]\n]*)\]\(((?:https?:)?\/\/[^)\s]+)\)\]\(([^)\s]+)\)/gi,
    (_tag, alt: string, _src: string, href: string) => `[${altLabel(alt)}](${href})`,
  ).replace(
    /(<a\b[^>]*>)\s*(<img\b[^>]*>)\s*(<\/a>)/gi,
    (match, open: string, tag: string, close: string) => {
      const src = tag.match(/\bsrc\s*=\s*["']([^"']+)["']/i)?.[1];
      if (!src || !remote.test(src)) return match;
      const alt = tag.match(/\balt\s*=\s*["']([^"']*)["']/i)?.[1] ?? '';
      return open + altLabel(alt).replaceAll('<', '&lt;').replaceAll('>', '&gt;') + close;
    },
  ).replace(/<img\b[^>]*>/gi, tag => {
    const src = tag.match(/\bsrc\s*=\s*["']([^"']+)["']/i)?.[1];
    if (!src || !remote.test(src)) return tag;
    const alt = tag.match(/\balt\s*=\s*["']([^"']*)["']/i)?.[1] ?? '';
    if (src.startsWith('https://img.shields.io/')) return altLabel(alt);
    // Blank lines let the Markdown figure render inside the notebook's HTML wrapper.
    return '\n\n' + figureOrLink(src, alt) + '\n\n';
  }).replace(image, (_tag, alt: string, src: string) => figureOrLink(src, alt));
}
