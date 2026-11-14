import { mapMarkdownProse } from './markdown-prose.js';

const courseRoot = '05 Источники/Courses/Scikit-learn MOOC/';
const courseRoute = 'sources/courses/scikit-learn-mooc/';
const slidePlayer = 'https://inria.github.io/scikit-learn-mooc/slides/index.html?file=../slides/';
const artRoot = 'https://raw.githubusercontent.com/allisonhorst/palmerpenguins/8957207b78d6ccd1b4654a9dd9c9041b657478ab/man/figures/';
const artAssets = 'Assets/Sources/Scikit-learn MOOC/palmerpenguins/';
const artPermission = 'https://github.com/allisonhorst/palmerpenguins/blob/8957207b78d6ccd1b4654a9dd9c9041b657478ab/vignettes/art.Rmd#L12';
const slideName = (file: string) => file.replaceAll('_', '-');

function mediaLinks(markdown: string): string {
  const rendered = mapMarkdownProse(markdown, prose => prose
    .replace(/<iframe class="video"[^>]*\bsrc="https:\/\/www\.youtube\.com\/embed\/([\w-]{11})(?:\?rel=0)?"[^>]*>\s*<\/iframe>/g,
      (_tag, id: string) => '[Watch the course video on YouTube](https://www.youtube.com/watch?v=' + id + ')')
    .replace(/<iframe class="slides"\s+src="https:\/\/inria\.github\.io\/scikit-learn-mooc\/slides\/index\.html\?file=\.\.\/slides\/([a-z_]+)\.md"><\/iframe>\s*(?:To navigate in the slides, \*\*first click on the slides\*\*, then:\n- press the \*\*arrow keys\*\* to go to the next\/previous slide;\n- press \*\*"P"\*\* to toggle presenter mode to see the notes;\n- press \*\*"F"\*\* to toggle full-screen mode\.\s*)?/g,
      (_tag, file: string) => '[Read the slides and speaker notes](/' + courseRoute + 'slides/' + slideName(file) + '/) · ' +
        '[Open the Inria slideshow](' + slidePlayer + file + '.md)\n')
    .replace(/!\[([^\]\n]+)\]\((https:\/\/raw\.githubusercontent\.com\/allisonhorst\/palmerpenguins\/[a-f0-9]{40}\/man\/figures\/(?:lter_penguins|culmen_depth)\.png)\)/g,
      (match, alt: string, url: string) => url.startsWith(artRoot) ? '![[' + artAssets + url.slice(artRoot.length) + '|' + alt + ']]' : match));
  return rendered.replace(
    /(!\[\[Assets\/Sources\/Scikit-learn MOOC\/palmerpenguins\/([^|]+)\|[^\]\n]+\]\]\n\n\*Artwork by @allison_horst\. \[Palmer Penguins teaching artwork\]\([^)]+\))\.\*/g,
    (_caption, lead: string, file: string) => lead + ' · [Original PNG](' + artRoot + file +
      ') · [Teaching-use permission](' + artPermission + ').*',
  );
}

const layout = /^\s*(?:\.[a-z][\w-]*)+\[/i;
const notesOpen = '<details>\n<summary>Speaker notes</summary>\n\n';
const notesClose = '\n\n</details>\n';

// The pinned decks use Remark's class wrappers around ordinary Markdown.
// Strip the layout only: square brackets inside links, figures and code are content.
export function readingSlides(markdown: string): string {
  const lines = markdown.split('\n');
  let fence: string | undefined;
  let wrappers = 0;
  let inNotes = false;

  function unwrap(line: string): string {
    const start = line.match(layout)?.[0];
    if (!start) return line.replace(/\.small\[(\(the census\))\]/g, '$1');
    let depth = 1;
    for (let i = start.length; i < line.length; i++) {
      if (line[i] === '\\') { i++; continue; }
      if (line[i] === '[') depth++;
      if (line[i] === ']' && --depth === 0) {
        return unwrap(line.slice(start.length, i)) + line.slice(i + 1);
      }
    }
    wrappers++;
    return unwrap(line.slice(start.length));
  }

  function closeNotes() {
    if (!inNotes) return '';
    inNotes = false;
    return notesClose;
  }

  const rendered = lines.map(line => {
    const marker = line.match(/^ {0,3}(\x60{3,}|~{3,})(.*)$/);
    if (fence) {
      if (marker?.[1][0] === fence[0] && marker[1].length >= fence.length && !marker[2].trim()) fence = undefined;
      return line;
    }
    if (marker) { fence = marker[1]; return line; }
    if (/^class:\s+[\w, -]+$/.test(line)) return '';
    if (/^---\s*$/.test(line)) {
      if (wrappers) throw new Error('Unclosed scikit-learn MOOC slide wrapper');
      return closeNotes() + '\n' + line;
    }
    if (/^--\s*$/.test(line)) return closeNotes();
    if (/^\?\?\?\s*$/.test(line)) {
      const previous = closeNotes();
      inNotes = true;
      return previous + '\n' + notesOpen;
    }
    if (wrappers && /^\s*\]\s*$/.test(line)) { wrappers--; return ''; }
    // The reader already has an H1. Deck headings must appear in its contents.
    return unwrap(line).replace(/^# (?=\S)/, '## ');
  }).join('\n') + closeNotes();
  // Some source fragments include a notes marker but no note.
  return rendered.replace(/<details>\n<summary>Speaker notes<\/summary>\s*<\/details>\n/g, '');
}

export function scikitMoocReaderBody(route: string, source: string, body: string): string {
  if (!route.startsWith(courseRoute) || !source.startsWith(courseRoot)) return body;
  const filename = source.slice(courseRoot.length).match(/^slides\/([a-z_]+)\.md\.md$/)?.[1];
  if (filename && filename !== 'intro_words') {
    return readingSlides(body) + '\n[Open the Inria slideshow](' + slidePlayer + filename + '.md)\n';
  }
  return mediaLinks(body);
}
