/** Transform prose and whole links, preserving fenced blocks and standalone inline code. */
export function mapMarkdownProse(markdown: string, transform: (prose: string) => string): string {
  const inline = (text: string): string => {
    // A code-formatted label is part of a real link. Match the enclosing link
    // before its inner code span; an outer code span still takes precedence.
    const spans = /!?\[[^\]\n]+\]\([^)\n]+\)|(?<!`)(`+)(?!`)[\s\S]*?(?<!`)\1(?!`)/g;
    let end = 0;
    let result = '';
    for (const span of text.matchAll(spans)) {
      result += transform(text.slice(end, span.index)) + (span[1] ? span[0] : transform(span[0]));
      end = span.index + span[0].length;
    }
    return result + transform(text.slice(end));
  };
  const openings = /^ {0,3}(`{3,}|~{3,})([^\n]*)(?:\n|$)/gm;
  let end = 0;
  let result = '';
  for (let open = openings.exec(markdown); open; open = openings.exec(markdown)) {
    if (open[1][0] === '`' && open[2].includes('`')) continue;
    const closings = new RegExp(`^ {0,3}${open[1][0]}{${open[1].length},}[ \\t]*$`, 'gm');
    closings.lastIndex = openings.lastIndex;
    const close = closings.exec(markdown);
    const fenceEnd = close ? close.index + close[0].length : markdown.length;
    result += inline(markdown.slice(end, open.index)) + markdown.slice(open.index, fenceEnd);
    end = fenceEnd;
    openings.lastIndex = fenceEnd;
  }
  return result + inline(markdown.slice(end));
}
