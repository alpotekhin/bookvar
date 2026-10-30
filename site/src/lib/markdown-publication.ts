interface TreeNode {
  type?: string;
  tagName?: string;
  properties?: Record<string, unknown>;
  children?: TreeNode[];
}

/** Keep accessible MathML in the document, but index the visible formula only. */
export function rehypeMathSearch() {
  return (tree: TreeNode) => {
    const visit = (node: TreeNode) => {
      const classes = node.properties?.className;
      if (node.tagName === 'span' && Array.isArray(classes) && classes.includes('katex-mathml')) {
        node.properties!.dataPagefindIgnore = 'all';
      }
      node.children?.forEach(visit);
    };
    visit(tree);
  };
}

/** Avoid token-level HTML expansion in complete, long source archive listings. */
export const archiveCodeDisplay = {
  name: 'bookvar-archive-code-display',
  hooks: {
    preprocessLanguage({ codeBlock }: {
      codeBlock: { code: string; language: string; parentDocument?: { sourceFilePath?: string } };
    }) {
      const path = codeBlock.parentDocument?.sourceFilePath?.replaceAll('\\', '/') ?? '';
      if (/\/generated\/(?:en\/)?sources\/courses\//.test(path) && codeBlock.code.length >= 5_000) {
        codeBlock.language = 'plaintext';
      }
    }
  }
};
