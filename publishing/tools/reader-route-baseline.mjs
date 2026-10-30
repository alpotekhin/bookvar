import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { beforeMathReflow } from './math-reflow-baseline.mjs';
import { beforeTopicConsolidation } from './topic-consolidation-baseline.mjs';
import { beforeConfidentLearningAddition } from './confident-learning-baseline.mjs';
import { beforeSelectionGapsAddition } from './selection-gaps-baseline.mjs';
import { beforeTensorFactorization } from './tensor-factorization-baseline.mjs';
import { beforeSystemsCompletion } from './systems-completion-baseline.mjs';
import { beforeServingOperations } from './serving-operations-baseline.mjs';
import { beforeServingNode } from './serving-node-baseline.mjs';
import { beforeCs336Residuals } from './cs336-residuals-baseline.mjs';
import { beforeHardwareSourceCompletion } from './hardware-source-baseline.mjs';

const defaultRoot = resolve(import.meta.dirname, '../..');

/**
 * Reconstruct the revision used by historical explanation audits. Only exact,
 * recorded display-math reflows and navigation edits are reversed; prose,
 * examples and assets are never normalized. Current route correctness is
 * tested separately in reading-paths; math preservation in math-reflow.
 * The historical SHA checks still reject any unrecorded body change.
 */
export function beforeReaderRouteNavigation(text, source, root = defaultRoot) {
  text = beforeTopicConsolidation(text, source, root);
  text = beforeCs336Residuals(text, source, root);
  text = beforeHardwareSourceCompletion(text, source, root);
  text = beforeServingNode(text, source, root);
  text = beforeServingOperations(text, source, root);
  text = beforeSystemsCompletion(text, source, root);
  // This later addition has a separate explanation audit and exact-byte guard.
  text = beforeTensorFactorization(text, source, root);
  text = beforeSelectionGapsAddition(text, source, root);
  text = beforeConfidentLearningAddition(text, source, root);
  text = beforeMathReflow(text, source, root);
  const report = JSON.parse(readFileSync(resolve(root,
    'docs/audit/2026-10-29-reader-route-navigation.json'), 'utf8'));
  const record = report.files.find(file => file.source === source);
  if (!record) return text;
  for (const change of [...record.changes].reverse()) {
    if (text.includes(change.new)) text = text.replace(change.new, change.old);
    else if (!text.includes(change.old)) {
      throw new Error(`Unrecognized navigation revision: ${source}: ${change.new}`);
    }
  }
  return text;
}

export function readBeforeReaderRouteNavigation(source, root = defaultRoot) {
  return beforeReaderRouteNavigation(readFileSync(resolve(root, source), 'utf8'), source, root);
}
