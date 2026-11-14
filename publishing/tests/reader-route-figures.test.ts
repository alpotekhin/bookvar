import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { resolve, join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { parse } from 'yaml';
import { beforeTopicConsolidation } from '../tools/topic-consolidation-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
describe('original source panels for reader-route review', () => {
  it('retains the complete S4 and Mamba author figures with their original permissions', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')).assets;
    const prefix = '00 Учебник/Assets/Figures/curated/';
    const owner = '00 Учебник/09 Dense FFN и Mixture of Experts/03 Mamba, RWKV, RetNet и гибридные архитектуры.md';
    const entry = (relative: string) => registry.find((row: { asset: string }) => row.asset === prefix + relative);
    const hash = (path: string) => createHash('sha256').update(readFileSync(resolve(root, path))).digest('hex');
    const s4 = 'mamba/s4-representations-original.png';
    const revision = 'e757cef57d89e448c413de7325ed5601aceaac13';
    const notice = 'site/public/licenses/s4-e757cef5-Apache-2.0.txt';
    const s4Hash = '11e2eeaac2697bc9757795e2b0eec73bba0c2622885625ae340e770f1f1122df';
    const noticeHash = 'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4';
    expect(hash(prefix + s4)).toBe(s4Hash);
    expect(hash(notice)).toBe(noticeHash);
    expect(entry(s4)).toMatchObject({
      sha256: s4Hash, source_asset_sha256: s4Hash, commit: revision,
      source_image_url: `https://raw.githubusercontent.com/state-spaces/s4/${revision}/assets/s4.png`,
      license: 'Apache-2.0', license_notice: notice, license_notice_sha256: noticeHash,
      metadata_status: 'verified', used_in: [owner],
    });
    for (const [file, id, filename, digest] of [
      ['mamba-v1-selective-copying.svg', 'S3.F2', 'copying.svg', '555785adcfd638cd35d7bc36b63e758f19df826d999d9d3bd70d83950057b253'],
      ['mamba-v1-architecture.svg', 'S3.F3', 'architecture.svg', 'b3e264a306798a4c518631be55247c6a4c3c60fdc8819b416b0dab12b3bdf6d0'],
    ]) {
      const path = 'mamba/' + file;
      expect(hash(prefix + path)).toBe(digest);
      expect(entry(path)).toMatchObject({
        source_url: 'https://arxiv.org/html/2312.00752v2#' + id,
        source_image_url: 'https://arxiv.org/html/2312.00752v2/' + filename,
        sha256: digest, source_asset_sha256: digest,
        license: 'CC BY 4.0', metadata_status: 'verified', used_in: [owner],
      });
    }
    expect(entry('alternatives-transformer-2026/mamba-selective-copying.png')).toMatchObject({
      publication_status: 'retired', replacement: prefix + 'mamba/mamba-v1-selective-copying.svg', used_in: [],
    });
    for (const file of ['mamba/ssm-three-views.png', 'mamba/mamba-block.png']) {
      expect(entry(file).used_in).not.toContain(owner);
      expect(entry(file).used_in).toContain('Concepts/Architectures/Mamba.md');
    }
  });

  it('preserves the three scaling figures and their distinct source permissions', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')).assets;
    const chapter = '00 Учебник/11 Pre-training и Scaling/43 Scaling laws.md';
    const p = '00 Учебник/Assets/Figures/curated/scaling-source-2026-11-10/';
    const old = '00 Учебник/Assets/Figures/curated/scaling-review-2026-10/';
    const entry = (asset: string) => registry.find((row: { asset: string }) => row.asset === asset);
    const hash = (asset: string) => createHash('sha256').update(readFileSync(resolve(root, asset))).digest('hex');
    for (const [name, previous, digest, dimensions_px, license] of [
      ['repeated-data-jmlr-figure1.png', 'repeated-data-effective-tokens-chart.png', '00868d33b07b44cc29e9db5e94119fd84bef8adc37eafd3165a549a07898225f', [2220, 1125], 'CC BY 4.0'],
      ['mup-sp-vs-mup.png', 'learning-rate-scaling-chart.png', 'd82bd9a55ed268a403e38916e1776fbd0baee1509512afd716b88c3f92f0b840', [1298, 632], 'MIT'],
      ['step-law-figure1.png', 'optimizer-search-surface-chart.png', 'c20ef1791d9fb88d2e182693db6e5c0a9bfb737c41991f21535dfb05d66a073d', [2642, 1099], 'CC BY-SA 4.0'],
    ] as const) {
      const path = p + name;
      expect(hash(path)).toBe(digest);
      const bytes = readFileSync(resolve(root, path));
      expect([bytes.readUInt32BE(16), bytes.readUInt32BE(20)]).toEqual(dimensions_px);
      expect(entry(path)).toMatchObject({
        sha256: digest, dimensions_px, license, used_in: [chapter], metadata_status: 'verified',
      });
      expect(entry(old + previous)).toMatchObject({
        publication_status: 'retired', replacement: path, used_in: [], metadata_status: 'needs-confirmation',
      });
    }
    const pdf = '05 Источники/Papers/Source PDFs/muennighoff-24-1000.pdf';
    const jmlr = 'https://jmlr.org/papers/volume26/24-1000/24-1000.pdf';
    expect(hash(pdf)).toBe('3e14dd572c42489f23bf6ffb741e0e9a686863c34704f5215c3e4dd39fc35502');
    expect(entry(p + 'repeated-data-jmlr-figure1.png')).toMatchObject({
      source_asset: pdf, source_asset_sha256: hash(pdf), source_image_url: jmlr + '#page=2',
      license_evidence: jmlr + '#page=1', source_page: 2, render_dpi: 360,
      crop_box_px: [425, 438, 2645, 1563],
    });
    expect(hash('site/public/licenses/mup-19814971-MIT.txt')).toBe('c2cfccb812fe482101a8f04597dfc5a9991a6b2748266c47ac91b6a5aae15383');
    expect(entry(p + 'mup-sp-vs-mup.png')).toMatchObject({
      source_image_url: 'https://raw.githubusercontent.com/microsoft/mup/19814971934ef91dd546f88e913fc963e096d11c/figures/sp_vs_mup_dashed.png',
      public_license_copy: 'site/public/licenses/mup-19814971-MIT.txt',
    });
    expect(entry(p + 'step-law-figure1.png')).toMatchObject({
      source_published_image_url: 'https://step-law.github.io/static/images/img1.png',
      source_image_url: 'https://raw.githubusercontent.com/step-law/step-law.github.io/d217c5cdea4e4daf7ad8ce9fdae3a6a3af1fcd4d/static/images/img1.png',
      license_evidence: 'https://github.com/step-law/step-law.github.io/blob/d217c5cdea4e4daf7ad8ce9fdae3a6a3af1fcd4d/index.html#L970-L974',
    });
  });
  it('records Voita attribution guidance without treating it as a named license', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')).assets;
    const guidance = 'https://lena-voita.github.io/nlp_course.html';
    const pinned = 'https://github.com/lena-voita/lena-voita.github.io/blob/6f2442dbb507571ac185c01ae5d5d006cf51a4c2/nlp_course.html#L226-L240';
    const rows = registry.filter((row: { attribution_guidance_evidence?: string }) => row.attribution_guidance_evidence === pinned);
    expect(rows).toHaveLength(13);
    for (const row of rows) {
      expect(row).toMatchObject({
        attribution_guidance_url: guidance, metadata_status: 'needs-confirmation',
        license: 'User permission record (not a named license)', license_url: null,
      });
      expect(['Lena Voita', 'Elena Voita']).toContain(row.author);
      expect(createHash('sha256').update(readFileSync(resolve(root, row.asset))).digest('hex')).toBe(row.sha256);
      expect(row.sha256).toBe(row.source_asset_sha256);
    }
    const thirdParty = registry.find((row: { id?: string }) => row.id === 'curated-attention-is-all-you-need-6c66971d0df0');
    expect(thirdParty).toMatchObject({ metadata_status: 'needs-confirmation', license_url: null });
    expect(thirdParty.attribution_guidance_url).toBeUndefined();
  });
  it('pins DistServe Figure 1 from arXiv and retires the USENIX crop without changing its record', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')).assets;
    const prefix = '00 Учебник/Assets/Figures/curated/inference-benchmark-review-2026-10/';
    const path = prefix + 'distserve-original-teaser.svg';
    const sha256 = 'bae5d6fba7ac2a797877f64ce30e1901b0e1140e4f24fc45a21e9c170f50e937';
    expect(createHash('sha256').update(readFileSync(resolve(root, path))).digest('hex')).toBe(sha256);
    expect(registry.find((row: { asset: string }) => row.asset === path)).toMatchObject({
      sha256, source_url: 'https://arxiv.org/html/2401.09670v3#S1.F1',
      source_image_url: 'https://arxiv.org/html/2401.09670v3/teaser.svg',
      license: 'CC BY-SA 4.0', license_url: 'https://creativecommons.org/licenses/by-sa/4.0/',
      license_evidence: 'https://arxiv.org/html/2401.09670v3', derivation: 'source-svg-copy',
      used_in: ['00 Учебник/14 Inference и оптимизация/58b Benchmarking, SLO и эксплуатация inference.md', 'en/00 Textbook/14 Inference and optimization/58b Benchmarking SLOs and inference operations.md'],
    });
    expect(registry.find((row: { asset: string }) => row.asset === prefix + 'distserve-figure1-original.png')).toMatchObject({
      publication_status: 'retired', replacement: path, used_in: [],
      metadata_status: 'needs-confirmation',
    });
  });
  it('pins the Gemma report figures and retires the old atlas crop', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')).assets;
    const prefix = '00 Учебник/Assets/Figures/curated/';
    const atlas = '02 Атлас моделей/Семейства/Gemma.md';
    const entry = (asset: string) => registry.find((row: { asset: string }) => row.asset === prefix + asset);
    for (const [name, report, figure, sha256, owner] of [
      ['original-eval_chart.svg', '2403.08295v4', 'S1.F1', '761406ee1516c24d7818a99b27507d7acb2a8c3474902e7d432ca61e44444abb', atlas],
      ['gemma-3-kv-cache.svg', '2503.19786v1', 'S5.F6', '304affd7e22f11162d405f074e64c23fa08ab580407af332cf57e8ffcd17abf5', atlas],
      ['gemma-4-mtp.png', '2607.02770v1', 'S2.F1', 'ec2cbe95b34ea81a85b3ee93c305bb4cf9e85375ff958248fed27bfb5eb2200b', '05 Источники/Papers/Gemma 3 и Gemma 4.md'],
    ]) {
      const path = 'gemma/' + name;
      const bytes = readFileSync(resolve(root, prefix + path));
      expect(createHash('sha256').update(bytes).digest('hex')).toBe(sha256);
      expect(entry(path)).toMatchObject({
        sha256, license: 'CC BY 4.0', source_url: `https://arxiv.org/html/${report}#${figure}`,
        used_in: expect.arrayContaining([owner]),
      });
    }
    const mtp = readFileSync(resolve(root, prefix + 'gemma/gemma-4-mtp.png'));
    expect([mtp.readUInt32BE(16), mtp.readUInt32BE(20)]).toEqual([1128, 924]);
    expect(entry('gemma/gemma-4-mtp.png')).toMatchObject({
      source_image_url: 'https://arxiv.org/src/2607.02770v1',
      source_archive_member: 'assets/mtp_gemma_4.pdf',
      source_asset_sha256: '652c0255a02c2539b29c941bab12eb32383b5dc1678f1b4ba70ecb7964b07554',
      render_dpi: 300,
    });
    expect(entry('atlas-courses-official/gemma-capability-comparison.png')).toMatchObject({
      publication_status: 'retired', used_in: [], replacement: prefix + 'gemma/original-eval_chart.svg',
    });
  });
  it('attributes both original Self-RAG panels to the canonical chapter', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')).assets;
    const chapter = '00 Учебник/15 Embeddings, Retrieval и RAG/63 RAG — полный конвейер.md';
    for (const [file, id, digest] of [
      ['original-teaser_self_rag_v8.svg', 'S1.F1', '910317861c0d471826a9d6a56f57da54ba4c5bf951430ed0794a490e90668866'],
      ['original-training_examples.svg', 'S3.F2', 'd6fd2c2ade2bda603f9521a80488852ef8fd02591e4cc54bb31f17d99d1b4430'],
    ]) {
      const path = '00 Учебник/Assets/Figures/curated/self-rag/' + file;
      expect(createHash('sha256').update(readFileSync(resolve(root, path))).digest('hex')).toBe(digest);
      expect(registry.find((row: { asset: string }) => row.asset === path)).toMatchObject({
        source_url: 'https://arxiv.org/html/2310.11511v1#' + id, sha256: digest,
        license: 'CC BY 4.0', used_in: expect.arrayContaining([chapter]),
      });
    }
  });
  it('pins each of the seven Gemma, Self-RAG and ASTRA originals with its own license', () => {
    const prefix = '00 Учебник/Assets/Figures/curated/';
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')).assets;
    const asset = (name: string) => registry.find((entry: { asset: string }) => entry.asset === prefix + name);
    const hash = (p: string) => createHash('sha256').update(readFileSync(resolve(root, p))).digest('hex');
    for (const [oldPath, original, digest, license] of [
      ['gemma/fig1.png', 'gemma/original-eval_chart.svg', '761406ee1516c24d7818a99b27507d7acb2a8c3474902e7d432ca61e44444abb', 'CC BY 4.0'],
      ['gemma/fig2.png', 'gemma/original-gemma_by_model_size_final.svg', 'aa00764a64affa0f22da77a66ac36330343c68d7410f9ffc8c1b68630b7f15d9', 'CC BY 4.0'],
      ['gemma/fig3.png', 'gemma/original-gemma_both_by_corpus_sensitive_final.svg', 'b39474a1f82aa6284eabdb9d93c73e0b5f346becfb5a01e4ce1ebe8cc645afc6', 'CC BY 4.0'],
      ['gemma/fig4.png', 'gemma/original-gemma_both_by_corpus_amem_final.svg', 'fb25f39ef1041e46ac47b86945f4189634faccae69a8d9482a35ced3c6e80386', 'CC BY 4.0'],
      ['self-rag/fig1.png', 'self-rag/original-teaser_self_rag_v8.svg', '910317861c0d471826a9d6a56f57da54ba4c5bf951430ed0794a490e90668866', 'CC BY 4.0'],
      ['astra/sft-pipeline.png', 'astra/original-sft-pipeline.png', '716ba2c62f61115c3a7de2bafcc2c553af3f6c8f1765b7085f9911ae867c5c4c', 'Apache-2.0'],
      ['astra/env-synthesis.png', 'astra/original-env.png', '929831c22a277a30824114ecaf79d72d41948e878174ff1d025afbe1a99e8b58', 'Apache-2.0'],
    ]) {
      expect(hash(prefix + original)).toBe(digest);
      expect(asset(original)).toMatchObject({ sha256: digest, license, metadata_status: 'verified' });
      expect(asset(oldPath)).toMatchObject({ publication_status: 'retired', replacement: prefix + original, used_in: [] });
    }
    const notice = 'site/public/licenses/astra-bdf6a46-Apache-2.0.txt';
    expect(hash(notice)).toBe('895371ba24d082f99f07353739e220ae74dcf2f5aecfec59827f9bcaced974f9');
    expect(readFileSync(resolve(root, notice), 'utf8')).toContain('Copyright 2026 Beike Language and Intelligence');
  });
  it('uses the single exact Alammar panel and the complete k1.5 overview SVG', () => {
    const assets = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')).assets;
    const read = (p: string) => readFileSync(resolve(root, p));
    const digest = (p: string) => createHash('sha256').update(read(p)).digest('hex');
    const gpt = '00 Учебник/Assets/Figures/curated/gpt-30/gpt2-transformer-block-vectors-2.png';
    const archive = '05 Источники/Originals/gpt2-2026-10-22/gpt2-transformer-block-vectors-2.png';
    expect(digest(gpt)).toBe('94d7c51658e7feffd01c66e61acd8897cb17e93989725f0a523a95ab8b771047');
    expect(assets.find((entry: { asset: string }) => entry.asset === gpt)).toMatchObject({
      sha256: digest(gpt), source_asset: archive, source_asset_sha256: digest(gpt), license: 'CC BY-NC-SA 4.0',
      used_in: expect.arrayContaining(['02 Атлас моделей/Семейства/GPT.md']),
    });
    const kimi = '00 Учебник/Assets/Figures/curated/grpo-2026-10-24/kimi-rl-system.svg';
    expect(digest(kimi)).toBe('8a3b559237567114ee58fda1be4a506049592ae23c9be4c2b1770226aa40bce3');
    expect(assets.find((entry: { asset: string }) => entry.asset === kimi)).toMatchObject({
      source_url: 'https://arxiv.org/html/2501.12599v1#S2.F3.sf1', sha256: digest(kimi),
      license: 'CC BY-NC-ND 4.0', used_in: ['02 Атлас моделей/Семейства/Kimi.md'],
    });
    for (const [suffix, replacement] of [['gpt-decoder-block.png', gpt], ['kimi-k15-rl-system.png', kimi]]) {
      expect(assets.find((entry: { asset: string }) => entry.asset?.endsWith('/' + suffix))).toMatchObject({
        publication_status: 'retired', replacement, used_in: [],
      });
    }
  });
  it('registers the complete K3 architecture diagram and its versioned report license', () => {
    const path = '00 Учебник/Assets/Figures/curated/kimi-k3/figure-2-architecture.png';
    const png = readFileSync(resolve(root, path));
    const sha256 = '2b79de684af4b6ec35c18cac56b7fb57667ef620c3be09bd41cc4684bdbb9ae8';
    const dimensions_px = [2472, 2316];
    expect(createHash('sha256').update(png).digest('hex')).toBe(sha256);
    expect([png.readUInt32BE(16), png.readUInt32BE(20)]).toEqual(dimensions_px);
    const assets = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')).assets;
    expect(assets.find((entry: { asset: string }) => entry.asset === path)).toMatchObject({
      author: 'Kimi Team', sha256, dimensions_px,
      license: 'CC BY-NC-ND 4.0', license_url: 'https://creativecommons.org/licenses/by-nc-nd/4.0/',
      source_url: 'https://arxiv.org/html/2607.24653v1#S2.F2', source_page: 3,
      source_asset_sha256: '936a7a3b655947b014ba96b8a790c3cdb6ea8b37eea514e4b7b52655e20af0f8',
      used_in: ['02 Атлас моделей/Семейства/Kimi.md'],
    });
  });
  it('registers the exact Engram host-offload diagram as a separately attributed source', () => {
    const path = '00 Учебник/Assets/Figures/curated/engram/engram-system.svg';
    const bytes = readFileSync(resolve(root, path));
    const digest = '055e55679d06de5b5841535daf793b145f9717788fe3d6ad506b53d0604c4e78';
    expect(createHash('sha256').update(bytes).digest('hex')).toBe(digest);
    const assets = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')).assets;
    expect(assets.find((entry: { asset: string }) => entry.asset === path)).toMatchObject({
      author: 'Cheng et al.',
      sha256: digest,
      license: 'CC BY 4.0',
      source_url: 'https://arxiv.org/html/2601.07372v1#S2.F2',
      source_image_url: 'https://arxiv.org/html/2601.07372v1/Engram_system.svg',
      used_in: ['05 Источники/Papers/Qwen3.8-Flash-Next.md'],
    });
  });
  it('records the exact author images and MIT license for the gated-attention comparison', () => {
    const registry = parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8')).assets;
    const sha = (bytes: Buffer) => createHash('sha256').update(bytes).digest('hex');
    const origin = 'https://github.com/qiuzh20/gated_attention/blob/f4c2a5f6ffd6ec709e0c60072c95ed4f5ce5b5d2/';
    const license = 'site/public/licenses/qiu-gated-attention-f4c2a5f6-MIT.txt';
    const licenseBytes = readFileSync(resolve(root, license));
    expect(sha(licenseBytes)).toBe('e7ad7da3894829cde8c6476ceb790bae394d4de11223f1eaebb0cef4da24a136');
    for (const [file, digest] of [
      ['baseline_selected_layer_attention_maps.png', 'd5c715763023390dd838a277fea5f70746e49e599ca02902b4b5171001f8aa0c'],
      ['gate_elementwise_selected_layer_attention_maps.png', '5eff1a578335c220b805a7fcebd960613515e3107b9f817eeb9125d941047652'],
    ]) {
      const path = '00 Учебник/Assets/Figures/curated/gated-attention-qiu/' + file;
      const row = registry.find((entry: { asset: string }) => entry.asset === path);
      expect(row).toMatchObject({
        author: 'Zihan Qiu', source_url: origin + file, license_url: origin + 'LICENSE',
        license: 'MIT', license_evidence: license, license_evidence_sha256: sha(licenseBytes),
        sha256: digest, dimensions_px: [1400, 1200], used_in: ['02 Атлас моделей/Семейства/Qwen.md'],
      });
      const png = readFileSync(resolve(root, path));
      expect(sha(png)).toBe(digest);
      expect([png.readUInt32BE(16), png.readUInt32BE(20)]).toEqual([1400, 1200]);
    }
  });
  it('keeps the Qwen block as the exact JPEG embedded in the course PDF', () => {
    const temp = mkdtempSync(join(tmpdir(), 'bookvar-qwen-proof-'));
    try {
      execFileSync('pdfimages', ['-f', '10', '-l', '10', '-j',
        resolve(root, '05 Источники/Courses/Stanford CS336 Spring 2026/Lectures/repository/lecture_04.pdf'),
        join(temp, 'slide')]);
      expect(readFileSync(resolve(root, '00 Учебник/Assets/Figures/curated/stanford-cs336-2026/architectures/qwen-next-block-original.jpg')))
        .toEqual(readFileSync(join(temp, 'slide-000.jpg')));
      const source = '02 Атлас моделей/Семейства/Qwen.md';
      const current = readFileSync(resolve(root, source), 'utf8');
      const page = beforeTopicConsolidation(current, source, root, '2026-11-09-qwen-72b-release-diff-additions.json');
      expect(page).toContain('qwen-next-block-original.jpg');
      expect(page).toContain('qwen-next-hybrid-architecture.jpg');
      expect(page).toContain('<summary>Полный слайд CS336');
      expect(current).not.toContain('qwen-next-block-original.jpg');
      expect(current).toContain('https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3-Next/model_architecture.png');
    } finally {
      rmSync(temp, { recursive: true, force: true });
    }
  });
});
