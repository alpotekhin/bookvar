export const MIN_FIGURE_ZOOM = 1;
export const MAX_FIGURE_ZOOM = 8;

export function clampFigureZoom(value: number): number {
  if (!Number.isFinite(value)) throw new Error('Figure zoom must be finite');
  return Math.max(MIN_FIGURE_ZOOM, Math.min(MAX_FIGURE_ZOOM, value));
}

export function figureDisplayWidth(naturalWidth: number, availableWidth: number, zoom: number, isVector = false): number {
  if (![naturalWidth, availableWidth].every(value => Number.isFinite(value) && value > 0))
    throw new Error('Figure dimensions must be positive and finite');
  return (isVector ? availableWidth : Math.min(naturalWidth, availableWidth)) * clampFigureZoom(zoom);
}

export function figureContainWidth(naturalWidth: number, naturalHeight: number,
  availableWidth: number, availableHeight: number, isVector = false): number {
  if (![naturalHeight, availableHeight].every(value => Number.isFinite(value) && value > 0))
    throw new Error('Figure dimensions must be positive and finite');
  return Math.min(figureDisplayWidth(naturalWidth, availableWidth, 1, isVector),
    availableHeight * naturalWidth / naturalHeight);
}

export function shouldOpenFigure(event: Pick<MouseEvent, 'button' | 'altKey' | 'ctrlKey' | 'metaKey' | 'shiftKey'>,
  linkUrl: string, imageUrl: string): boolean {
  if (event.button !== 0 || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return false;
  try {
    const link = new URL(linkUrl), image = new URL(imageUrl);
    return ['http:', 'https:'].includes(link.protocol) && link.href === image.href;
  } catch {
    return false;
  }
}

export function initializeFigureViewer(doc: Document = document): void {
  const dialog = doc.querySelector<HTMLDialogElement>('[data-figure-viewer]');
  if (!dialog || dialog.dataset.initialized || typeof dialog.showModal !== 'function') return;
  dialog.dataset.initialized = 'true';
  const viewport = dialog.querySelector<HTMLElement>('[data-viewer-viewport]')!;
  const image = dialog.querySelector<HTMLImageElement>('[data-viewer-image]')!;
  const caption = dialog.querySelector<HTMLElement>('[data-viewer-caption]')!;
  const error = dialog.querySelector<HTMLElement>('[data-viewer-error]')!;
  const original = dialog.querySelector<HTMLAnchorElement>('[data-viewer-original]')!;
  const minus = dialog.querySelector<HTMLButtonElement>('[data-viewer-minus]')!;
  const plus = dialog.querySelector<HTMLButtonElement>('[data-viewer-plus]')!;
  const close = dialog.querySelector<HTMLButtonElement>('[data-viewer-close]')!;
  const fit = dialog.querySelector<HTMLButtonElement>('[data-viewer-fit]')!;
  const scale = dialog.querySelector<HTMLOutputElement>('[data-viewer-scale]')!;
  const english = doc.documentElement.lang.startsWith('en');
  const labels = english
    ? { title: 'Illustration', fit: 'Fit', file: 'File', close: 'Close',
        minus: 'Zoom out', plus: 'Zoom in', viewport: 'Image area', hint: 'Enlarge illustration',
        error: 'The image could not be loaded. Open the original file.' }
    : { title: 'Иллюстрация', fit: 'Целиком', file: 'Файл', close: 'Закрыть',
        minus: 'Уменьшить', plus: 'Увеличить', viewport: 'Область изображения', hint: 'Увеличить иллюстрацию',
        error: 'Не удалось загрузить изображение. Откройте исходный файл.' };
  for (const key of ['title', 'fit', 'file', 'close'] as const)
    dialog.querySelector<HTMLElement>('[data-viewer-label="' + key + '"]')!.textContent = labels[key];
  minus.setAttribute('aria-label', labels.minus);
  plus.setAttribute('aria-label', labels.plus);
  viewport.setAttribute('aria-label', labels.viewport);
  error.textContent = labels.error;

  let zoom = 1;
  let trigger: HTMLAnchorElement | null = null;
  let previousOverflow = '';

  function renderZoom(next: number, preserveCenter = true): void {
    if (!image.naturalWidth || !dialog!.open) return;
    const oldWidth = image.getBoundingClientRect().width;
    const oldHeight = image.getBoundingClientRect().height;
    const centerX = (viewport.scrollLeft + viewport.clientWidth / 2) / Math.max(oldWidth, 1);
    const centerY = (viewport.scrollTop + viewport.clientHeight / 2) / Math.max(oldHeight, 1);
    const style = getComputedStyle(viewport);
    const available = viewport.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
    const availableHeight = viewport.clientHeight - parseFloat(style.paddingTop) - parseFloat(style.paddingBottom);
    zoom = clampFigureZoom(next);
    // Dimensionless SVGs may report a 300px intrinsic width. They can still
    // fill the viewport without raster upscaling or altering the original file.
    const isVector = new URL(image.currentSrc || image.src, doc.baseURI).pathname.toLowerCase().endsWith('.svg');
    const width = figureContainWidth(image.naturalWidth, image.naturalHeight,
      available, availableHeight, isVector) * zoom;
    image.style.width = width + 'px';
    scale.textContent = zoom + '×';
    minus.disabled = zoom === MIN_FIGURE_ZOOM;
    plus.disabled = zoom === MAX_FIGURE_ZOOM;
    if (preserveCenter && oldWidth > 0) {
      viewport.scrollLeft = centerX * width - viewport.clientWidth / 2;
      viewport.scrollTop = centerY * image.getBoundingClientRect().height - viewport.clientHeight / 2;
    } else {
      viewport.scrollLeft = viewport.scrollTop = 0;
    }
  }

  image.addEventListener('load', () => {
    image.style.display = 'block';
    renderZoom(1, false);
  });
  image.addEventListener('error', () => { image.style.display = 'none'; error.hidden = false; });
  minus.addEventListener('click', () => renderZoom(zoom - .5));
  plus.addEventListener('click', () => renderZoom(zoom + .5));
  fit.addEventListener('click', () => renderZoom(1, false));
  close.addEventListener('click', () => dialog.close());
  dialog.addEventListener('close', () => {
    doc.documentElement.style.overflowY = previousOverflow;
    if (trigger?.isConnected) trigger.focus({ preventScroll: true });
  });
  dialog.addEventListener('click', event => {
    if (event.target !== dialog) return;
    const box = dialog.getBoundingClientRect();
    if (event.clientX < box.left || event.clientX > box.right ||
        event.clientY < box.top || event.clientY > box.bottom) dialog.close();
  });
  dialog.addEventListener('keydown', event => {
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.key === '+' || event.key === '=') { event.preventDefault(); renderZoom(zoom + .5); }
    if (event.key === '-') { event.preventDefault(); renderZoom(zoom - .5); }
    if (event.key === '0') { event.preventDefault(); renderZoom(1, false); }
  });

  for (const link of doc.querySelectorAll<HTMLAnchorElement>('.sl-markdown-content a')) {
    const source = link.querySelector<HTMLImageElement>(':scope > img');
    if (!source) continue;
    // Keep PDF links, external references, modifier clicks and no-JS navigation intact.
    if (!shouldOpenFigure({ button: 0, altKey: false, ctrlKey: false, metaKey: false, shiftKey: false },
      link.href, source.currentSrc || source.src)) continue;
    link.title = labels.hint;
    link.addEventListener('click', event => {
      if (!shouldOpenFigure(event, link.href, source.currentSrc || source.src)) return;
      event.preventDefault();
      trigger = link;
      const figure = link.closest('figure');
      // Comparison images share the attribution immediately below their table.
      const block = link.closest('table') || link.closest('p');
      const following = block?.nextElementSibling;
      const attribution = figure?.querySelector('figcaption') ||
        (following?.matches('p') && following.querySelector('em') ? following : null);
      // Preserve the existing attribution links and KaTeX markup. textContent
      // would concatenate the visual, MathML and TeX versions of each formula.
      caption.replaceChildren();
      if (attribution) caption.append(...Array.from(attribution.childNodes, node => node.cloneNode(true)));
      original.href = link.href;
      image.alt = source.alt;
      image.style.display = 'none';
      error.hidden = true;
      image.src = source.currentSrc || source.src;
      zoom = 1;
      previousOverflow = doc.documentElement.style.overflowY;
      doc.documentElement.style.overflowY = 'hidden';
      dialog.showModal();
      caption.scrollTop = caption.scrollLeft = 0;
      if (image.complete && image.naturalWidth) {
        image.style.display = 'block';
        renderZoom(1, false);
      }
      close.focus();
    });
  }
  const resized = () => { if (dialog.open) renderZoom(zoom, false); };
  window.addEventListener('resize', resized);
  doc.addEventListener('astro:before-swap', () => {
    if (dialog.open) dialog.close();
    window.removeEventListener('resize', resized);
  }, { once: true });
}
