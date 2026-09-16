/* =========================================================================
 *  DOM "stripper" — keeps empty controls **and** guarantees unique
 *  parser-semantic-id values by appending numeric suffixes
 * ========================================================================= */

const parse = () => {
  /* ---------- globals --------------------------------------------------- */
  const BLACKLISTED_TAGS = new Set([
    'script', 'style', 'link', 'meta', 'noscript', 'template',
    'iframe', 'svg', 'canvas', 'picture', 'video', 'audio',
    'object', 'embed'
  ]);

  /* When true, content occluded by an overlay (e.g. an open mega-menu) or
   * sitting behind an active aria-modal is stripped from the output, so the
   * agent only sees what a human could on the topmost interactable layer. */
  const HIDE_OCCLUDED = true;

  /* When true, an element whose bounding-box center sits outside the visible
   * window of an overflow-clipping ancestor (e.g. the off-screen slides of a
   * carousel/slider) is treated like occluded content: its own
   * text/interactivity is dropped, children are still traversed. Keeps the
   * simplified HTML aligned with the full-page screenshot, which never paints
   * overflow-clipped content. */
  const HIDE_CLIPPED = true;

  const ALLOWED_ATTR = new Set([
    'id', 'type', 'name', 'value', 'placeholder',
    'checked', 'disabled', 'readonly', 'required', 'maxlength',
    'min', 'max', 'step', 'role', 'tabindex', 'alt', 'title',
    'for', 'action', 'method', 'contenteditable', 'selected',
    'multiple', 'autocomplete', 'href'
  ]);

  const PRESERVE_EMPTY_TAGS = new Set([
    'input', 'select', 'textarea', 'button', 'img', 'head', 'title', 'form'
  ]);

  const USED_SEMANTIC_IDS = new Set();

  /* ---------- helpers -------------------------------------------------- */
  const copyAllowed = (src, dst) => {
    for (const a of src.attributes) {
      if (
        ALLOWED_ATTR.has(a.name) ||
        a.name.startsWith('aria-') ||
        a.name.startsWith('parser-')
      ) {
        dst.setAttribute(a.name, a.value);
      }
    }
  };

  const slug = (t) =>
    t.toLowerCase().replace(/\s+/g, ' ').trim()
      .replace(/[^\w]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 20);

  const uniqueName = (base) => {
    let name = base || 'item';
    if (!USED_SEMANTIC_IDS.has(name)) {
      USED_SEMANTIC_IDS.add(name);
      return name;
    }
    let i = 1;
    while (USED_SEMANTIC_IDS.has(name + i)) i++;
    USED_SEMANTIC_IDS.add(name + i);
    return name + i;
  };

  const isEmpty = (el) => {
    if (PRESERVE_EMPTY_TAGS.has(el.tagName.toLowerCase())) return false;
    if (el.hasAttribute('parser-semantic-id')) return false;
    for (const n of el.childNodes) {
      if (n.nodeType === 3 && n.textContent.trim()) return false;
      if (n.nodeType === 1 && !isEmpty(n)) return false;
    }
    return true;
  };

  const isVisible = (el) => {
    const style = window.getComputedStyle(el);
    const hidden =
      style.display === 'none' ||
      style.visibility === 'hidden' ||
      parseFloat(style.opacity) === 0;

    const zeroSize = el.offsetWidth === 0 && el.offsetHeight === 0;

    const rect = el.getBoundingClientRect();
    const scrollLeft = window.scrollX || document.documentElement.scrollLeft;
    const right = rect.right;
    const top = rect.top;
    const outOfPort = (right + scrollLeft < 0);

    let belowPortNotScrollable = false;
    if (top > window.innerHeight && !(document.documentElement.scrollHeight > window.innerHeight)) {
      let hasScrollableAncestor = false;
      for (let p = el?.parentElement; p; p = p.parentElement) {
        const cs = getComputedStyle(p);
        const canScrollY = /(auto|scroll)/.test(cs.overflowY) && p.scrollHeight > p.clientHeight;
        if (canScrollY) { hasScrollableAncestor = true; break; }
      }
      belowPortNotScrollable = !hasScrollableAncestor;
    }
    if (hidden || zeroSize || outOfPort || belowPortNotScrollable) return false;
    return true;
  };

  const replaceElement = (el, newTag, child) => {
    const r = document.createElement(newTag);
    for (const a of el.attributes) r.setAttribute(a.name, a.value);
    copyAllowed(child, r);
    r.innerHTML = child.innerHTML;
    return r;
  };

  const pullUpChild = (parent, child) => {
    copyAllowed(child, parent);
    parent.innerHTML = child.innerHTML;
  };

  const flatten = (el) => {
    while (el.children.length === 1) {
      const child = el.children[0];
      const p = el.tagName.toLowerCase();
      const c = child.tagName.toLowerCase();
      if (p !== 'div' && c !== 'div') break;
      el = (p === 'div' && c !== 'div')
        ? replaceElement(el, child.tagName, child)
        : (pullUpChild(el, child), el);
    }
    return el;
  };

  /* ---------- clear parser-* attrs before run -------------------------- */
  (() => {
    const clearParserAttrs = (el) => {
      if (!el || !el.attributes) return;
      for (const a of Array.from(el.attributes)) {
        if (a.name.startsWith('parser-')) el.removeAttribute(a.name);
      }
    };

    // Clear on <html> itself first
    clearParserAttrs(document.documentElement);

    // Walk every element efficiently and clear any parser-* attrs
    const walker = document.createTreeWalker(document, NodeFilter.SHOW_ELEMENT);
    let node = walker.currentNode;
    while (node) {
      clearParserAttrs(node);
      node = walker.nextNode();
    }
  })();


  /* ==================================================================== */
  /* Detect the single active "blocking layer" once. Returns { root, modal }
   * or null. `root` is the topmost interactable surface; everything outside it
   * that it visually covers is unreachable (just like a real human sees). */
  const activeOverlay = (() => {
    // 1) Explicit modal dialog — only its contents are reachable.
    for (const el of document.querySelectorAll('[aria-modal="true"]')) {
      const s = window.getComputedStyle(el);
      if (s.display !== 'none' && s.visibility !== 'hidden' && parseFloat(s.opacity) !== 0) {
        return { root: el, modal: true };
      }
    }

    // 2) Non-modal overlay (e.g. an open mega-menu / dropdown panel). Probe the
    //    viewport center: when no overlay is open this lands on page content
    //    (inside <main>) and we bail; when an overlay covers the page it lands
    //    on the overlay instead.
    const main = document.querySelector('main');
    const cx = window.innerWidth / 2, cy = window.innerHeight / 2;
    const hit = document.elementFromPoint(cx, cy);
    if (!hit) return null;
    if (main && main.contains(hit)) return null; // center is on page content → no overlay

    // Climb to the outermost positioned ancestor below <body> — the overlay root.
    let ov = null;
    for (let p = hit; p && p !== document.body && p !== document.documentElement; p = p.parentElement) {
      const cs = window.getComputedStyle(p);
      if (cs.position === 'fixed' || cs.position === 'absolute') ov = p;
    }
    if (!ov) return null;
    if (main && ov.contains(main)) return null; // a positioned page wrapper, not an overlay
    const r = ov.getBoundingClientRect();
    if (r.width * r.height < 0.15 * window.innerWidth * window.innerHeight) return null; // too small to be blocking
    return { root: ov, modal: false };
  })();

  // Viewport center-point occlusion test for a single element.
  // Returns true (covered), false (clear), or null (center is off-screen → untestable).
  function pointObscured(el) {
    const rect = el.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return false;
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    if (cx < 0 || cy < 0 || cx >= window.innerWidth || cy >= window.innerHeight) return null;
    const top = document.elementFromPoint(cx, cy);
    if (!top) return null;
    return top !== el && !el.contains(top);
  }

  // Intersect the client boxes of every overflow-clipping ancestor, per-axis.
  // No clipping ancestor → unbounded region (nothing dropped). A position:fixed
  // subtree escapes ancestor clipping, so it gets an unbounded region too.
  function clipRegionFor(el) {
    if (window.getComputedStyle(el).position === 'fixed') {
      return { top: -Infinity, left: -Infinity, right: Infinity, bottom: Infinity };
    }
    let top = -Infinity, left = -Infinity, right = Infinity, bottom = Infinity;
    for (let p = el.parentElement; p; p = p.parentElement) {
      const cs = window.getComputedStyle(p);
      if (cs.position === 'fixed') break;       // ancestors above don't clip a fixed subtree
      const clipsX = cs.overflowX !== 'visible';
      const clipsY = cs.overflowY !== 'visible';
      if (!clipsX && !clipsY) continue;
      const r = p.getBoundingClientRect();
      if (clipsX) { left = Math.max(left, r.left); right = Math.min(right, r.right); }
      if (clipsY) { top = Math.max(top, r.top); bottom = Math.min(bottom, r.bottom); }
    }
    return { top, left, right, bottom };
  }

  // True when the element's center lies within the visible clipping window.
  function centerInClip(el) {
    const r = el.getBoundingClientRect();
    if (r.width === 0 && r.height === 0) return false;
    const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
    const c = clipRegionFor(el);
    return cx >= c.left && cx < c.right && cy >= c.top && cy < c.bottom;
  }

  function isObscured(el) {
    if (activeOverlay) {
      if (activeOverlay.root.contains(el)) return false;  // inside the active layer → reachable
      if (el.contains(activeOverlay.root)) return false;  // ancestor/wrapper → keep as container
      if (activeOverlay.modal) return true;               // modal blocks everything else
      // Non-modal overlay: uncovered chrome stays (point test clears it), while
      // covered and off-screen content is blocked.
      const p = pointObscured(el);
      return p === null ? true : p;
    }
    // No overlay: off-screen content is reachable by scrolling.
    const p = pointObscured(el);
    return p === null ? false : p;
  }

  function automaticStripElement(original, parentName = '', parentIsClickable = false) {
    if (!original || original.nodeType !== 1) return null;
    const tag = original.tagName.toLowerCase();
    if (BLACKLISTED_TAGS.has(tag)) return null;
    if (tag.includes('-')) {
      const wrapper = document.createElement('div');
      const kids = original.shadowRoot
        ? Array.from(original.shadowRoot.children)
        : Array.from(original.children);
      for (const child of kids) {
        const cleaned = automaticStripElement(child, parentName, parentIsClickable);
        if (cleaned && !isEmpty(cleaned)) wrapper.appendChild(cleaned);
      }
      return wrapper.children.length > 0 ? wrapper : null;
    }
    if (!isVisible(original)) return null;

    // Computed once and reused for every interactivity / content gate below.
    const clipped = HIDE_CLIPPED && !centerInClip(original);
    // Treat off-window-by-center the same as occluded: suppress this node's own
    // text/interactivity but keep recursing so on-window descendants survive.
    const obscured = clipped || (HIDE_OCCLUDED && isObscured(original));

    let clone = document.createElement(original.tagName);
    copyAllowed(original, clone);

    const computedStyle = window.getComputedStyle(original);
    if (computedStyle.pointerEvents !== 'auto') {
      clone.setAttribute('parser-pointer-events', computedStyle.pointerEvents);
    }
    if (document.activeElement === original) {
      clone.setAttribute('parser-is-focused', 'true');
    }

    const isDisabled = original.disabled ||
      original.hasAttribute('disabled') ||
      computedStyle.pointerEvents === 'none';

    const probablyClickable = (() => {
      if (['button', 'select', 'summary', 'area', 'input'].includes(tag)) return true;
      if (tag === 'a' && original.hasAttribute('href')) return true;
      if (original.hasAttribute('onclick')) return true;
      const r = original.getAttribute('role');
      if (['button', 'link', 'checkbox', 'radio', 'option'].includes(r)) return true;
      return computedStyle.cursor === 'pointer';
    })();

    const isClickable = !parentIsClickable && probablyClickable && !isDisabled && !obscured;

    let thisName = '';
    if (isClickable) {
      let _baseName = (original.innerText || '').trim() ||
        original.getAttribute('aria-label') ||
        original.getAttribute('title') ||
        original.getAttribute('placeholder');
      if (!_baseName && tag === 'a') {
        const href = original.getAttribute('href') || '';
        _baseName = href.split('/').filter(Boolean).pop() || '';
      }
      const base = slug(_baseName || tag);
      thisName = uniqueName(parentName ? `${parentName}.${base}` : base);
      for (const e of [clone, original]) {
        e.setAttribute('parser-semantic-id', thisName);
        e.setAttribute('parser-clickable', 'true');
      }
    }

    if (original.closest('[parser-maybe-hoverable="true"]') && !obscured) {
      clone.setAttribute('parser-maybe-hoverable', 'true');
      original.setAttribute('parser-maybe-hoverable', 'true');
    }

    if (tag === 'input' || tag === 'textarea' || original.hasAttribute('contenteditable')) {
      const t = original.getAttribute('type') || 'text';
      const inputIsDisabled = original.disabled || original.readOnly || obscured;
      if (!inputIsDisabled && !thisName) {
        const base = slug((original.getAttribute('placeholder') ||
          original.getAttribute('name') ||
          original.value || '').trim() || tag);
        thisName = uniqueName(parentName ? `${parentName}.${base}` : base);
      }
      if (!inputIsDisabled && thisName) {
        clone.setAttribute('parser-semantic-id', thisName);
        clone.setAttribute('value', original.value || '');
        clone.setAttribute('parser-input-disabled', 'false');
        clone.setAttribute('parser-can-edit', !original.readOnly ? 'true' : 'false');
        original.setAttribute('parser-semantic-id', thisName);
      }
      if (!inputIsDisabled && thisName && t === 'number') {
        clone.setAttribute('parser-numeric-value', original.valueAsNumber || '');
      }
      if (!inputIsDisabled && thisName && original.selectionStart !== undefined) {
        clone.setAttribute('parser-selection-start', original.selectionStart);
        clone.setAttribute('parser-selection-end', original.selectionEnd);
      }
    }

    if (tag === 'select') {
      const selectIsDisabled = original.disabled || original.hasAttribute('disabled') || obscured;
      if (!selectIsDisabled) {
        if (!thisName) {
          const base = slug((original.getAttribute('name') || tag));
          thisName = uniqueName(parentName ? `${parentName}.${base}` : base);
        }
        clone.setAttribute('parser-semantic-id', thisName);
        clone.setAttribute('parser-value', original.value);
        clone.setAttribute('parser-selected-index', original.selectedIndex);
        clone.setAttribute('parser-has-multiple', original.multiple ? 'true' : 'false');
        const selectedOptions = Array.from(original.selectedOptions).map(opt => opt.value).join(',');
        clone.setAttribute('parser-selected-values', selectedOptions);
        original.setAttribute('parser-semantic-id', thisName);
        for (const opt of original.querySelectorAll('option')) {
          const o = document.createElement('option');
          o.textContent = opt.textContent.trim();
          o.setAttribute('value', opt.value);
          o.setAttribute('parser-selected', opt.selected ? 'true' : 'false');
          const optName = uniqueName(`${thisName}.${slug(opt.textContent)}`);
          o.setAttribute('parser-semantic-id', optName);
          opt.setAttribute('parser-semantic-id', optName);
          clone.appendChild(o);
        }
      }
    }

    for (const child of original.children) {
      const cleaned = automaticStripElement(
        child,
        thisName || parentName,
        parentIsClickable || isClickable
      );
      if (cleaned && (!isEmpty(cleaned))) {
        clone.appendChild(cleaned);
      }
    }

    // Skip this element's own text when it is occluded — its visible
    // descendants (if any peek out from under the overlay) are kept via the
    // recursion above, but the covered text behind the overlay is not emitted.
    if (!obscured) {
      for (const n of original.childNodes) {
        if (n.nodeType === 3 && n.textContent.trim()) {
          clone.appendChild(document.createTextNode(n.textContent.trim()));
        }
      }
    }

    clone = flatten(clone);
    for (let i = clone.children.length - 1; i >= 0; i--) {
      const c = clone.children[i];
      if (!PRESERVE_EMPTY_TAGS.has(c.tagName.toLowerCase()) && isEmpty(c)) {
        clone.removeChild(c);
      }
    }

    // Drop a bare occluded leaf (e.g. an <img> or empty control behind the
    // overlay) that survived only because it is in PRESERVE_EMPTY_TAGS. Keep
    // containers that still hold visible descendants.
    if (obscured && clone.children.length === 0 && !clone.textContent.trim()) {
      return null;
    }

    return clone;
  }

  const result = automaticStripElement(document.documentElement);
  return {
    html: result.outerHTML,
    url_map: Object.fromEntries(
      Array.from(result.querySelectorAll('a[parser-semantic-id][href]'))
        .map(el => [el.getAttribute('parser-semantic-id'), el.getAttribute('href')])
    ),
    clickable_elements: Array.from(result.querySelectorAll('[parser-clickable="true"]'))
      .map(el => el.getAttribute('parser-semantic-id')),
    hoverable_elements: Array.from(result.querySelectorAll('[parser-maybe-hoverable="true"]'))
      .map(el => el.getAttribute('parser-semantic-id')),
    input_elements: Array.from(result.querySelectorAll('input[parser-semantic-id], textarea[parser-semantic-id], [contenteditable][parser-semantic-id]'))
      .map(el => ({
        id: el.getAttribute('parser-semantic-id'),
        disabled: el.hasAttribute('parser-input-disabled'),
        type: el.getAttribute('type') || (el.tagName.toLowerCase() === 'textarea' ? 'textarea' : 'contenteditable'),
        value: el.value || el.textContent,
        canEdit: el.getAttribute('parser-can-edit') === 'true',
        isFocused: el.getAttribute('parser-is-focused') === 'true'
      })),
    select_elements: Array.from(result.querySelectorAll('select[parser-semantic-id]'))
      .map(el => ({
        id: el.getAttribute('parser-semantic-id'),
        value: el.value,
        selectedIndex: el.selectedIndex,
        multiple: el.multiple,
        selectedValues: Array.from(el.selectedOptions).map(opt => opt.value)
      })),
  };
}

parse();
