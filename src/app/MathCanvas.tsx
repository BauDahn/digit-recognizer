import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import katex from 'katex';
import {
  Calculator,
  CircleDot,
  Download,
  Eraser,
  Grid,
  Hand,
  Highlighter,
  FileText,
  Keyboard,
  Lasso,
  Maximize2,
  PenTool,
  RotateCcw,
  RotateCw,
  Sparkles,
  Trash2,
  ZoomIn,
  ZoomOut,
  ChevronDown,
} from 'lucide-react';
import { textToStrokes } from '../ml/handwritingGlyphs';
import { parseRelation } from '../ml/mathParser';
import { toLatex } from '../ml/mathAst';
import { toBytePixels } from '../ml/userModel';
import { MathCanvasCard, Point, Stroke } from '../types';
import { FunctionGraphCard } from './FunctionGraphCard';
import { drawPaper, drawStroke, MAX_ZOOM, MIN_ZOOM, PaperStyle, PaperTheme, pointInPolygon, strokeBounds, strokeTouched, toScreen, toWorld, View } from './ink';
import { LineResult, recognizePage, relayoutLine, resolveLines } from './recognition';
import { RecognitionPanel } from './RecognitionPanel';
import { ShortcutsDialog } from './ShortcutsDialog';
import { SymbolPicker } from './SymbolPicker';
import { SymbolLibraryModal } from './SymbolLibraryModal';
import { EMPTY_DOCUMENT, PageDocument, useDocument } from './useDocument';
import { useMathModel } from './useMathModel';

type ToolType = 'pen' | 'highlighter' | 'eraser' | 'lasso' | 'hand';

const PAGE_KEY = 'neuroscribe:page';
const DEFAULT_VIEW: View = { x: 100, y: 100, zoom: 1 };
const ANSWER_COLOR = '#2563eb';

/** Examples written on the page with the handwriting generator (see ml/handwritingGlyphs). */
const EXAMPLES: Array<{ label: string; text: string }> = [
  { label: '24 + 16 =', text: '24+16=' },
  { label: '2x + 3 = 11', text: '2x+3=11' },
  { label: 'x² − 4  (draws a graph)', text: 'x^2-4' },
  { label: '½ + ⅓ =', text: '\\frac{1}{2}+\\frac{1}{3}=' },
  { label: '√16 =', text: '\\sqrt{16}=' },
  { label: 'Σ x²,  x = 1…4', text: '\\sum_{x=1}^{4} x^2=' },
  { label: '∫₀² x² dx', text: '\\int_0^2 x^2 dx' },
  { label: '∂/∂x (x³)', text: '\\frac{\\partial}{\\partial x}(x^3)' },
  { label: '∂/∂x (x² + 3x)', text: '\\frac{\\partial}{\\partial x}(x^2+3x)' },
];

function loadPage(): PageDocument {
  try {
    const raw = localStorage.getItem(PAGE_KEY);
    if (!raw) return EMPTY_DOCUMENT;
    const parsed = JSON.parse(raw) as PageDocument;
    return Array.isArray(parsed.strokes) && Array.isArray(parsed.cards) ? parsed : EMPTY_DOCUMENT;
  } catch {
    return EMPTY_DOCUMENT;
  }
}

let idCounter = 0;
const newId = (prefix: string) => `${prefix}-${Date.now().toString(36)}-${(idCounter++).toString(36)}`;

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

/** "2 · x + 3" → LaTeX, or the text as is when it is not math */
function answerLatex(text: string): string {
  try {
    const node = parseRelation(text.replace(/·/g, '*'));
    return toLatex(node);
  } catch {
    return `\\text{${text.replace(/[\\{}$&#_%^~]/g, '')}}`;
  }
}

/** What a recognized line adds to the page next to it: its answer, written by hand, as a formula, or as a graph. */
function artifactsFor(line: LineResult): { strokes: Stroke[]; cards: MathCanvasCard[] } {
  const strokes: Stroke[] = [];
  const cards: MathCanvasCard[] = [];
  const { solved, bounds } = line;
  const right = bounds.x + bounds.width;

  if (solved.kind === 'function' && solved.graphExpr) {
    cards.push({
      id: newId('graph'),
      answerFor: line.key,
      x: right + 40,
      y: bounds.y,
      width: 320,
      height: 220,
      type: 'graph-plot',
      title: solved.readableText,
      graphExpr: solved.graphExpr,
    });
    return { strokes, cards };
  }

  const answer = solved.answer;
  if (!answer || (solved.kind !== 'evaluation' && solved.kind !== 'equation')) return { strokes, cards };

  const base = line.symbols.filter((s) => s.role === 'base' && s.box.height > 0.3 * s.box.width && s.pixels);
  const size = Math.max(24, median(base.map((s) => s.box.height)) || bounds.height);
  const bottom = median(base.map((s) => s.box.y + s.box.height)) || bounds.y + bounds.height;
  const writable = /^[0-9.\-+ ]+$/.test(answer);

  if (writable && line.expression.endsWith('=')) {
    // A number after the equals sign: written by hand, like a notebook would
    textToStrokes(answer, right + size * 0.3, bottom - size, size).forEach((points, i) =>
      strokes.push({ id: newId(`answer-${i}`), answerFor: line.key, points, color: ANSWER_COLOR, width: 3.5, tool: 'pen' })
    );
  } else if (answer !== '✓' && answer !== '✗') {
    cards.push({
      id: newId('answer'),
      answerFor: line.key,
      x: right + size * 0.4,
      y: bounds.y + bounds.height / 2 - 28,
      width: Math.min(640, Math.max(200, answer.length * 11 + 40)),
      height: 56,
      type: 'math-solution',
      latex: line.expression.endsWith('=') || solved.kind === 'equation' ? answerLatex(answer) : `= ${answerLatex(answer)}`,
    });
  } else {
    cards.push({ id: newId('answer'), answerFor: line.key, x: right + size * 0.4, y: bounds.y, width: 80, height: 56, type: 'math-solution', latex: answer === '✓' ? '\\checkmark' : '\\times' });
  }
  return { strokes, cards };
}

export const MathCanvas: React.FC = () => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const { doc, canUndo, canRedo, commit, undo, redo } = useDocument(useMemo(loadPage, []));
  const [draft, setDraft] = useState<PageDocument | null>(null); // while dragging (erasing, moving)
  const shown = draft ?? doc;

  const [view, setView] = useState<View>(DEFAULT_VIEW);
  const [tool, setTool] = useState<ToolType>('pen');
  const [penColor, setPenColor] = useState('#0f172a');
  const [penWidth, setPenWidth] = useState(3.5);
  const [paperStyle, setPaperStyle] = useState<PaperStyle>('lined');
  const [paperTheme] = useState<PaperTheme>('warm');
  const [selected, setSelected] = useState<string[]>([]);
  const [lasso, setLasso] = useState<Point[]>([]);
  const [spaceHeld, setSpaceHeld] = useState(false);
  const [, setFrame] = useState(0);

  const [toast, setToast] = useState<string | null>(null);
  const [examplesOpen, setExamplesOpen] = useState(false);
  const [shortcutsOpen, setShortcutsOpen] = useState(false);
  const [libraryOpen, setLibraryOpen] = useState(false);

  // Recognition session
  const mathModel = useMathModel();
  const [lines, setLines] = useState<LineResult[] | null>(null);
  const [readSignature, setReadSignature] = useState('');
  const [active, setActive] = useState<{ line: number; symbol: number } | null>(null);
  const [recognizing, setRecognizing] = useState(false);

  const activeStroke = useRef<Stroke | null>(null);
  const drag = useRef<{ kind: 'pan' | 'erase' | 'move' | 'move-image' | 'resize-image'; last: Point; start: Point; moved: boolean } | null>(null);
  const imageCache = useRef(new Map<string, HTMLImageElement>());
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const canvasSize = useRef({ w: 0, h: 0 });

  const showToast = useCallback((text: string, ms = 3800) => {
    setToast(text);
    window.clearTimeout((showToast as unknown as { t?: number }).t);
    (showToast as unknown as { t?: number }).t = window.setTimeout(() => setToast(null), ms);
  }, []);

  // ------------------------------------------------------------ persistence
  useEffect(() => {
    const timer = window.setTimeout(() => {
      try {
        localStorage.setItem(PAGE_KEY, JSON.stringify(doc));
      } catch {
        // too big for localStorage (pasted images are the heavy part): keep at least the ink
        try {
          localStorage.setItem(PAGE_KEY, JSON.stringify({ ...doc, cards: doc.cards.filter((c) => c.type !== 'image') }));
        } catch {
          // not saved
        }
      }
    }, 500);
    return () => window.clearTimeout(timer);
  }, [doc]);

  // ------------------------------------------------------------ sizing + drawing
  useEffect(() => {
    const resize = () => {
      const container = containerRef.current;
      const canvas = canvasRef.current;
      if (!container || !canvas) return;
      const rect = container.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(rect.width * dpr);
      canvas.height = Math.round(rect.height * dpr);
      canvasSize.current = { w: rect.width, h: rect.height };
      setFrame((f) => f + 1);
    };
    resize();
    const observer = new ResizeObserver(resize);
    if (containerRef.current) observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const { w, h } = canvasSize.current;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawPaper(ctx, w, h, view, paperStyle, paperTheme);

    ctx.save();
    ctx.translate(view.x, view.y);
    ctx.scale(view.zoom, view.zoom);
    const selectedSet = new Set(selected);
    for (const card of shown.cards) {
      if (card.type !== 'image' || !card.src) continue;
      let img = imageCache.current.get(card.id);
      if (!img) {
        img = new Image();
        img.onload = () => setFrame((f) => f + 1);
        img.src = card.src;
        imageCache.current.set(card.id, img);
      }
      if (!img.complete || img.naturalWidth === 0) continue;
      ctx.drawImage(img, card.x, card.y, card.width, card.height);
      if (card.id === selectedImage) {
        ctx.strokeStyle = '#6366f1';
        ctx.lineWidth = 1.5 / view.zoom;
        ctx.setLineDash([6 / view.zoom, 4 / view.zoom]);
        ctx.strokeRect(card.x, card.y, card.width, card.height);
        ctx.setLineDash([]);
        ctx.fillStyle = '#6366f1';
        const handle = 12 / view.zoom;
        ctx.fillRect(card.x + card.width - handle, card.y + card.height - handle, handle, handle);
      }
    }
    for (const stroke of shown.strokes) drawStroke(ctx, stroke, selectedSet.has(stroke.id));
    if (activeStroke.current) drawStroke(ctx, activeStroke.current);

    const box = strokeBounds(shown.strokes.filter((s) => selectedSet.has(s.id)));
    if (box) {
      ctx.strokeStyle = '#6366f1';
      ctx.lineWidth = 1.5 / view.zoom;
      ctx.setLineDash([6 / view.zoom, 4 / view.zoom]);
      ctx.strokeRect(box.x - 8, box.y - 8, box.width + 16, box.height + 16);
    }
    ctx.restore();

    if (lasso.length > 1) {
      ctx.save();
      ctx.strokeStyle = '#6366f1';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(lasso[0].x, lasso[0].y);
      for (const p of lasso) ctx.lineTo(p.x, p.y);
      ctx.stroke();
      ctx.restore();
    }
  });

  // ------------------------------------------------------------ helpers
  const eventPoint = (e: { clientX: number; clientY: number }): Point => {
    const rect = containerRef.current!.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  };

  const zoomAround = useCallback((sx: number, sy: number, factor: number) => {
    setView((v) => {
      const zoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, v.zoom * factor));
      const wx = (sx - v.x) / v.zoom;
      const wy = (sy - v.y) / v.zoom;
      return { zoom, x: sx - wx * zoom, y: sy - wy * zoom };
    });
  }, []);

  const centerZoom = useCallback((factor: number) => zoomAround(canvasSize.current.w / 2, canvasSize.current.h / 2, factor), [zoomAround]);

  const selectionBounds = useMemo(() => strokeBounds(shown.strokes.filter((s) => selected.includes(s.id))), [shown.strokes, selected]);

  const currentSignature = useMemo(() => doc.strokes.filter((s) => s.tool === 'pen' && !s.answerFor).map((s) => s.id).join(','), [doc.strokes]);
  const stale = lines !== null && readSignature !== currentSignature;

  // ------------------------------------------------------------ pointer
  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const sp = eventPoint(e);
    e.currentTarget.setPointerCapture(e.pointerId);
    const panning = tool === 'hand' || spaceHeld || e.button === 1 || e.buttons === 4;
    if (panning) {
      drag.current = { kind: 'pan', last: sp, start: sp, moved: false };
      return;
    }
    const world = toWorld(view, sp.x, sp.y);

    if (tool === 'pen' || tool === 'highlighter') {
      activeStroke.current = { id: newId('stroke'), color: penColor, width: penWidth, tool, points: [world] };
      setSelected([]);
    } else if (tool === 'eraser') {
      drag.current = { kind: 'erase', last: sp, start: sp, moved: false };
      eraseAt(world, doc);
    } else if (tool === 'lasso') {
      const image = [...shown.cards].reverse().find((c) => c.type === 'image' && world.x >= c.x && world.x <= c.x + c.width && world.y >= c.y && world.y <= c.y + c.height);
      const insideInk = selectionBounds && world.x > selectionBounds.x - 8 && world.x < selectionBounds.x + selectionBounds.width + 8 && world.y > selectionBounds.y - 8 && world.y < selectionBounds.y + selectionBounds.height + 8;
      if (image && !insideInk) {
        setSelected([]);
        setSelectedImage(image.id);
        const corner = 16 / view.zoom;
        const onHandle = world.x > image.x + image.width - corner && world.y > image.y + image.height - corner;
        drag.current = { kind: onHandle ? 'resize-image' : 'move-image', last: world, start: world, moved: false };
      } else if (selectionBounds && world.x > selectionBounds.x - 8 && world.x < selectionBounds.x + selectionBounds.width + 8 && world.y > selectionBounds.y - 8 && world.y < selectionBounds.y + selectionBounds.height + 8) {
        drag.current = { kind: 'move', last: world, start: world, moved: false };
      } else {
        setLasso([sp]);
        setSelected([]);
        setSelectedImage(null);
      }
    }
  };

  const eraseAt = (world: Point, base: PageDocument) => {
    const radius = 14 / view.zoom;
    const current = draft ?? base;
    const strokes = current.strokes.filter((s) => !strokeTouched(s, world, radius));
    if (strokes.length !== current.strokes.length) {
      setDraft({ ...current, strokes });
      if (drag.current) drag.current.moved = true;
    }
  };

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const sp = eventPoint(e);
    const d = drag.current;
    if (d?.kind === 'pan') {
      setView((v) => ({ ...v, x: v.x + sp.x - d.last.x, y: v.y + sp.y - d.last.y }));
      d.last = sp;
      return;
    }
    const world = toWorld(view, sp.x, sp.y);
    if (activeStroke.current) {
      activeStroke.current.points.push(world);
      setFrame((f) => f + 1);
    } else if (d?.kind === 'erase') {
      eraseAt(world, doc);
    } else if ((d?.kind === 'move-image' || d?.kind === 'resize-image') && selectedImage) {
      const dx = world.x - d.last.x;
      const dy = world.y - d.last.y;
      const base = draft ?? doc;
      setDraft({
        ...base,
        cards: base.cards.map((c) => {
          if (c.id !== selectedImage) return c;
          if (d.kind === 'move-image') return { ...c, x: c.x + dx, y: c.y + dy };
          const width = Math.max(60, c.width + dx);
          return { ...c, width, height: (c.height / c.width) * width };
        }),
      });
      d.last = world;
      d.moved = true;
    } else if (d?.kind === 'move') {
      const dx = world.x - d.last.x;
      const dy = world.y - d.last.y;
      const ids = new Set(selected);
      const base = draft ?? doc;
      setDraft({
        ...base,
        strokes: base.strokes.map((s) => (ids.has(s.id) ? { ...s, points: s.points.map((p) => ({ x: p.x + dx, y: p.y + dy })) } : s)),
      });
      d.last = world;
      d.moved = true;
    } else if (tool === 'lasso' && lasso.length > 0) {
      setLasso((prev) => [...prev, sp]);
    }
  };

  const onPointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const d = drag.current;
    drag.current = null;
    if (activeStroke.current) {
      const finished = activeStroke.current;
      activeStroke.current = null;
      commit({ ...doc, strokes: [...doc.strokes, finished] });
    } else if (d && d.kind !== 'pan' && draft) {
      if (d.moved) commit(draft);
      setDraft(null);
    }
    if (tool === 'lasso' && lasso.length > 2) {
      const polygon = lasso.map((p) => toWorld(view, p.x, p.y));
      setSelected(doc.strokes.filter((s) => s.points.some((p) => pointInPolygon(p, polygon))).map((s) => s.id));
    }
    setLasso([]);
    try {
      e.currentTarget.releasePointerCapture(e.pointerId);
    } catch {
      // not captured
    }
  };

  // Wheel pans like paper; with Ctrl (or pinching on a trackpad) it zooms
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      if (e.ctrlKey || e.metaKey) {
        const rect = el.getBoundingClientRect();
        zoomAround(e.clientX - rect.left, e.clientY - rect.top, Math.exp(-e.deltaY * 0.0025));
      } else {
        setView((v) => ({ ...v, x: v.x - (e.shiftKey ? e.deltaY : e.deltaX), y: v.y - (e.shiftKey ? 0 : e.deltaY) }));
      }
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [zoomAround]);

  // ------------------------------------------------------------ actions
  const deleteSelection = useCallback(() => {
    if (selected.length === 0 && selectedImage) {
      commit({ ...doc, cards: doc.cards.filter((c) => c.id !== selectedImage) });
      setSelectedImage(null);
      return;
    }
    if (selected.length === 0) return;
    const ids = new Set(selected);
    commit({ ...doc, strokes: doc.strokes.filter((s) => !ids.has(s.id)) });
    setSelected([]);
  }, [selected, selectedImage, doc, commit]);

  /** Replace what the app wrote for these lines (answers, graphs) by the current results. */
  const writeArtifacts = useCallback(
    (results: LineResult[], base: PageDocument): PageDocument => {
      const keys = new Set(results.map((l) => l.key));
      let strokes = base.strokes.filter((s) => !s.answerFor || !keys.has(s.answerFor));
      let cards = base.cards.filter((c) => !c.answerFor || !keys.has(c.answerFor));
      for (const line of results) {
        const made = artifactsFor(line);
        strokes = [...strokes, ...made.strokes];
        cards = [...cards, ...made.cards];
      }
      return { strokes, cards };
    },
    []
  );

  /** Ctrl+V with an image on the clipboard: the exercise appears on the page, under the ink. */
  const pasteImage = useCallback(
    async (file: Blob) => {
      try {
        const bitmap = await createImageBitmap(file);
        const maxSide = 1400; // keeps the saved page small
        const scale = Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height));
        const canvas = document.createElement('canvas');
        canvas.width = Math.round(bitmap.width * scale);
        canvas.height = Math.round(bitmap.height * scale);
        const ctx = canvas.getContext('2d')!;
        ctx.fillStyle = '#ffffff'; // screenshots with transparency
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
        const src = canvas.toDataURL('image/jpeg', 0.9);

        const { w, h } = canvasSize.current;
        const width = Math.min(canvas.width, (w * 0.7) / view.zoom);
        const height = (canvas.height / canvas.width) * width;
        const at = toWorld(view, w * 0.15, h * 0.12);
        const card: MathCanvasCard = { id: newId('image'), x: at.x, y: at.y, width, height, type: 'image', src, title: 'Pasted exercise' };
        commit({ ...doc, cards: [...doc.cards, card] });
        setSelected([]);
        setSelectedImage(card.id);
        setTool('lasso');
        showToast('Exercise pasted. Write next to it; drag to move it, corner to resize, Del removes it.', 6500);
      } catch {
        showToast('Could not read that image.');
      }
    },
    [doc, view, commit, showToast]
  );

  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const t = e.target;
      if (t instanceof HTMLElement && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
      const item = Array.from(e.clipboardData?.items ?? []).find((i) => i.kind === 'file' && i.type.startsWith('image/'));
      const file = item?.getAsFile();
      if (!file) return;
      e.preventDefault();
      void pasteImage(file);
    };
    window.addEventListener('paste', onPaste);
    return () => window.removeEventListener('paste', onPaste);
  }, [pasteImage]);

  const recognize = useCallback(() => {
    if (mathModel.status === 'loading') return showToast('The network is still loading, one moment…');
    if (!mathModel.model) return showToast(`Could not load the network: ${mathModel.error ?? 'unknown error'}`);
    const chosen = selected.length > 0 ? doc.strokes.filter((s) => selected.includes(s.id)) : doc.strokes;
    if (!chosen.some((s) => s.tool === 'pen')) return showToast('Write some math with the pen first, for example 12 + 7 =');

    setRecognizing(true);
    // let the button show its state before the (synchronous) recognition runs
    window.setTimeout(() => {
      try {
        // reading only a selection still knows the definitions of the lines read before
        const context = selected.length > 0 ? (lines ?? []).map((l) => l.expression) : [];
        const results = recognizePage(chosen, mathModel.model!, selected.length === 0, context);
        if (results.length === 0) return showToast('No ink to read.');
        setLines(results);
        setReadSignature(currentSignature);
        setActive(null);
        commit(writeArtifacts(results, doc));
        const first = results[0].solved;
        showToast(
          results.length > 1
            ? `Read ${results.length} lines`
            : first.error
              ? `Read "${results[0].expression}" but: ${first.error}`
              : `Read "${results[0].expression}"${first.answer ? ` → ${first.answer}` : ''}`
        );
      } catch (err) {
        showToast(`Recognition failed: ${err instanceof Error ? err.message : String(err)}`);
      } finally {
        setRecognizing(false);
      }
    }, 20);
  }, [mathModel, selected, doc, lines, commit, showToast, writeArtifacts, currentSignature]);

  /** The user says what a symbol really is. */
  const correct = useCallback(
    (lineIndex: number, symbolIndex: number, char: string) => {
      if (!lines) return;
      const line = lines[lineIndex];
      const symbol = line.symbols[symbolIndex];
      setActive(null);
      if (symbol.char === char && !symbol.corrected) return;

      const previousChar = symbol.char;
      symbol.char = char;
      symbol.corrected = true;
      const updated = relayoutLine(line);
      // a definition changed: the lines that use it are solved again
      const all = resolveLines(lines.map((l, i) => (i === lineIndex ? updated : l)));
      setLines(all);
      commit(writeArtifacts(all.filter((l, i) => i === lineIndex || l.solved.answer !== lines[i].solved.answer || l.solved.kind !== lines[i].solved.kind), doc));

      if (symbol.pixels && mathModel.model?.classes.includes(char)) {
        showToast(`Learning that this is “${char}”, not “${previousChar}”…`, 2500);
        mathModel
          .teach(char, toBytePixels(symbol.pixels))
          .then((r) => showToast(r.learned ? `Learned: it now reads this as “${char}” (${(r.after.confidence * 100).toFixed(0)}%).` : `Saved your correction; the network still prefers “${r.after.char}” for it, correct it again to teach it more.`, 4500))
          .catch((err) => showToast(`Could not learn: ${err instanceof Error ? err.message : String(err)}`));
      }
    },
    [lines, doc, commit, writeArtifacts, mathModel, showToast]
  );

  const insertCard = useCallback(
    (latex: string) => {
      const world = toWorld(view, canvasSize.current.w / 2 - 120, canvasSize.current.h / 2);
      commit({ ...doc, cards: [...doc.cards, { id: newId('latex'), x: world.x, y: world.y, width: 260, height: 120, type: 'math-solution', latex }] });
    },
    [view, doc, commit]
  );

  const writeExample = (text: string) => {
    const origin = toWorld(view, 260, 200 + (doc.strokes.length > 0 ? 160 : 0));
    // below whatever is already on the page
    const bounds = strokeBounds(doc.strokes);
    const y = bounds ? Math.max(origin.y, bounds.y + bounds.height + 70) : origin.y;
    const made: Stroke[] = textToStrokes(text, origin.x, y, 48).map((points) => ({ id: newId('example'), points, color: '#0f172a', width: 3.5, tool: 'pen' }));
    commit({ ...doc, strokes: [...doc.strokes, ...made] });
    setExamplesOpen(false);
  };

  const exportPng = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const link = document.createElement('a');
    link.download = `neuroscribe-${Date.now()}.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
  }, []);

  const clearAll = useCallback(() => {
    if (doc.strokes.length === 0 && doc.cards.length === 0) return;
    if (window.confirm('Clear the whole page? You can undo this with Ctrl+Z.')) {
      commit(EMPTY_DOCUMENT);
      setSelected([]);
      setLines(null);
    }
  }, [doc, commit]);

  // ------------------------------------------------------------ keyboard
  useEffect(() => {
    const typing = (t: EventTarget | null) => t instanceof HTMLElement && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable);
    const onKeyDown = (e: KeyboardEvent) => {
      if (typing(e.target)) return;
      const mod = e.ctrlKey || e.metaKey;
      const key = e.key.toLowerCase();
      const handled = () => e.preventDefault();

      if (mod && key === 'z') {
        handled();
        if (e.shiftKey) redo();
        else undo();
      } else if (mod && key === 'y') {
        handled();
        redo();
      } else if (mod && key === 'a') {
        handled();
        setSelected(doc.strokes.filter((s) => !s.answerFor).map((s) => s.id));
      } else if (mod && key === 'enter') {
        handled();
        recognize();
      } else if (mod && key === 'k') {
        handled();
        setLibraryOpen((o) => !o);
      } else if (mod && key === 's') {
        handled();
        exportPng();
      } else if (mod && (key === '=' || key === '+')) {
        handled();
        centerZoom(1.2);
      } else if (mod && key === '-') {
        handled();
        centerZoom(1 / 1.2);
      } else if (mod && key === '0') {
        handled();
        setView((v) => ({ ...v, zoom: 1 }));
      } else if (!mod && !e.altKey) {
        if (key === 'enter') {
          handled();
          recognize();
        } else if (key === 'escape') {
          if (active) setActive(null);
          else if (shortcutsOpen) setShortcutsOpen(false);
          else if (libraryOpen) setLibraryOpen(false);
          else if (selectedImage) setSelectedImage(null);
          else if (selected.length > 0) setSelected([]);
          else if (lines) setLines(null);
        } else if (key === 'delete' || key === 'backspace') {
          if (selected.length > 0 || selectedImage) {
            handled();
            deleteSelection();
          }
        } else if (key === 'p') setTool('pen');
        else if (key === 'm') setTool('highlighter');
        else if (key === 'e') setTool('eraser');
        else if (key === 'l') setTool('lasso');
        else if (key === 'h') setTool('hand');
        else if (key === '1') setPenWidth(2.2);
        else if (key === '2') setPenWidth(3.5);
        else if (key === '3') setPenWidth(6.5);
        else if (key === '?' || (key === '/' && e.shiftKey)) setShortcutsOpen((o) => !o);
        else if (key === 'home') setView(DEFAULT_VIEW);
        else if (key === ' ') {
          handled();
          setSpaceHeld(true);
        }
      }
    };
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.key === ' ') setSpaceHeld(false);
    };
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
    };
  }, [doc, undo, redo, recognize, exportPng, centerZoom, deleteSelection, selected, selectedImage, active, lines, shortcutsOpen, libraryOpen]);

  // ------------------------------------------------------------ view
  const cursor = spaceHeld || tool === 'hand' ? 'cursor-grab active:cursor-grabbing' : tool === 'eraser' ? 'cursor-pointer' : 'cursor-crosshair';
  const activeSymbol = active && lines ? lines[active.line]?.symbols[active.symbol] : undefined;

  const toolButton = (id: ToolType, icon: React.ReactNode, title: string, activeColor = 'text-slate-900') => (
    <button
      onClick={() => setTool(id)}
      className={`p-2 rounded-md transition-colors ${tool === id ? `bg-white ${activeColor} shadow-2xs` : 'text-slate-600 hover:text-slate-900'}`}
      title={title}
      aria-pressed={tool === id}
    >
      {icon}
    </button>
  );

  const cardsToShow = shown.cards;
  const learningSummary = {
    learning: mathModel.learning,
    count: mathModel.corrections.length,
    headline: mathModel.corrections.length === 0 ? 'Using the pre-trained network' : `Taught ${mathModel.corrections.length} ${mathModel.corrections.length === 1 ? 'correction' : 'corrections'} by you`,
    detail:
      mathModel.corrections.length === 0
        ? 'Correct a symbol and the network updates itself for next time.'
        : 'They are saved in this browser and used from now on.',
    onForget: () => {
      if (window.confirm('Forget everything the network learned from your corrections?')) mathModel.forget().then(() => showToast('Back to the pre-trained network.'));
    },
  };

  return (
    <div className="relative w-full h-full overflow-hidden bg-slate-100 select-none">
      {/* Toolbars */}
      <div className="absolute top-3 left-3 z-20 flex flex-wrap items-start gap-2 pointer-events-none" style={{ right: lines ? 396 : 12 }}>
        <div className="pointer-events-auto bg-white/95 backdrop-blur-md border border-slate-200 shadow-md rounded-xl px-3 py-2 flex items-center gap-3">
          <div className="flex items-center gap-1 p-0.5 bg-slate-100 rounded-lg">
            {toolButton('pen', <PenTool className="w-4 h-4" />, 'Pen (P)')}
            {toolButton('highlighter', <Highlighter className="w-4 h-4" />, 'Highlighter (M)', 'text-amber-600')}
            {toolButton('eraser', <Eraser className="w-4 h-4" />, 'Eraser (E): removes the whole stroke it touches', 'text-rose-600')}
            {toolButton('lasso', <Lasso className="w-4 h-4" />, 'Lasso (L): circle ink to select it, then drag to move or press Delete', 'text-indigo-600')}
            {toolButton('hand', <Hand className="w-4 h-4" />, 'Hand (H, or hold Space): move around the page')}
          </div>

          <div className="h-5 w-px bg-slate-200" />

          <div className="flex items-center gap-1.5">
            {[
              ['#0f172a', 'Ink black'],
              ['#2563eb', 'Royal blue'],
              ['#059669', 'Emerald'],
              ['#dc2626', 'Crimson'],
              ['#7c3aed', 'Amethyst'],
              ['#d97706', 'Amber'],
            ].map(([color, name]) => (
              <button
                key={color}
                onClick={() => {
                  setPenColor(color);
                  if (tool === 'eraser' || tool === 'hand') setTool('pen');
                }}
                style={{ backgroundColor: color }}
                title={name}
                aria-label={name}
                className={`w-5 h-5 rounded-full transition-transform ${penColor === color ? 'scale-125 ring-2 ring-slate-400 ring-offset-1' : 'hover:scale-110'}`}
              />
            ))}
          </div>

          <div className="h-5 w-px bg-slate-200" />

          <div className="flex items-center gap-1">
            {[
              [2.2, 'Fine', '1'],
              [3.5, 'Medium', '2'],
              [6.5, 'Bold', '3'],
            ].map(([size, label, key]) => (
              <button
                key={label}
                onClick={() => setPenWidth(size as number)}
                title={`${label} (${key})`}
                className={`px-2 py-1 text-xs rounded transition-colors ${penWidth === size ? 'bg-slate-200 text-slate-900 font-semibold' : 'text-slate-600 hover:bg-slate-100'}`}
              >
                {label}
              </button>
            ))}
          </div>

          <div className="h-5 w-px bg-slate-200" />

          <button
            onClick={recognize}
            disabled={recognizing}
            className="flex items-center gap-1.5 px-3.5 py-1.5 bg-slate-900 text-white rounded-lg text-xs font-semibold hover:bg-slate-800 transition-all shadow-xs disabled:opacity-70"
            title="Recognize and solve (Enter or Ctrl+Enter). With ink selected, only the selection is read."
          >
            {recognizing ? (
              <>
                <div className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                <span>Reading…</span>
              </>
            ) : (
              <>
                <Calculator className="w-3.5 h-3.5" />
                <span>{selected.length > 0 ? 'Read selection' : 'Recognize & solve'}</span>
              </>
            )}
          </button>
        </div>

        <div className="pointer-events-auto flex items-center gap-2">
          <div className="relative">
            <button
              onClick={() => setExamplesOpen((o) => !o)}
              className="bg-white/95 backdrop-blur-md border border-slate-200 shadow-xs rounded-xl px-3 py-2 text-xs font-medium text-slate-700 flex items-center gap-1.5 hover:bg-white"
              aria-expanded={examplesOpen}
            >
              Examples <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
            </button>
            {examplesOpen && (
              <div className="absolute top-full mt-1 left-0 w-60 bg-white border border-slate-200 rounded-xl shadow-xl py-1.5 z-30">
                <div className="px-3 py-1 text-[10px] uppercase tracking-wider text-slate-400">Write on the page</div>
                {EXAMPLES.map((ex) => (
                  <button key={ex.text} onClick={() => writeExample(ex.text)} className="w-full text-left px-3 py-1.5 text-xs text-slate-700 hover:bg-slate-50 font-serif">
                    {ex.label}
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className="bg-white/95 backdrop-blur-md border border-slate-200 shadow-xs rounded-xl p-1 flex items-center gap-1">
            {([
              ['lined', <FileText key="l" className="w-4 h-4" />, 'Lined paper'],
              ['grid', <Grid key="g" className="w-4 h-4" />, 'Grid paper'],
              ['dots', <CircleDot key="d" className="w-4 h-4" />, 'Dotted paper'],
            ] as const).map(([id, icon, title]) => (
              <button key={id} onClick={() => setPaperStyle(id)} title={title} className={`p-1.5 rounded-md ${paperStyle === id ? 'bg-slate-200 text-slate-900' : 'text-slate-500 hover:bg-slate-100'}`}>
                {icon}
              </button>
            ))}
          </div>

          <div className="bg-white/95 backdrop-blur-md border border-slate-200 shadow-xs rounded-xl p-1 flex items-center gap-1">
            <button onClick={exportPng} className="p-1.5 text-slate-600 hover:text-slate-900 rounded hover:bg-slate-100" title="Save as image (Ctrl+S)">
              <Download className="w-4 h-4" />
            </button>
            <button onClick={clearAll} className="p-1.5 text-slate-500 hover:text-rose-600 rounded hover:bg-slate-100" title="Clear the page">
              <Trash2 className="w-4 h-4" />
            </button>
            <button onClick={() => setShortcutsOpen(true)} className="p-1.5 text-slate-500 hover:text-slate-900 rounded hover:bg-slate-100" title="Keyboard shortcuts (?)">
              <Keyboard className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Undo / redo / zoom */}
      <div className="absolute bottom-4 left-4 z-20 bg-white/95 backdrop-blur-md border border-slate-200 shadow-xs rounded-xl p-1 flex items-center gap-1 text-xs text-slate-600">
        <button onClick={undo} disabled={!canUndo} className="p-1.5 rounded hover:bg-slate-100 disabled:opacity-30 disabled:hover:bg-transparent" title="Undo (Ctrl+Z)">
          <RotateCcw className="w-4 h-4" />
        </button>
        <button onClick={redo} disabled={!canRedo} className="p-1.5 rounded hover:bg-slate-100 disabled:opacity-30 disabled:hover:bg-transparent" title="Redo (Ctrl+Shift+Z)">
          <RotateCw className="w-4 h-4" />
        </button>
        <div className="h-4 w-px bg-slate-200 mx-1" />
        <button onClick={() => centerZoom(1 / 1.2)} className="p-1.5 rounded hover:bg-slate-100" title="Zoom out (Ctrl+-)">
          <ZoomOut className="w-4 h-4" />
        </button>
        <button onClick={() => setView((v) => ({ ...v, zoom: 1 }))} className="px-2 py-1 font-mono font-medium text-slate-700 hover:bg-slate-100 rounded" title="Zoom 100% (Ctrl+0)">
          {Math.round(view.zoom * 100)}%
        </button>
        <button onClick={() => centerZoom(1.2)} className="p-1.5 rounded hover:bg-slate-100" title="Zoom in (Ctrl++)">
          <ZoomIn className="w-4 h-4" />
        </button>
        <div className="h-4 w-px bg-slate-200 mx-1" />
        <button onClick={() => setView(DEFAULT_VIEW)} className="p-1.5 rounded hover:bg-slate-100" title="Back to the start (Home)">
          <Maximize2 className="w-4 h-4" />
        </button>
      </div>

      {toast && (
        <div role="status" className="absolute top-16 left-1/2 -translate-x-1/2 z-40 max-w-[min(90vw,640px)] bg-emerald-900 text-white text-xs px-4 py-2 rounded-xl shadow-lg flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-emerald-300 shrink-0" />
          <span className="truncate">{toast}</span>
        </div>
      )}

      {mathModel.status === 'loading' && (
        <div className="absolute bottom-4 right-4 z-20 text-[11px] text-slate-500 bg-white/90 border border-slate-200 rounded-lg px-3 py-1.5">Loading the network…</div>
      )}

      {/* Page */}
      <div ref={containerRef} className="absolute inset-0">
        <canvas
          ref={canvasRef}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
          className={`w-full h-full block touch-none ${cursor}`}
          aria-label="Page: write math here"
        />

        {cardsToShow.map((card) => {
          const pos = toScreen(view, card.x, card.y);
          const scale = Math.min(1.2, Math.max(0.6, view.zoom));
          const common = { left: pos.x, top: pos.y, transform: `scale(${scale})`, transformOrigin: 'top left' } as const;
          const close = () => commit({ ...doc, cards: doc.cards.filter((c) => c.id !== card.id) });
          if (card.type === 'graph-plot' && card.graphExpr) {
            return (
              <div key={card.id} className="absolute z-10" style={common}>
                <FunctionGraphCard id={card.id} x={card.x} y={card.y} width={card.width} height={card.height} title={card.title || 'Graph'} expr={card.graphExpr} onClose={close} />
              </div>
            );
          }
          if (card.type === 'math-solution' && card.latex) {
            return (
              <div key={card.id} className="absolute z-10 bg-white/95 backdrop-blur-md rounded-xl px-4 py-2 border border-slate-200 shadow-md min-w-[80px]" style={common}>
                <button onClick={close} className="absolute -top-2 -right-2 w-5 h-5 rounded-full bg-white border border-slate-200 text-slate-400 hover:text-rose-500 text-xs leading-none" aria-label="Remove">
                  ×
                </button>
                <div
                  className="text-slate-900 text-lg"
                  dangerouslySetInnerHTML={{ __html: katex.renderToString(card.latex, { throwOnError: false }) }}
                />
              </div>
            );
          }
          return null;
        })}

        {/* Where each symbol was found: click to correct it */}
        {lines && !stale &&
          lines.flatMap((line, li) =>
            line.symbols.map((sym, si) => {
              const tl = toScreen(view, sym.box.x, sym.box.y);
              const isActive = active?.line === li && active.symbol === si;
              const low = sym.confidence < 0.6 && !sym.corrected;
              return (
                <button
                  key={`${line.key}-${sym.id}`}
                  onClick={() => setActive({ line: li, symbol: si })}
                  className={`absolute z-[15] rounded-md border-2 transition-colors ${
                    isActive ? 'border-indigo-500 bg-indigo-500/10' : sym.corrected ? 'border-emerald-400/70 hover:bg-emerald-400/10' : low ? 'border-amber-400/80 hover:bg-amber-400/10' : 'border-indigo-300/40 hover:border-indigo-400 hover:bg-indigo-400/10'
                  }`}
                  style={{ left: tl.x - 4, top: tl.y - 4, width: Math.max(14, sym.box.width * view.zoom + 8), height: Math.max(14, sym.box.height * view.zoom + 8) }}
                  title={`${sym.char === '*' ? '·' : sym.char}  ${(sym.confidence * 100).toFixed(0)}%: click to correct`}
                  aria-label={`Symbol ${sym.char}, click to correct`}
                >
                  <span className={`absolute -top-5 left-0 px-1 rounded text-[11px] font-serif leading-4 ${sym.corrected ? 'bg-emerald-600 text-white' : low ? 'bg-amber-500 text-white' : 'bg-slate-800/85 text-white'}`}>
                    {sym.char === '*' ? '·' : sym.char}
                  </span>
                </button>
              );
            })
          )}

        {lines && (
          <RecognitionPanel
            lines={lines}
            stale={stale}
            activeSymbol={active}
            onSelectSymbol={(line, symbol) => setActive({ line, symbol })}
            onInsertCard={insertCard}
            onClose={() => {
              setLines(null);
              setActive(null);
            }}
            onRecognizeAgain={recognize}
            learning={learningSummary}
          />
        )}

        {activeSymbol && mathModel.model && (
          <SymbolPicker
            symbol={activeSymbol}
            classes={mathModel.model.classes}
            onPick={(char) => correct(active!.line, active!.symbol, char)}
            onClose={() => setActive(null)}
          />
        )}
      </div>

      {shortcutsOpen && <ShortcutsDialog onClose={() => setShortcutsOpen(false)} />}
      <SymbolLibraryModal isOpen={libraryOpen} onClose={() => setLibraryOpen(false)} />
    </div>
  );
};
