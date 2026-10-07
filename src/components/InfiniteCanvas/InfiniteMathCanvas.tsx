import React, { useRef, useState, useEffect, useCallback } from 'react';
import { Stroke, Point, MathRecognitionResult, MathCanvasCard } from '../../types';
import { MathResultModal } from './MathResultModal';
import { FunctionGraphCard } from './FunctionGraphCard';
import katex from 'katex';
import { loadBuiltinModel } from '../../ml/modelFile';
import { displayChar, recognizeStrokes } from '../../ml/strokeRecognizer';
import { solve } from '../../ml/mathParser';
import { textToStrokes } from '../../ml/handwritingGlyphs';
import {
  PenTool,
  Highlighter,
  Eraser,
  Hand,
  Maximize2,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  RotateCw,
  Sparkles,
  Download,
  Trash2,
  Grid,
  FileText,
  CircleDot,
  Calculator,
  Lasso,
  Check,
  PlusCircle,
  HelpCircle,
} from 'lucide-react';

type ToolType = 'pen' | 'highlighter' | 'eraser' | 'lasso' | 'hand';
type PaperStyle = 'lined' | 'grid' | 'dots' | 'blank';
type PaperTheme = 'warm' | 'white' | 'dark';

export const InfiniteMathCanvas: React.FC = () => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Canvas Viewport Transformation
  const [viewTransform, setViewTransform] = useState({
    x: 100, // world origin offset
    y: 100,
    zoom: 1.0,
  });

  // Tools & Styling
  const [currentTool, setCurrentTool] = useState<ToolType>('pen');
  const [penColor, setPenColor] = useState<string>('#0f172a');
  const [penWidth, setPenWidth] = useState<number>(3.5);
  const [paperStyle, setPaperStyle] = useState<PaperStyle>('lined');
  const [paperTheme, setPaperTheme] = useState<PaperTheme>('warm');

  // Strokes state
  const [strokes, setStrokes] = useState<Stroke[]>([]);
  const [history, setHistory] = useState<Stroke[][]>([]);
  const [historyIndex, setHistoryIndex] = useState<number>(-1);

  // Active stroke / drag
  const isPointerDownRef = useRef(false);
  const activeStrokeRef = useRef<Stroke | null>(null);
  const lastPanPointRef = useRef<{ x: number; y: number } | null>(null);

  // Lasso selection loop points
  const [lassoPoints, setLassoPoints] = useState<Point[]>([]);
  const [selectedStrokeIds, setSelectedStrokeIds] = useState<string[]>([]);

  // Recognition state
  const [isRecognizing, setIsRecognizing] = useState(false);
  const [recognitionResult, setRecognitionResult] = useState<MathRecognitionResult | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [autoSolveToast, setAutoSolveToast] = useState<string | null>(null);

  // Canvas Cards (Interactive Function Plots & Sticky KaTeX notes)
  const [cards, setCards] = useState<MathCanvasCard[]>([]);

  // Push new state to history for Undo/Redo
  const pushToHistory = useCallback(
    (newStrokes: Stroke[]) => {
      setHistory((prev) => {
        const trimmed = prev.slice(0, historyIndex + 1);
        return [...trimmed, newStrokes];
      });
      setHistoryIndex((prev) => prev + 1);
      setStrokes(newStrokes);
    },
    [historyIndex]
  );

  const handleUndo = useCallback(() => {
    if (historyIndex > 0) {
      const nextIndex = historyIndex - 1;
      setHistoryIndex(nextIndex);
      setStrokes(history[nextIndex]);
    } else if (historyIndex === 0) {
      setHistoryIndex(-1);
      setStrokes([]);
    }
  }, [history, historyIndex]);

  const handleRedo = useCallback(() => {
    if (historyIndex < history.length - 1) {
      const nextIndex = historyIndex + 1;
      setHistoryIndex(nextIndex);
      setStrokes(history[nextIndex]);
    }
  }, [history, historyIndex]);

  // Coordinate transformations
  const screenToWorld = useCallback(
    (sx: number, sy: number) => {
      return {
        x: (sx - viewTransform.x) / viewTransform.zoom,
        y: (sy - viewTransform.y) / viewTransform.zoom,
      };
    },
    [viewTransform]
  );

  const worldToScreen = useCallback(
    (wx: number, wy: number) => {
      return {
        x: wx * viewTransform.zoom + viewTransform.x,
        y: wy * viewTransform.zoom + viewTransform.y,
      };
    },
    [viewTransform]
  );

  // Resize canvas to fill container
  useEffect(() => {
    const handleResize = () => {
      const container = containerRef.current;
      const canvas = canvasRef.current;
      if (!container || !canvas) return;

      const rect = container.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = rect.height;
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Main Render Loop for the Infinite Canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    const { x: vx, y: vy, zoom } = viewTransform;

    // 1. Draw Paper Background
    let bgColor = '#faf9f6'; // Warm notebook cream
    if (paperTheme === 'white') bgColor = '#ffffff';
    if (paperTheme === 'dark') bgColor = '#0f172a';

    ctx.fillStyle = bgColor;
    ctx.fillRect(0, 0, w, h);

    // Draw Grid / Lined Paper Pattern
    ctx.save();
    if (paperTheme === 'dark') {
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.07)';
    } else {
      ctx.strokeStyle = 'rgba(15, 23, 42, 0.08)';
    }

    if (paperStyle === 'lined') {
      // OneNote Lined Paper
      const lineSpacing = 36 * zoom;
      const startY = vy % lineSpacing;
      ctx.lineWidth = 1;

      for (let y = startY; y < h; y += lineSpacing) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }

      // Red Margin line (classic ruled notebook)
      const marginX = worldToScreen(80, 0).x;
      if (marginX >= 0 && marginX <= w) {
        ctx.strokeStyle = paperTheme === 'dark' ? 'rgba(244, 63, 94, 0.25)' : 'rgba(244, 63, 94, 0.35)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(marginX, 0);
        ctx.lineTo(marginX, h);
        ctx.stroke();
      }
    } else if (paperStyle === 'grid') {
      // 5mm Math Graph Paper
      const gridSize = 24 * zoom;
      const startX = vx % gridSize;
      const startY = vy % gridSize;
      ctx.lineWidth = 1;

      for (let x = startX; x < w; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
      }
      for (let y = startY; y < h; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }
    } else if (paperStyle === 'dots') {
      // Dot Grid
      const dotSpacing = 28 * zoom;
      const startX = vx % dotSpacing;
      const startY = vy % dotSpacing;
      ctx.fillStyle = paperTheme === 'dark' ? 'rgba(255, 255, 255, 0.15)' : 'rgba(15, 23, 42, 0.18)';

      for (let x = startX; x < w; x += dotSpacing) {
        for (let y = startY; y < h; y += dotSpacing) {
          ctx.beginPath();
          ctx.arc(x, y, 1.2 * Math.min(2, Math.max(0.7, zoom)), 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }
    ctx.restore();

    // 2. Render Ink Strokes with Transformation
    ctx.save();
    ctx.translate(vx, vy);
    ctx.scale(zoom, zoom);

    for (const stroke of strokes) {
      if (stroke.points.length < 1) continue;

      ctx.save();
      const isSelected = selectedStrokeIds.includes(stroke.id);

      if (stroke.tool === 'highlighter') {
        ctx.globalCompositeOperation = 'multiply';
        ctx.globalAlpha = 0.45;
        ctx.strokeStyle = stroke.color;
        ctx.lineWidth = stroke.width * 5;
        ctx.lineCap = 'square';
        ctx.lineJoin = 'bevel';
      } else {
        ctx.globalAlpha = 1.0;
        ctx.strokeStyle = isSelected ? '#4f46e5' : stroke.color;
        ctx.lineWidth = stroke.width;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
      }

      ctx.beginPath();
      const pts = stroke.points;
      if (pts.length === 1) {
        ctx.arc(pts[0].x, pts[0].y, stroke.width / 2, 0, Math.PI * 2);
        ctx.fillStyle = stroke.color;
        ctx.fill();
      } else {
        ctx.moveTo(pts[0].x, pts[0].y);
        for (let i = 1; i < pts.length; i++) {
          // Quadratic smoothing for natural ink feeling
          const midX = (pts[i - 1].x + pts[i].x) / 2;
          const midY = (pts[i - 1].y + pts[i].y) / 2;
          ctx.quadraticCurveTo(pts[i - 1].x, pts[i - 1].y, midX, midY);
        }
        ctx.lineTo(pts[pts.length - 1].x, pts[pts.length - 1].y);
        ctx.stroke();
      }
      ctx.restore();
    }

    // Render active stroke currently being drawn
    if (activeStrokeRef.current && activeStrokeRef.current.points.length > 0) {
      const active = activeStrokeRef.current;
      ctx.save();
      if (active.tool === 'highlighter') {
        ctx.globalCompositeOperation = 'multiply';
        ctx.globalAlpha = 0.45;
        ctx.strokeStyle = active.color;
        ctx.lineWidth = active.width * 5;
        ctx.lineCap = 'square';
      } else {
        ctx.strokeStyle = active.color;
        ctx.lineWidth = active.width;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
      }

      ctx.beginPath();
      const pts = active.points;
      ctx.moveTo(pts[0].x, pts[0].y);
      for (let i = 1; i < pts.length; i++) {
        const midX = (pts[i - 1].x + pts[i].x) / 2;
        const midY = (pts[i - 1].y + pts[i].y) / 2;
        ctx.quadraticCurveTo(pts[i - 1].x, pts[i - 1].y, midX, midY);
      }
      ctx.lineTo(pts[pts.length - 1].x, pts[pts.length - 1].y);
      ctx.stroke();
      ctx.restore();
    }

    ctx.restore();

    // 3. Render Lasso Selection overlay on screen space
    if (currentTool === 'lasso' && lassoPoints.length > 1) {
      ctx.save();
      ctx.strokeStyle = '#6366f1';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(lassoPoints[0].x, lassoPoints[0].y);
      for (let i = 1; i < lassoPoints.length; i++) {
        ctx.lineTo(lassoPoints[i].x, lassoPoints[i].y);
      }
      ctx.stroke();
      ctx.restore();
    }
  }, [
    paperStyle,
    paperTheme,
    strokes,
    viewTransform,
    worldToScreen,
    currentTool,
    lassoPoints,
    selectedStrokeIds,
  ]);

  // Pointer Handlers
  const handlePointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const container = containerRef.current;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;

    isPointerDownRef.current = true;
    e.currentTarget.setPointerCapture(e.pointerId);

    // Pan Tool or Middle-Mouse Drag or Space-held
    if (currentTool === 'hand' || e.button === 1 || e.buttons === 4) {
      lastPanPointRef.current = { x: sx, y: sy };
      return;
    }

    const worldPt = screenToWorld(sx, sy);

    if (currentTool === 'pen' || currentTool === 'highlighter') {
      const newStroke: Stroke = {
        id: `stroke-${Date.now()}-${Math.random()}`,
        color: penColor,
        width: penWidth,
        tool: currentTool,
        points: [worldPt],
      };
      activeStrokeRef.current = newStroke;
    } else if (currentTool === 'eraser') {
      // Stroke eraser: delete touched stroke immediately
      deleteStrokeAt(worldPt);
    } else if (currentTool === 'lasso') {
      setLassoPoints([{ x: sx, y: sy }]);
      setSelectedStrokeIds([]);
    }
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!isPointerDownRef.current) return;
    const container = containerRef.current;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;

    if (lastPanPointRef.current) {
      const dx = sx - lastPanPointRef.current.x;
      const dy = sy - lastPanPointRef.current.y;
      setViewTransform((prev) => ({ ...prev, x: prev.x + dx, y: prev.y + dy }));
      lastPanPointRef.current = { x: sx, y: sy };
      return;
    }

    const worldPt = screenToWorld(sx, sy);

    if (activeStrokeRef.current) {
      activeStrokeRef.current.points.push(worldPt);
      // Force quick redraw
      const canvas = canvasRef.current;
      if (canvas) {
        // Redraw is handled by state/requestAnimationFrame
        setStrokes((s) => [...s]);
      }
    } else if (currentTool === 'eraser') {
      deleteStrokeAt(worldPt);
    } else if (currentTool === 'lasso') {
      setLassoPoints((prev) => [...prev, { x: sx, y: sy }]);
    }
  };

  const handlePointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    isPointerDownRef.current = false;
    lastPanPointRef.current = null;

    if (activeStrokeRef.current) {
      const completedStroke = activeStrokeRef.current;
      activeStrokeRef.current = null;
      if (completedStroke.points.length > 0) {
        const next = [...strokes, completedStroke];
        pushToHistory(next);
      }
    }

    if (currentTool === 'lasso' && lassoPoints.length > 2) {
      // Determine which strokes are enclosed by lasso
      findEnclosedStrokes(lassoPoints);
      setLassoPoints([]);
    }

    try {
      e.currentTarget.releasePointerCapture(e.pointerId);
    } catch {
      // ignore
    }
  };

  // Stroke Eraser logic
  const deleteStrokeAt = (worldPt: Point) => {
    const eraserRadius = 15 / viewTransform.zoom;
    const remaining = strokes.filter((st) => {
      for (const p of st.points) {
        const dist = Math.hypot(p.x - worldPt.x, p.y - worldPt.y);
        if (dist <= eraserRadius + st.width) {
          return false; // hit! delete this stroke
        }
      }
      return true;
    });

    if (remaining.length !== strokes.length) {
      pushToHistory(remaining);
    }
  };

  // Lasso enclosed strokes detection using ray-casting point-in-polygon
  const findEnclosedStrokes = (screenLasso: Point[]) => {
    const worldPolygon = screenLasso.map((p) => screenToWorld(p.x, p.y));

    const isInside = (pt: Point) => {
      let inside = false;
      for (let i = 0, j = worldPolygon.length - 1; i < worldPolygon.length; j = i++) {
        const xi = worldPolygon[i].x,
          yi = worldPolygon[i].y;
        const xj = worldPolygon[j].x,
          yj = worldPolygon[j].y;
        const intersect = yi > pt.y !== yj > pt.y && pt.x < ((xj - xi) * (pt.y - yi)) / (yj - yi) + xi;
        if (intersect) inside = !inside;
      }
      return inside;
    };

    const selectedIds = strokes
      .filter((st) => st.points.some((p) => isInside(p)))
      .map((st) => st.id);

    setSelectedStrokeIds(selectedIds);
  };

  // Zoom with Wheel
  const handleWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const container = containerRef.current;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const zoomFactor = e.deltaY < 0 ? 1.12 : 0.89;
    const newZoom = Math.min(3.0, Math.max(0.25, viewTransform.zoom * zoomFactor));

    // Zoom centered around mouse pointer
    const wx = (mx - viewTransform.x) / viewTransform.zoom;
    const wy = (my - viewTransform.y) / viewTransform.zoom;

    setViewTransform({
      zoom: newZoom,
      x: mx - wx * newZoom,
      y: my - wy * newZoom,
    });
  };

  // Recognize the handwriting locally: strokes -> symbols (neural network) -> expression -> solver
  const handleRecognizeMath = async () => {
    if (strokes.length === 0) {
      setAutoSolveToast('Write a math expression first (e.g. 12 + 7 =).');
      setTimeout(() => setAutoSolveToast(null), 3000);
      return;
    }

    setIsRecognizing(true);
    setIsModalOpen(true);
    try {
      const targetStrokes = (
        selectedStrokeIds.length > 0 ? strokes.filter((s) => selectedStrokeIds.includes(s.id)) : strokes
      ).filter((s) => s.tool === 'pen' && s.points.length > 0 && !s.id.startsWith('answer-'));

      if (targetStrokes.length === 0) throw new Error('No pen strokes to recognize.');

      const model = await loadBuiltinModel('symbols');
      const { symbols, expression } = recognizeStrokes(
        targetStrokes.map((s) => s.points),
        model
      );
      const solved = solve(expression);

      let minX = Infinity,
        minY = Infinity,
        maxX = -Infinity,
        maxY = -Infinity;
      for (const sym of symbols) {
        minX = Math.min(minX, sym.box.x);
        minY = Math.min(minY, sym.box.y);
        maxX = Math.max(maxX, sym.box.x + sym.box.width);
        maxY = Math.max(maxY, sym.box.y + sym.box.height);
      }

      const data: MathRecognitionResult = {
        latex: solved.latex,
        readableText: solved.readableText,
        symbolsDetected: symbols.map((s) => displayChar(s.char)),
        symbols,
        isSolvable: solved.kind === 'evaluation' || solved.kind === 'equation',
        solution: solved.solution,
        autoAnswer: solved.answer,
        steps: solved.steps,
        isGraphable: solved.kind === 'function',
        graphExpr: solved.graphExpr,
        graphTitle: solved.kind === 'function' ? solved.readableText : undefined,
        error: solved.error,
        rawBoundingBox: { x: minX, y: minY, width: maxX - minX, height: maxY - minY },
      };
      setRecognitionResult(data);

      // Auto-solve on '=': write the answer right after the equal sign, like a notebook would
      if (solved.kind === 'evaluation' && solved.answer && expression.endsWith('=')) {
        const lineHeight = Math.max(20, maxY - minY);
        const answerStrokes: Stroke[] = textToStrokes(solved.answer, maxX + lineHeight * 0.3, minY, lineHeight).map(
          (points, i) => ({
            id: `answer-${Date.now()}-${i}`,
            points,
            color: '#2563eb',
            width: 3.5,
            tool: 'pen',
          })
        );
        pushToHistory([...strokes, ...answerStrokes]);
      }

      setAutoSolveToast(
        solved.error
          ? `Recognized "${expression}" but: ${solved.error}`
          : `Recognized: "${expression}"${solved.answer ? ` → ${solved.answer}` : ''}`
      );
      setTimeout(() => setAutoSolveToast(null), 4000);

      if (solved.kind === 'function' && solved.graphExpr) {
        setCards((prev) => [
          ...prev,
          {
            id: `graph-${Date.now()}`,
            x: maxX + 40,
            y: minY,
            width: 320,
            height: 220,
            type: 'graph-plot',
            title: solved.readableText,
            graphExpr: solved.graphExpr,
          },
        ]);
      }
    } catch (err: any) {
      console.error('Recognition error:', err);
      setRecognitionResult({
        latex: '',
        readableText: '',
        symbolsDetected: [],
        isSolvable: false,
        error: err?.message || String(err),
      });
    } finally {
      setIsRecognizing(false);
    }
  };

  // Insert Sticky KaTeX Math Card onto canvas
  const handleInsertLatexCard = (latexCode: string) => {
    const centerScreen = {
      x: (containerRef.current?.clientWidth || 800) / 2,
      y: (containerRef.current?.clientHeight || 600) / 2,
    };
    const worldPt = screenToWorld(centerScreen.x, centerScreen.y);

    const newCard: MathCanvasCard = {
      id: `latex-card-${Date.now()}`,
      x: worldPt.x - 120,
      y: worldPt.y - 60,
      width: 260,
      height: 120,
      type: 'math-solution',
      latex: latexCode,
    };
    setCards((prev) => [...prev, newCard]);
  };

  // Write an example expression on the canvas with the built-in handwriting glyphs
  const handleLoadMathPreset = (text: string) => {
    const origin = screenToWorld(300, 220);
    const sampleStrokes: Stroke[] = textToStrokes(text, origin.x, origin.y, 48).map((points, i) => ({
      id: `preset-${Date.now()}-${i}`,
      points,
      color: '#0f172a',
      width: 3.5,
      tool: 'pen',
    }));
    pushToHistory([...strokes, ...sampleStrokes]);
  };

  // Export Canvas to PNG
  const handleExportPNG = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const link = document.createElement('a');
    link.download = `NeuroScribe-Notes-${Date.now()}.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
  };

  const handleClearAll = () => {
    if (strokes.length === 0 && cards.length === 0) return;
    if (confirm('Clear all drawings and math cards on the infinite canvas?')) {
      pushToHistory([]);
      setCards([]);
      setSelectedStrokeIds([]);
    }
  };

  return (
    <div className="relative w-full h-[calc(100vh-61px)] overflow-hidden flex flex-col bg-slate-100 select-none">
      {/* Top toolbars: one wrapping row so they never overlap on narrow screens */}
      <div className="absolute top-3 inset-x-4 z-20 flex flex-wrap items-start justify-center gap-2 pointer-events-none">
        {/* Top Floating Toolbar (OneNote Ribbon style) */}
        <div className="pointer-events-auto bg-white/95 backdrop-blur-md border border-slate-200 shadow-md rounded-xl px-3 py-2 flex items-center gap-3">
          {/* Tool Selector */}
          <div className="flex items-center gap-1 p-0.5 bg-slate-100 rounded-lg">
            <button
              onClick={() => setCurrentTool('pen')}
              className={`p-2 rounded-md transition-colors ${
                currentTool === 'pen'
                  ? 'bg-white text-slate-900 shadow-2xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
              title="Pen (Natural handwriting)"
            >
              <PenTool className="w-4 h-4" />
            </button>
            <button
              onClick={() => setCurrentTool('highlighter')}
              className={`p-2 rounded-md transition-colors ${
                currentTool === 'highlighter'
                  ? 'bg-white text-amber-600 shadow-2xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
              title="Highlighter (Semi-transparent)"
            >
              <Highlighter className="w-4 h-4" />
            </button>
            <button
              onClick={() => setCurrentTool('eraser')}
              className={`p-2 rounded-md transition-colors ${
                currentTool === 'eraser'
                  ? 'bg-white text-rose-600 shadow-2xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
              title="Stroke Eraser (Deletes entire stroke on contact)"
            >
              <Eraser className="w-4 h-4" />
            </button>
            <button
              onClick={() => setCurrentTool('lasso')}
              className={`p-2 rounded-md transition-colors ${
                currentTool === 'lasso'
                  ? 'bg-white text-indigo-600 shadow-2xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
              title="Lasso Selection (Circle math strokes to recognize or move)"
            >
              <Lasso className="w-4 h-4" />
            </button>
            <button
              onClick={() => setCurrentTool('hand')}
              className={`p-2 rounded-md transition-colors ${
                currentTool === 'hand'
                  ? 'bg-white text-slate-900 shadow-2xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
              title="Hand Pan Tool"
            >
              <Hand className="w-4 h-4" />
            </button>
          </div>

          <div className="h-5 w-px bg-slate-200" />

          {/* Color Palette */}
          <div className="flex items-center gap-1.5">
            {[
              { color: '#0f172a', name: 'Ink Black' },
              { color: '#2563eb', name: 'Royal Blue' },
              { color: '#059669', name: 'Emerald' },
              { color: '#dc2626', name: 'Crimson' },
              { color: '#7c3aed', name: 'Amethyst' },
              { color: '#d97706', name: 'Amber' },
            ].map((c) => (
              <button
                key={c.color}
                onClick={() => {
                  setPenColor(c.color);
                  if (currentTool === 'eraser' || currentTool === 'hand') setCurrentTool('pen');
                }}
                style={{ backgroundColor: c.color }}
                title={c.name}
                className={`w-5 h-5 rounded-full transition-transform ${
                  penColor === c.color ? 'scale-125 ring-2 ring-slate-400 ring-offset-1' : 'hover:scale-110'
                }`}
              />
            ))}
          </div>

          <div className="h-5 w-px bg-slate-200" />

          {/* Stroke Thickness */}
          <div className="flex items-center gap-1">
            {[
              { size: 2.2, label: 'Fine' },
              { size: 3.8, label: 'Medium' },
              { size: 6.5, label: 'Bold' },
            ].map((t) => (
              <button
                key={t.label}
                onClick={() => setPenWidth(t.size)}
                className={`px-2 py-1 text-xs rounded transition-colors ${
                  penWidth === t.size
                    ? 'bg-slate-200 text-slate-900 font-semibold'
                    : 'text-slate-600 hover:bg-slate-100'
                }`}
              >
                {t.label}
              </button>
            ))}
          </div>

          <div className="h-5 w-px bg-slate-200" />

          {/* Math OCR & Solve Action Button */}
          <button
            onClick={handleRecognizeMath}
            disabled={isRecognizing}
            className="flex items-center gap-1.5 px-3.5 py-1.5 bg-slate-900 text-white rounded-lg text-xs font-semibold hover:bg-slate-800 transition-all shadow-xs hover:shadow"
            title="Convert handwriting to LaTeX and compute step-by-step solutions"
          >
            {isRecognizing ? (
              <>
                <div className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                <span>Recognizing...</span>
              </>
            ) : (
              <>
                <Calculator className="w-3.5 h-3.5" />
                <span>Recognize & Solve Math</span>
              </>
            )}
          </button>
        </div>

        {/* Top Right Quick Settings (Paper Styles, Preset Formulas, Export) */}
        <div className="pointer-events-auto flex items-center gap-2">
          {/* Preset Math equations drop-in */}
          <div className="bg-white/95 backdrop-blur-md border border-slate-200 shadow-xs rounded-xl p-1 flex items-center gap-1">
            <span className="text-[11px] font-medium text-slate-500 px-2">Presets:</span>
            {[
              { text: '24+16=', label: '24 + 16 =' },
              { text: '2x+3=11', label: '2x + 3 = 11' },
              { text: 'x^2-4', label: 'x² - 4' },
            ].map((preset) => (
              <button
                key={preset.text}
                onClick={() => handleLoadMathPreset(preset.text)}
                className="px-2 py-1 text-xs bg-slate-100 hover:bg-slate-200 rounded text-slate-700 transition-colors"
                title={`Write '${preset.label}' by hand on the canvas`}
              >
                {preset.label}
              </button>
            ))}
          </div>

          {/* Paper style selector */}
          <div className="bg-white/95 backdrop-blur-md border border-slate-200 shadow-xs rounded-xl p-1 flex items-center gap-1">
            <button
              onClick={() => setPaperStyle('lined')}
              className={`p-1.5 rounded-md ${paperStyle === 'lined' ? 'bg-slate-200 text-slate-900' : 'text-slate-500 hover:bg-slate-100'}`}
              title="Lined Notebook Paper"
            >
              <FileText className="w-4 h-4" />
            </button>
            <button
              onClick={() => setPaperStyle('grid')}
              className={`p-1.5 rounded-md ${paperStyle === 'grid' ? 'bg-slate-200 text-slate-900' : 'text-slate-500 hover:bg-slate-100'}`}
              title="Math 5mm Grid Paper"
            >
              <Grid className="w-4 h-4" />
            </button>
            <button
              onClick={() => setPaperStyle('dots')}
              className={`p-1.5 rounded-md ${paperStyle === 'dots' ? 'bg-slate-200 text-slate-900' : 'text-slate-500 hover:bg-slate-100'}`}
              title="Dot Grid Paper"
            >
              <CircleDot className="w-4 h-4" />
            </button>
          </div>

          {/* Export / Clear */}
          <div className="bg-white/95 backdrop-blur-md border border-slate-200 shadow-xs rounded-xl p-1 flex items-center gap-1">
            <button
              onClick={handleExportPNG}
              className="p-1.5 text-slate-600 hover:text-slate-900 rounded hover:bg-slate-100"
              title="Export note to PNG"
            >
              <Download className="w-4 h-4" />
            </button>
            <button
              onClick={handleClearAll}
              className="p-1.5 text-slate-500 hover:text-rose-600 rounded hover:bg-slate-100"
              title="Clear canvas"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Bottom Floating Canvas Controls (Undo, Redo, Zoom, Coordinates) */}
      <div className="absolute bottom-4 left-4 z-20 flex items-center gap-2">
        <div className="bg-white/95 backdrop-blur-md border border-slate-200 shadow-xs rounded-xl p-1 flex items-center gap-1 text-xs text-slate-600">
          <button
            onClick={handleUndo}
            disabled={historyIndex < 0}
            className="p-1.5 rounded hover:bg-slate-100 disabled:opacity-30 disabled:hover:bg-transparent"
            title="Undo (Ctrl+Z)"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
          <button
            onClick={handleRedo}
            disabled={historyIndex >= history.length - 1}
            className="p-1.5 rounded hover:bg-slate-100 disabled:opacity-30 disabled:hover:bg-transparent"
            title="Redo (Ctrl+Y)"
          >
            <RotateCw className="w-4 h-4" />
          </button>
          <div className="h-4 w-px bg-slate-200 mx-1" />
          <button
            onClick={() =>
              setViewTransform((prev) => ({
                ...prev,
                zoom: Math.max(0.25, prev.zoom - 0.2),
              }))
            }
            className="p-1.5 rounded hover:bg-slate-100"
            title="Zoom Out"
          >
            <ZoomOut className="w-4 h-4" />
          </button>
          <button
            onClick={() => setViewTransform((prev) => ({ ...prev, zoom: 1.0 }))}
            className="px-2 py-1 font-mono font-medium text-slate-700 hover:bg-slate-100 rounded"
            title="Reset Zoom to 100%"
          >
            {Math.round(viewTransform.zoom * 100)}%
          </button>
          <button
            onClick={() =>
              setViewTransform((prev) => ({
                ...prev,
                zoom: Math.min(3.0, prev.zoom + 0.2),
              }))
            }
            className="p-1.5 rounded hover:bg-slate-100"
            title="Zoom In"
          >
            <ZoomIn className="w-4 h-4" />
          </button>
          <div className="h-4 w-px bg-slate-200 mx-1" />
          <button
            onClick={() => setViewTransform({ x: 100, y: 100, zoom: 1.0 })}
            className="p-1.5 rounded hover:bg-slate-100"
            title="Center Viewport"
          >
            <Maximize2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Auto-Solve Toast Notification (Apple Math Notes style) */}
      {autoSolveToast && (
        <div className="absolute top-18 left-1/2 -translate-x-1/2 z-30 bg-emerald-900 text-white text-xs px-4 py-2 rounded-xl shadow-lg flex items-center gap-2 animate-in fade-in slide-in-from-top-2">
          <Sparkles className="w-4 h-4 text-emerald-300" />
          <span>{autoSolveToast}</span>
        </div>
      )}

      {/* Canvas Viewport */}
      <div ref={containerRef} className="relative flex-1 w-full h-full overflow-hidden">
        <canvas
          ref={canvasRef}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onWheel={handleWheel}
          className={`w-full h-full block touch-none ${
            currentTool === 'hand'
              ? 'cursor-grab active:cursor-grabbing'
              : currentTool === 'lasso'
              ? 'cursor-crosshair'
              : currentTool === 'eraser'
              ? 'cursor-pointer'
              : 'cursor-crosshair'
          }`}
        />

        {/* Embedded Interactive 2D Function Cards & Sticky LaTeX Cards */}
        {cards.map((card) => {
          const screenPos = worldToScreen(card.x, card.y);

          if (card.type === 'graph-plot' && card.graphExpr) {
            return (
              <div
                key={card.id}
                className="absolute z-10"
                style={{
                  left: `${screenPos.x}px`,
                  top: `${screenPos.y}px`,
                  transform: `scale(${Math.min(1.2, Math.max(0.6, viewTransform.zoom))})`,
                  transformOrigin: 'top left',
                }}
              >
                <FunctionGraphCard
                  id={card.id}
                  x={card.x}
                  y={card.y}
                  width={card.width}
                  height={card.height}
                  title={card.title || 'Function Graph'}
                  expr={card.graphExpr}
                  onClose={() => setCards((prev) => prev.filter((c) => c.id !== card.id))}
                />
              </div>
            );
          }

          if (card.type === 'math-solution' && card.latex) {
            return (
              <div
                key={card.id}
                className="absolute z-10 bg-white/95 backdrop-blur-md rounded-xl p-4 border border-slate-200 shadow-md min-w-[200px]"
                style={{
                  left: `${screenPos.x}px`,
                  top: `${screenPos.y}px`,
                  transform: `scale(${Math.min(1.2, Math.max(0.6, viewTransform.zoom))})`,
                  transformOrigin: 'top left',
                }}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">
                    Formula Card
                  </span>
                  <button
                    onClick={() => setCards((prev) => prev.filter((c) => c.id !== card.id))}
                    className="text-slate-400 hover:text-rose-500 text-xs"
                  >
                    ×
                  </button>
                </div>
                <div
                  className="text-slate-900 font-serif text-lg text-center py-1"
                  dangerouslySetInnerHTML={{
                    __html: (() => {
                      try {
                        return katex.renderToString(card.latex, { throwOnError: false });
                      } catch {
                        return card.latex;
                      }
                    })(),
                  }}
                />
              </div>
            );
          }

          return null;
        })}
      </div>

      {/* Math Result & Step-by-Step Solving Modal */}
      <MathResultModal
        isOpen={isModalOpen}
        result={recognitionResult}
        isLoading={isRecognizing}
        onClose={() => setIsModalOpen(false)}
        onInsertLatexCard={handleInsertLatexCard}
      />
    </div>
  );
};
