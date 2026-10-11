import React, { useRef, useEffect, useState, useMemo } from 'react';
import { X, ZoomIn, ZoomOut, Move } from 'lucide-react';
import { compileFunction } from '../ml/mathParser';

interface FunctionGraphCardProps {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  title: string;
  expr: string; // expression in x, e.g. "x^2-4"
  onClose: () => void;
}

export const FunctionGraphCard: React.FC<FunctionGraphCardProps> = ({
  id,
  x,
  y,
  width = 340,
  height = 240,
  title,
  expr,
  onClose,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [domain, setDomain] = useState<[number, number]>([-10, 10]);
  const [hoverCoord, setHoverCoord] = useState<{ x: number; y: number } | null>(null);

  // Parse once with our own math parser (no eval), then evaluate point by point
  const fn = useMemo(() => {
    try {
      return compileFunction(expr);
    } catch {
      return null;
    }
  }, [expr]);

  const evalFunc = (xVal: number): number | null => {
    if (!fn) return null;
    try {
      const res = fn(xVal);
      return Number.isFinite(res) ? res : null;
    } catch {
      return null;
    }
  };

  // Render graph
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;

    // Background
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, w, h);

    const [xMin, xMax] = domain;
    const xRange = xMax - xMin;
    const yMin = -xRange / 2;
    const yMax = xRange / 2;
    const yRange = yMax - yMin;

    const toScreenX = (xVal: number) => ((xVal - xMin) / xRange) * w;
    const toScreenY = (yVal: number) => h - ((yVal - yMin) / yRange) * h;

    // Grid lines
    ctx.strokeStyle = '#f1f5f9';
    ctx.lineWidth = 1;

    const step = Math.max(1, Math.round(xRange / 10));
    for (let gx = Math.ceil(xMin / step) * step; gx <= xMax; gx += step) {
      const sx = toScreenX(gx);
      ctx.beginPath();
      ctx.moveTo(sx, 0);
      ctx.lineTo(sx, h);
      ctx.stroke();
    }
    for (let gy = Math.ceil(yMin / step) * step; gy <= yMax; gy += step) {
      const sy = toScreenY(gy);
      ctx.beginPath();
      ctx.moveTo(0, sy);
      ctx.lineTo(w, sy);
      ctx.stroke();
    }

    // Axes
    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 1.5;

    // X Axis
    const originY = toScreenY(0);
    ctx.beginPath();
    ctx.moveTo(0, originY);
    ctx.lineTo(w, originY);
    ctx.stroke();

    // Y Axis
    const originX = toScreenX(0);
    ctx.beginPath();
    ctx.moveTo(originX, 0);
    ctx.lineTo(originX, h);
    ctx.stroke();

    // Plot Function curve
    ctx.strokeStyle = '#2563eb';
    ctx.lineWidth = 2.5;
    ctx.lineJoin = 'round';
    ctx.beginPath();

    let started = false;
    const sampleCount = w;
    for (let px = 0; px <= sampleCount; px++) {
      const xVal = xMin + (px / sampleCount) * xRange;
      const yVal = evalFunc(xVal);

      if (yVal !== null && yVal >= yMin - 10 && yVal <= yMax + 10) {
        const sy = toScreenY(yVal);
        if (!started) {
          ctx.moveTo(px, sy);
          started = true;
        } else {
          ctx.lineTo(px, sy);
        }
      } else {
        started = false;
      }
    }
    ctx.stroke();

    // Draw hover point if present
    if (hoverCoord) {
      const sx = toScreenX(hoverCoord.x);
      const sy = toScreenY(hoverCoord.y);
      ctx.fillStyle = '#ef4444';
      ctx.beginPath();
      ctx.arc(sx, sy, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [domain, fn, hoverCoord]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const [xMin, xMax] = domain;
    const xVal = xMin + (px / rect.width) * (xMax - xMin);
    const yVal = evalFunc(xVal);

    if (yVal !== null) {
      setHoverCoord({ x: Math.round(xVal * 100) / 100, y: Math.round(yVal * 100) / 100 });
    }
  };

  const handleZoom = (factor: number) => {
    setDomain(([min, max]) => [min * factor, max * factor]);
  };

  return (
    <div
      className="bg-white rounded-xl shadow-lg border border-slate-200 overflow-hidden select-none"
      style={{ width: `${width}px` }}
    >
      <div className="flex items-center justify-between px-3 py-2 bg-slate-50 border-b border-slate-200">
        <div className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-blue-500"></span>
          <span className="text-xs font-semibold text-slate-800 truncate max-w-[180px]">
            {title || 'Function Plot'}
          </span>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={() => handleZoom(0.8)}
            className="p-1 text-slate-500 hover:text-slate-800 rounded hover:bg-slate-200/60"
            title="Zoom in"
          >
            <ZoomIn className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => handleZoom(1.25)}
            className="p-1 text-slate-500 hover:text-slate-800 rounded hover:bg-slate-200/60"
            title="Zoom out"
          >
            <ZoomOut className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={onClose}
            className="p-1 text-slate-400 hover:text-rose-600 rounded hover:bg-slate-200/60"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      <div className="relative">
        <canvas
          ref={canvasRef}
          width={width}
          height={height}
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setHoverCoord(null)}
          className="w-full h-auto cursor-crosshair block"
        />
        {hoverCoord && (
          <div className="absolute bottom-2 left-2 bg-slate-900/80 text-white text-[10px] font-mono px-2 py-0.5 rounded backdrop-blur-xs">
            x: {hoverCoord.x}, y: {hoverCoord.y}
          </div>
        )}
      </div>
    </div>
  );
};
