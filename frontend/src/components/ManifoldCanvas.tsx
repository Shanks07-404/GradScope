import React, { useRef, useEffect, useState } from 'react';
import { ManifoldPoint } from '../api/socket';
import { getInterpolatedLayerCoords } from '../utils/interpolate';
import { LayerScrubber } from './LayerScrubber';

interface ManifoldCanvasProps {
  points: ManifoldPoint[];
  numLayers: number;
  layerLabels: string[];
}

export const ManifoldCanvas: React.FC<ManifoldCanvasProps> = ({
  points,
  numLayers,
  layerLabels,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [scrubPosition, setScrubPosition] = useState<number>(0);
  const animFrameRef = useRef<number | null>(null);

  // Redraw canvas whenever points or scrubPosition changes
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Handle high DPI displays
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;

    // 1. Draw Chalkboard background & subtle grid
    ctx.fillStyle = '#0E1512';
    ctx.fillRect(0, 0, w, h);

    const gridSize = 28;
    ctx.strokeStyle = '#2A332E';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = 0; x < w; x += gridSize) {
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
    }
    for (let y = 0; y < h; y += gridSize) {
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
    }
    ctx.stroke();

    // 2. Draw coordinate axes (chalk-white faint lines)
    const cx = w / 2;
    const cy = h / 2;

    ctx.strokeStyle = 'rgba(237, 234, 224, 0.2)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    // X axis
    ctx.moveTo(20, cy);
    ctx.lineTo(w - 20, cy);
    // Y axis
    ctx.moveTo(cx, 20);
    ctx.lineTo(cx, h - 20);
    ctx.stroke();

    // Subtle axis labels
    ctx.fillStyle = '#8FA69D';
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillText('0', cx + 4, cy + 12);
    ctx.fillText('+1.5', w - 45, cy + 14);
    ctx.fillText('-1.5', 24, cy + 14);
    ctx.fillText('+1.5', cx + 6, 32);
    ctx.fillText('-1.5', cx + 6, h - 24);

    // Coordinate scaling: range roughly [-2.0, 2.0] maps to [40, w - 40]
    const scale = Math.min(w, h) / 4.4;

    // 3. Render Data Points
    if (points && points.length > 0) {
      for (const pt of points) {
        if (!pt.layers || pt.layers.length === 0) continue;

        const [xVal, yVal] = getInterpolatedLayerCoords(pt.layers, scrubPosition);

        const screenX = cx + xVal * scale;
        const screenY = cy - yVal * scale; // inverted Y for screen coords

        // Color based on class: Class 0 = Amber (#F2B84B), Class 1 = Teal (#4C9A8E)
        const isClass0 = pt.label === 0;
        const baseColor = isClass0 ? '#F2B84B' : '#4C9A8E';

        // Chalkboard point rendering: faint glow + solid chalk core
        ctx.beginPath();
        ctx.arc(screenX, screenY, 4.5, 0, Math.PI * 2);
        ctx.fillStyle = baseColor;
        ctx.shadowColor = baseColor;
        ctx.shadowBlur = 6;
        ctx.fill();

        // Reset shadow
        ctx.shadowBlur = 0;

        // Inner chalk-white center for contrast
        ctx.beginPath();
        ctx.arc(screenX, screenY, 1.5, 0, Math.PI * 2);
        ctx.fillStyle = '#EDEAE0';
        ctx.fill();
      }
    }

    // 4. Decision boundary hint if near output layer
    const isNearOutput = scrubPosition >= numLayers - 1.2;
    if (isNearOutput) {
      // At the output layer, boundary is at x = 0 (decision threshold)
      ctx.save();
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = '#F2B84B';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(cx, 30);
      ctx.lineTo(cx, h - 30);
      ctx.stroke();
      ctx.fillStyle = '#F2B84B';
      ctx.font = '11px "JetBrains Mono", monospace';
      ctx.fillText('Decision Threshold (0.0)', cx + 8, 48);
      ctx.restore();
    }

  }, [points, scrubPosition, numLayers]);

  return (
    <div className="flex flex-col h-full bg-chalkboard-bg border-l border-chalkboard-grid select-none relative overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-chalkboard-grid bg-chalkboard-card/30">
        <div className="flex items-center gap-2">
          <span className="font-serif font-semibold text-chalkboard-chalk">
            Manifold View
          </span>
          <span className="text-[11px] font-mono text-chalkboard-sage">
            (Layer-by-Layer Untangling)
          </span>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-4 text-xs font-mono">
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-chalkboard-amber shadow-[0_0_6px_#F2B84B]" />
            <span className="text-chalkboard-sage">Class 0</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-chalkboard-teal shadow-[0_0_6px_#4C9A8E]" />
            <span className="text-chalkboard-sage">Class 1</span>
          </div>
        </div>
      </div>

      {/* Canvas */}
      <div className="flex-1 relative w-full h-full">
        <canvas
          ref={canvasRef}
          className="w-full h-full block"
        />
      </div>

      {/* Layer Scrubber */}
      <LayerScrubber
        numLayers={numLayers}
        layerLabels={layerLabels}
        scrubPosition={scrubPosition}
        onScrubChange={setScrubPosition}
      />
    </div>
  );
};
