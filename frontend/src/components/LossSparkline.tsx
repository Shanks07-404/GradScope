import React from 'react';
import { LossPoint } from '../hooks/useTrainingStream';

interface LossSparklineProps {
  history: LossPoint[];
  currentLoss: number | null;
  currentAccuracy: number | null;
  step: number;
  connected: boolean;
  isRunning?: boolean;
}

export const LossSparkline: React.FC<LossSparklineProps> = ({
  history,
  currentLoss,
  currentAccuracy,
  step,
  connected,
  isRunning = false,
}) => {
  // Compute SVG polyline coordinates for heartbeat look
  const width = 320;
  const height = 36;
  const padding = 3;

  let pointsStr = '';
  if (history.length > 1) {
    const losses = history.map((h) => h.loss);
    const minLoss = Math.min(...losses, 0);
    const maxLoss = Math.max(...losses, 1.0);
    const range = maxLoss - minLoss || 1;

    pointsStr = history
      .map((pt, i) => {
        const x = padding + (i / (history.length - 1)) * (width - 2 * padding);
        const y =
          height -
          padding -
          ((pt.loss - minLoss) / range) * (height - 2 * padding);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');
  }

  const getStatusText = () => {
    if (!connected) return 'Connecting...';
    if (isRunning) return 'Training...';
    if (step > 0) return `Paused (Step ${step})`;
    return 'Ready';
  };

  return (
    <div className="flex items-center gap-6 px-4 py-2 border-b border-chalkboard-grid bg-chalkboard-bg text-chalkboard-chalk select-none">
      {/* Brand / Logo */}
      <div className="flex items-center gap-2">
        <span className="font-serif text-xl tracking-tight text-chalkboard-chalk font-semibold">
          GradScope
        </span>
        <span className="text-[10px] uppercase font-mono tracking-widest text-chalkboard-amber border border-chalkboard-amber/40 px-1.5 py-0.5 rounded-sm">
          Autograd
        </span>
      </div>

      {/* State Indicator */}
      <div className="flex items-center gap-1.5 text-xs font-mono text-chalkboard-sage">
        <span
          className={`w-2 h-2 rounded-full transition-colors ${
            !connected
              ? 'bg-red-500'
              : isRunning
              ? 'bg-chalkboard-teal shadow-[0_0_6px_#4C9A8E] animate-pulse'
              : step > 0
              ? 'bg-chalkboard-amber shadow-[0_0_6px_#F2B84B]'
              : 'bg-chalkboard-sage'
          }`}
        />
        <span className={isRunning ? 'text-chalkboard-teal' : step > 0 ? 'text-chalkboard-amber' : 'text-chalkboard-sage'}>
          {getStatusText()}
        </span>
      </div>

      {/* Heartbeat Loss Sparkline */}
      <div className="flex-1 flex items-center justify-center gap-4 max-w-xl mx-auto">
        <span className="text-xs font-mono text-chalkboard-sage">Loss Heartbeat</span>
        <div className="relative w-80 h-9 border border-chalkboard-grid/60 bg-chalkboard-card/50 rounded-sm overflow-hidden flex items-center px-1">
          {history.length > 1 ? (
            <svg
              className="w-full h-full overflow-visible"
              viewBox={`0 0 ${width} ${height}`}
              preserveAspectRatio="none"
            >
              <defs>
                <linearGradient id="lossGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#F2B84B" stopOpacity="0.3" />
                  <stop offset="100%" stopColor="#F2B84B" stopOpacity="0.0" />
                </linearGradient>
              </defs>
              <polyline
                fill="none"
                stroke="#F2B84B"
                strokeWidth="1.75"
                strokeLinecap="round"
                strokeLinejoin="round"
                points={pointsStr}
              />
            </svg>
          ) : (
            <div className="w-full text-center text-[11px] font-mono text-chalkboard-sage/50">
              Awaiting step pulses...
            </div>
          )}
        </div>
      </div>

      {/* Metrics Readout */}
      <div className="flex items-center gap-5 text-xs font-mono">
        <div>
          <span className="text-chalkboard-sage">Step: </span>
          <span className="text-chalkboard-chalk font-semibold">{step}</span>
        </div>
        <div>
          <span className="text-chalkboard-sage">Loss: </span>
          <span className="text-chalkboard-amber font-semibold">
            {currentLoss !== null ? currentLoss.toFixed(4) : '—'}
          </span>
        </div>
        <div className="flex items-center gap-1.5 bg-chalkboard-grid/40 px-2 py-0.5 rounded border border-chalkboard-grid">
          <span className="text-chalkboard-sage">Acc:</span>
          <span className="text-chalkboard-teal font-semibold">
            {currentAccuracy !== null ? `${currentAccuracy.toFixed(1)}%` : '—'}
          </span>
        </div>
      </div>
    </div>
  );
};
