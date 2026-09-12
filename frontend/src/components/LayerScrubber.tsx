import React from 'react';

interface LayerScrubberProps {
  numLayers: number;
  layerLabels: string[];
  scrubPosition: number;
  onScrubChange: (newPos: number) => void;
}

export const LayerScrubber: React.FC<LayerScrubberProps> = ({
  numLayers,
  layerLabels,
  scrubPosition,
  onScrubChange,
}) => {
  if (numLayers <= 1) return null;
  const maxLayer = numLayers - 1;

  return (
    <div className="flex flex-col gap-2 px-4 py-2.5 bg-chalkboard-card/40 border-t border-chalkboard-grid select-none">
      <div className="flex items-center justify-between text-xs font-mono">
        <span className="text-chalkboard-sage flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-chalkboard-amber" />
          Layer Manifold Scrubber
        </span>
        <span className="text-chalkboard-amber font-mono font-medium">
          {layerLabels[Math.round(scrubPosition)] || `Layer ${scrubPosition.toFixed(1)}`}
        </span>
      </div>

      {/* Scrubber slider track */}
      <div className="relative flex items-center">
        <input
          type="range"
          min={0}
          max={maxLayer}
          step={0.01}
          value={scrubPosition}
          onChange={(e) => onScrubChange(parseFloat(e.target.value))}
          className="w-full h-1 bg-chalkboard-grid rounded-lg appearance-none cursor-pointer accent-[#F2B84B] focus:outline-none"
        />
      </div>

      {/* Discrete snap buttons */}
      <div className="flex justify-between items-center text-[11px] font-mono text-chalkboard-sage">
        {layerLabels.map((label, idx) => {
          const isSelected = Math.abs(scrubPosition - idx) < 0.15;
          return (
            <button
              key={idx}
              onClick={() => onScrubChange(idx)}
              className={`transition-colors px-2 py-0.5 rounded cursor-pointer ${
                isSelected
                  ? 'text-chalkboard-amber font-semibold border-b border-chalkboard-amber'
                  : 'hover:text-chalkboard-chalk'
              }`}
            >
              {label}
            </button>
          );
        })}
      </div>
    </div>
  );
};
