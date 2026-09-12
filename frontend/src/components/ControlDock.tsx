import React from 'react';
import { Play, Pause, SkipForward, RotateCcw } from 'lucide-react';
import { TrainingConfig } from '../api/socket';

interface ControlDockProps {
  config: TrainingConfig;
  isRunning: boolean;
  step?: number;
  onPlay: () => void;
  onPause: () => void;
  onStep: () => void;
  onReset: () => void;
  onUpdateConfig: (newCfg: Partial<TrainingConfig>) => void;
}

export const ControlDock: React.FC<ControlDockProps> = ({
  config,
  isRunning,
  step = 0,
  onPlay,
  onPause,
  onStep,
  onReset,
  onUpdateConfig,
}) => {
  const datasets = [
    { id: 'spirals', name: 'Two Spirals' },
    { id: 'moons', name: 'Moons' },
    { id: 'xor', name: 'XOR' },
  ] as const;

  const archOptions = [
    { label: '[2, 2]', dims: [2, 2] },
    { label: '[4, 4]', dims: [4, 4] },
    { label: '[2, 2, 2]', dims: [2, 2, 2] },
  ];

  const isPaused = !isRunning && step > 0;

  return (
    <div className="flex flex-wrap items-center justify-between gap-4 px-5 py-3 border-t border-chalkboard-grid bg-chalkboard-bg text-chalkboard-chalk select-none">
      {/* Playback Controls */}
      <div className="flex items-center gap-2">
        {isRunning ? (
          <button
            onClick={onPause}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-chalkboard-amber text-chalkboard-bg font-mono text-xs font-semibold rounded hover:brightness-105 transition cursor-pointer"
          >
            <Pause size={14} />
            <span>Pause</span>
          </button>
        ) : (
          <button
            onClick={onPlay}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-chalkboard-amber text-chalkboard-bg font-mono text-xs font-semibold rounded hover:brightness-105 transition cursor-pointer shadow-[0_0_8px_rgba(242,184,75,0.25)]"
          >
            <Play size={14} />
            <span>{isPaused ? 'Resume' : 'Train'}</span>
          </button>
        )}

        <button
          onClick={onStep}
          disabled={isRunning}
          title="Step One Epoch"
          className="flex items-center gap-1 px-2.5 py-1.5 border border-chalkboard-grid hover:border-chalkboard-sage text-chalkboard-chalk font-mono text-xs rounded transition disabled:opacity-40 cursor-pointer"
        >
          <SkipForward size={14} />
          <span>Step</span>
        </button>

        <button
          onClick={onReset}
          title="Reset Weights & Data"
          className="flex items-center gap-1 px-2.5 py-1.5 border border-chalkboard-grid hover:border-chalkboard-sage text-chalkboard-sage hover:text-chalkboard-chalk font-mono text-xs rounded transition cursor-pointer"
        >
          <RotateCcw size={14} />
          <span>Reset</span>
        </button>
      </div>

      {/* Dataset Picker */}
      <div className="flex items-center gap-2">
        <span className="text-xs font-mono text-chalkboard-sage">Dataset:</span>
        <div className="flex items-center border border-chalkboard-grid rounded overflow-hidden">
          {datasets.map((d) => (
            <button
              key={d.id}
              onClick={() => onUpdateConfig({ dataset: d.id })}
              className={`px-3 py-1 text-xs font-mono transition cursor-pointer ${
                config.dataset === d.id
                  ? 'bg-chalkboard-card text-chalkboard-amber font-semibold border-b-2 border-chalkboard-amber'
                  : 'text-chalkboard-sage hover:text-chalkboard-chalk'
              }`}
            >
              {d.name}
            </button>
          ))}
        </div>
      </div>

      {/* Architecture Picker */}
      <div className="flex items-center gap-2">
        <span className="text-xs font-mono text-chalkboard-sage">Layers:</span>
        <div className="flex items-center border border-chalkboard-grid rounded overflow-hidden">
          {archOptions.map((a, i) => {
            const isSelected =
              JSON.stringify(config.hidden_dims) === JSON.stringify(a.dims);
            return (
              <button
                key={i}
                onClick={() => onUpdateConfig({ hidden_dims: a.dims })}
                className={`px-2.5 py-1 text-xs font-mono transition cursor-pointer ${
                  isSelected
                    ? 'bg-chalkboard-card text-chalkboard-amber font-semibold border-b-2 border-chalkboard-amber'
                    : 'text-chalkboard-sage hover:text-chalkboard-chalk'
                }`}
              >
                {a.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Sliders: LR, Noise, Delay */}
      <div className="flex items-center gap-5">
        {/* Learning Rate */}
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-chalkboard-sage">LR:</span>
          <input
            type="range"
            min={0.01}
            max={0.5}
            step={0.01}
            value={config.learning_rate}
            onChange={(e) =>
              onUpdateConfig({ learning_rate: parseFloat(e.target.value) })
            }
            className="w-18 h-1 bg-chalkboard-grid rounded appearance-none cursor-pointer accent-[#F2B84B]"
          />
          <span className="text-xs font-mono text-chalkboard-chalk w-8">
            {config.learning_rate.toFixed(2)}
          </span>
        </div>

        {/* Noise */}
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-chalkboard-sage">Noise:</span>
          <input
            type="range"
            min={0.0}
            max={0.25}
            step={0.01}
            value={config.noise}
            onChange={(e) =>
              onUpdateConfig({ noise: parseFloat(e.target.value) })
            }
            className="w-18 h-1 bg-chalkboard-grid rounded appearance-none cursor-pointer accent-[#F2B84B]"
          />
          <span className="text-xs font-mono text-chalkboard-chalk w-8">
            {config.noise.toFixed(2)}
          </span>
        </div>

        {/* Speed / Delay */}
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-chalkboard-sage">Speed:</span>
          <input
            type="range"
            min={10}
            max={300}
            step={10}
            value={310 - config.step_delay_ms} // invert so right = faster
            onChange={(e) =>
              onUpdateConfig({ step_delay_ms: 310 - parseInt(e.target.value) })
            }
            className="w-16 h-1 bg-chalkboard-grid rounded appearance-none cursor-pointer accent-[#F2B84B]"
          />
        </div>
      </div>
    </div>
  );
};
