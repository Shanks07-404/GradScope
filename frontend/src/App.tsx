import React, { useEffect } from 'react';
import { useTrainingStream } from './hooks/useTrainingStream';
import { LossSparkline } from './components/LossSparkline';
import { GraphCanvas } from './components/GraphCanvas';
import { ManifoldCanvas } from './components/ManifoldCanvas';
import { ControlDock } from './components/ControlDock';

export const App: React.FC = () => {
  const {
    connected,
    snapshot,
    config,
    lossHistory,
    play,
    pause,
    stepOnce,
    reset,
    updateConfig,
  } = useTrainingStream();

  const isRunning = snapshot?.is_running ?? false;

  // Global keyboard shortcuts (Space to toggle train/pause, Right Arrow to step, R to reset)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't trigger if user is interacting with an input
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes((e.target as HTMLElement)?.tagName)) {
        return;
      }

      if (e.code === 'Space') {
        e.preventDefault();
        if (isRunning) pause();
        else play();
      } else if (e.code === 'ArrowRight') {
        e.preventDefault();
        stepOnce();
      } else if (e.key === 'r' || e.key === 'R') {
        e.preventDefault();
        reset();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isRunning, play, pause, stepOnce, reset]);

  return (
    <div className="flex flex-col h-screen w-screen bg-chalkboard-bg text-chalkboard-chalk font-mono overflow-hidden">
      {/* Top Bar: Heartbeat Loss Sparkline & Stats */}
      <LossSparkline
        history={lossHistory}
        currentLoss={snapshot?.loss ?? null}
        currentAccuracy={snapshot?.accuracy ?? null}
        step={snapshot?.step ?? 0}
        connected={connected}
        isRunning={isRunning}
      />

      {/* Main Two-Pane Visualizer */}
      <div className="flex-1 grid grid-cols-1 md:grid-cols-2 overflow-hidden">
        {/* Left Pane: Computational DAG with Gradient Flow */}
        <div className="h-full overflow-hidden border-r border-chalkboard-grid">
          <GraphCanvas
            graph={snapshot?.graph ?? null}
            step={snapshot?.step ?? 0}
          />
        </div>

        {/* Right Pane: Manifold Layer-by-Layer Untangling */}
        <div className="h-full overflow-hidden">
          <ManifoldCanvas
            points={snapshot?.manifold ?? []}
            numLayers={snapshot?.num_layers ?? 3}
            layerLabels={snapshot?.layer_labels ?? ['Input Space (2D)', 'Hidden 1 (2D)', 'Output Space (1D)']}
          />
        </div>
      </div>

      {/* Bottom Dock: Controls & Hyperparameters */}
      <ControlDock
        config={config}
        isRunning={isRunning}
        step={snapshot?.step ?? 0}
        onPlay={play}
        onPause={pause}
        onStep={stepOnce}
        onReset={reset}
        onUpdateConfig={updateConfig}
      />
    </div>
  );
};

export default App;
