import { useState, useEffect, useCallback, useRef } from 'react';
import {
  socketService,
  TrainingConfig,
  TrainingSnapshot
} from '../api/socket';

export interface LossPoint {
  step: number;
  loss: number;
  accuracy: number;
}

const DEFAULT_CONFIG: TrainingConfig = {
  dataset: 'spirals',
  noise: 0.08,
  n_points: 100,
  hidden_dims: [2, 2],
  activation: 'tanh',
  learning_rate: 0.1,
  step_delay_ms: 60,
};

export function useTrainingStream() {
  const [connected, setConnected] = useState(false);
  const [snapshot, setSnapshot] = useState<TrainingSnapshot | null>(null);
  const [config, setConfig] = useState<TrainingConfig>(DEFAULT_CONFIG);
  const [lossHistory, setLossHistory] = useState<LossPoint[]>([]);
  const isMountedRef = useRef(true);

  useEffect(() => {
    isMountedRef.current = true;
    socketService.connect();

    const unsubStatus = socketService.onStatus((status) => {
      if (isMountedRef.current) setConnected(status);
    });

    const unsubSnapshot = socketService.onSnapshot((snap) => {
      if (!isMountedRef.current) return;
      setSnapshot(snap);

      setLossHistory((prev) => {
        // If training was reset (step 0 or decreased step), start fresh history
        if (snap.step === 0 || (prev.length > 0 && snap.step < prev[prev.length - 1].step)) {
          return [{ step: snap.step, loss: snap.loss, accuracy: snap.accuracy }];
        }
        // If same step snapshot (e.g. paused or config updated), update in place
        if (prev.length > 0 && snap.step === prev[prev.length - 1].step) {
          const updated = [...prev];
          updated[updated.length - 1] = { step: snap.step, loss: snap.loss, accuracy: snap.accuracy };
          return updated;
        }
        // Otherwise append new step progression (capped to 150 points)
        const next = [...prev, { step: snap.step, loss: snap.loss, accuracy: snap.accuracy }];
        return next.length > 150 ? next.slice(next.length - 150) : next;
      });
    });

    return () => {
      isMountedRef.current = false;
      unsubStatus();
      unsubSnapshot();
    };
  }, []);

  const play = useCallback(() => {
    socketService.sendCommand('start');
  }, []);

  const pause = useCallback(() => {
    socketService.sendCommand('pause');
  }, []);

  const stepOnce = useCallback(() => {
    socketService.sendCommand('step');
  }, []);

  const reset = useCallback(() => {
    setLossHistory([]);
    socketService.sendCommand('reset');
  }, []);

  const updateConfig = useCallback((newConfig: Partial<TrainingConfig>) => {
    setConfig((prev) => {
      const merged = { ...prev, ...newConfig };
      socketService.sendCommand('config', merged);
      return merged;
    });
  }, []);

  return {
    connected,
    snapshot,
    config,
    lossHistory,
    play,
    pause,
    stepOnce,
    reset,
    updateConfig,
  };
}
