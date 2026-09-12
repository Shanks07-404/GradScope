/**
 * WebSocket client for GradScope training stream.
 */

export interface TrainingConfig {
  dataset: 'spirals' | 'moons' | 'xor';
  noise: number;
  n_points: number;
  hidden_dims: number[];
  activation: string;
  learning_rate: number;
  step_delay_ms: number;
}

export interface GraphNode {
  id: string;
  label: string;
  value: number;
  grad: number;
  op: string;
  depth: number;
}

export interface GraphEdge {
  source: string;
  target: string;
  grad_flow: number;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface ManifoldPoint {
  id: number;
  label: number; // 0 or 1
  layers: [number, number][]; // [x, y] coordinates for each layer
}

export interface TrainingSnapshot {
  type: 'snapshot';
  step: number;
  epoch: number;
  loss: number;
  accuracy: number;
  graph: GraphData;
  manifold: ManifoldPoint[];
  num_layers: number;
  layer_labels: string[];
  is_running: boolean;
}

export type SnapshotListener = (snapshot: TrainingSnapshot) => void;
export type StatusListener = (connected: boolean) => void;

export class TrainingSocket {
  private ws: WebSocket | null = null;
  private url: string;
  private snapshotListeners: Set<SnapshotListener> = new Set();
  private statusListeners: Set<StatusListener> = new Set();
  private shouldReconnect = true;
  private reconnectTimer: any = null;

  constructor(url?: string) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.hostname || 'localhost';
    this.url = url || `${protocol}//${host}:8000/ws/train`;
  }

  public connect(): void {
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    try {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        this.notifyStatus(true);
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'snapshot') {
            this.notifySnapshot(data as TrainingSnapshot);
          }
        } catch (err) {
          console.error('Error parsing snapshot message:', err);
        }
      };

      this.ws.onclose = () => {
        this.notifyStatus(false);
        if (this.shouldReconnect) {
          this.scheduleReconnect();
        }
      };

      this.ws.onerror = () => {
        this.notifyStatus(false);
      };
    } catch (err) {
      console.error('WebSocket connection error:', err);
      this.scheduleReconnect();
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer) return;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, 1500);
  }

  public disconnect(): void {
    this.shouldReconnect = false;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  public sendCommand(action: 'start' | 'pause' | 'step' | 'reset' | 'config', config?: TrainingConfig): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.warn('Cannot send command: WebSocket is not connected.');
      return;
    }
    this.ws.send(JSON.stringify({ action, config }));
  }

  public onSnapshot(cb: SnapshotListener): () => void {
    this.snapshotListeners.add(cb);
    return () => this.snapshotListeners.delete(cb);
  }

  public onStatus(cb: StatusListener): () => void {
    this.statusListeners.add(cb);
    return () => this.statusListeners.delete(cb);
  }

  private notifySnapshot(snapshot: TrainingSnapshot): void {
    this.snapshotListeners.forEach((cb) => cb(snapshot));
  }

  private notifyStatus(connected: boolean): void {
    this.statusListeners.forEach((cb) => cb(connected));
  }
}

export const socketService = new TrainingSocket();
