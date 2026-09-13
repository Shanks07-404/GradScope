# GradScope

> A from-scratch reverse-mode autograd engine paired with a live, interactive visualization of how a neural network untangles 2D data manifolds layer-by-layer.

---

```
   ┌────────────────────────────────────────────────────────┐
   │ ∇ GradScope: Live Autograd & Manifold Visualizer       │
   ├────────────────────────────┬───────────────────────────┤
   │ Computational DAG          │ Manifold View (Morph)     │
   │  [x0] ──(*)──> (+) ──(tanh)│   ● Class 0 (Amber)       │
   │           ▲     │          │   ▲ Class 1 (Teal)        │
   │  [w0] ────┘     ▼          │                           │
   │  (Amber pulse on backward) │ [Layer 0] ──> [Layer L]   │
   ├────────────────────────────┴───────────────────────────┤
   │ Controls: Spirals / Moons / XOR │ LR │ Play / Pause    │
   └────────────────────────────────────────────────────────┘
```

## Why I Built This

Most machine learning practitioners rely on PyTorch or TensorFlow, treating `.backward()` as an opaque black box. At the same time, explanations of neural networks often use static 2D diagrams that fail to communicate the geometry of what deep representations actually achieve: **progressively untangling non-linear manifolds into linearly separable spaces**.

GradScope fuses two ideas into one unified project:
1. **Engine**: A tiny, pure Python reverse-mode automatic differentiation library built without PyTorch, NumPy, or any external dependencies. Every scalar operation (`+`, `*`, `tanh`, `relu`, `**`, etc.) remembers its parent nodes and registers its own local gradient closure. One call to `.backward()` topologically sorts the DAG and applies the multivariable chain rule to compute exact gradients.
2. **Visualizer**: An interactive, chalkboard-styled frontend that streams live full-batch training via WebSockets. As training runs, it animates:
   - **The computational DAG**, with traveling amber pulses illuminating the reverse flow of $\frac{\partial L}{\partial w}$ along graph edges.
   - **The data manifold**, letting you scrub through intermediate layers and watch intertwined classes (spirals, moons, XOR) stretch, fold, and untangle into separation.

---

## Tech Stack

### Core Engine (Pure Python 3.11+)
- **Zero dependencies beyond Python's standard library** (`math`, `random`, `typing`).
- Built from the mathematical ground up: `Value`, `Neuron`, `Layer`, `MLP`.
- Verified against numerical finite differences ($f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}$) with numerical error $< 10^{-5}$.

### Server & Streaming
- **FastAPI** ASGI server + WebSockets.
- Real-time bi-directional streaming (`/ws/train`): pushes training snapshots (loss, per-node gradients, per-point layer coordinates) every step.
- Full-batch gradient descent semantics: 1 training step = 1 complete pass over all points + 1 backward pass + 1 SGD parameter update.

### Interactive Frontend
- **React 18 + TypeScript + Vite**.
- **Chalkboard Notebook Design System**:
  - `#0E1512` — near-black chalkboard green background
  - `#EDEAE0` — chalk-white line strokes & text
  - `#F2B84B` — amber chalk for active gradient flow pulses & Class 0
  - `#4C9A8E` — muted teal for Class 1 (colorblind-distinguishable)
  - `#2A332E` — soft graph-paper grid lines
  - `#8FA69D` — dim sage secondary text
  - Typography: `Fraunces` (warm serif) & `JetBrains Mono` (crisp data readout).
- **Interactive SVG DAG**: Layered graph layout with animated SVG particle pulses flowing backwards during backprop.
- **2D Manifold Canvas**: Smooth cubic Hermite interpolation between layer representations as you drag the layer scrubber.

---

## File Structure

```
GradScope/
├── engine/
│   ├── __init__.py
│   ├── value.py             # Core Value class — scalar autograd engine
│   ├── nn.py                # Neuron, Layer, MLP built purely from Value
│   ├── graph_export.py      # Serializes computational DAG for D3/SVG
│   └── tests/
│       ├── test_value.py    # Finite-difference gradient checks
│       └── test_training.py # End-to-end XOR training verification
│
├── datasets/
│   ├── __init__.py
│   └── toy_datasets.py      # Spirals, moons, XOR generators (pure Python)
│
├── server/
│   ├── __init__.py
│   ├── main.py              # FastAPI app & WebSocket endpoint (/ws/train)
│   ├── training.py          # Full-batch training loop & snapshot builder
│   └── schemas.py           # Pydantic models for socket communication
│
├── frontend/
│   ├── src/
│   │   ├── api/socket.ts             # WebSocket client
│   │   ├── components/
│   │   │   ├── GraphCanvas.tsx       # DAG layout & traveling gradient pulses
│   │   │   ├── ManifoldCanvas.tsx    # 2D layer scatterplot & morph
│   │   │   ├── ControlDock.tsx       # Dataset picker, LR, playback controls
│   │   │   ├── LossSparkline.tsx     # Heartbeat-style loss sparkline
│   │   │   └── LayerScrubber.tsx     # Drag through layers
│   │   ├── hooks/useTrainingStream.ts
│   │   ├── styles/globals.css
│   │   ├── utils/interpolate.ts      # Smooth layer tweening
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   ├── vite.config.ts
│   └── tailwind.config.ts
│
├── notebooks/
│   └── experiments.ipynb    # Scratchpad for engine math & verification
├── requirements.txt
└── README.md
```

---

## Quickstart: Run It Yourself

### 1. Prerequisites
- Python 3.11+
- Node.js 18+ and npm

### 2. Start the Backend Server
```bash
# Install Python requirements
pip install -r requirements.txt

# Launch FastAPI on port 8000
uvicorn server.main:app --port 8000 --reload
```

The health check will be available at `http://localhost:8000/api/health` and the WebSocket at `ws://localhost:8000/ws/train`.

### 3. Start the Frontend Visualizer
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

### Keyboard Shortcuts
- `Space`: Toggle Train / Pause
- `Right Arrow`: Step one epoch
- `R`: Reset network weights and dataset

---

## Mathematical Appendix: The Chain Rule in Scalar Autograd

Consider a single multiplication node $z = x \cdot y$.

Suppose during the backward pass we have already computed $\frac{\partial L}{\partial z}$ (stored in `z.grad`). By the chain rule of multivariable calculus:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial x} = \frac{\partial L}{\partial z} \cdot y$$

$$\frac{\partial L}{\partial y} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial y} = \frac{\partial L}{\partial z} \cdot x$$

In `engine/value.py`, this is expressed as a local closure stored on $z$:

```python
def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad

    out._backward = _backward
    return out
```

When `.backward()` is called on the scalar loss node $L$:
1. It performs a **topological sort** of the graph (a directed acyclic graph) using depth-first search.
2. It sets the seed gradient $L.\text{grad} = 1.0$ because $\frac{\partial L}{\partial L} = 1$.
3. It iterates through the nodes in **reverse topological order**, invoking each node's `_backward()` function.
4. Because nodes can be reused in multiple operations (e.g. $y = x + x$), gradients **accumulate** (`+=`), accurately implementing:

$$\frac{\partial L}{\partial v_i} = \sum_{v_j \in \text{children}(v_i)} \frac{\partial L}{\partial v_j} \cdot \frac{\partial v_j}{\partial v_i}$$

---

## License
MIT License. Built for learning, exploration, and demystifying deep learning fundamentals.
