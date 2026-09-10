"""
GradScope Server: Pydantic Schemas
Defines structured schemas for client commands and WebSocket training snapshots.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    dataset: str = Field(default="spirals", description="spirals, moons, or xor")
    noise: float = Field(default=0.08, ge=0.0, le=0.5)
    n_points: int = Field(default=100, ge=20, le=300)
    hidden_dims: List[int] = Field(default_factory=lambda: [2, 2])
    activation: str = Field(default="tanh", description="tanh, relu, sigmoid")
    learning_rate: float = Field(default=0.1, ge=0.001, le=2.0)
    step_delay_ms: int = Field(default=50, ge=0, le=1000)


class ClientMessage(BaseModel):
    action: str = Field(description="'start', 'pause', 'step', 'reset', 'config'")
    config: Optional[TrainingConfig] = None


class GraphNode(BaseModel):
    id: str
    label: str
    value: float
    grad: float
    op: str
    depth: int


class GraphEdge(BaseModel):
    source: str
    target: str
    grad_flow: float


class GraphData(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]


class ManifoldPoint(BaseModel):
    id: int
    label: int  # 0 or 1
    layers: List[List[float]]  # List of [x, y] coordinates across all layers


class TrainingSnapshot(BaseModel):
    type: str = "snapshot"
    step: int
    epoch: int
    loss: float
    accuracy: float
    graph: GraphData
    manifold: List[ManifoldPoint]
    num_layers: int
    layer_labels: List[str]
    is_running: bool
