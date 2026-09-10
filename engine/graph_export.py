"""
GradScope Engine: Graph Export
Traverses the computational DAG of a Value and serializes nodes and edges to JSON format
for rendering in the D3 GraphCanvas visualizer.
"""

from __future__ import annotations
from typing import Dict, Any, Set, List
from engine.value import Value


def trace(root: Value) -> tuple[Set[Value], Set[tuple[Value, Value]]]:
    """
    Builds a set of all nodes and directed edges in the computational graph
    rooted at `root`. In the forward graph, parent -> child (inputs -> output).
    """
    nodes: Set[Value] = set()
    edges: Set[tuple[Value, Value]] = set()

    def build(v: Value):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                # In forward computation: child was an input to v
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def export_graph(root: Value, max_nodes: int = 300) -> Dict[str, Any]:
    """
    Serializes the computational graph of `root` into JSON structure:
    {
      "nodes": [
         {"id": "n0", "label": "x0", "value": 0.5, "grad": 0.02, "op": "", "depth": 0}
      ],
      "edges": [
         {"source": "n0", "target": "n1", "grad_flow": 0.02}
      ]
    }
    """
    raw_nodes, raw_edges = trace(root)

    # Compute topological levels (depths)
    # Forward depth: distance from input leaves
    in_degree: Dict[Value, int] = {n: 0 for n in raw_nodes}
    adj: Dict[Value, List[Value]] = {n: [] for n in raw_nodes}
    for p, c in raw_edges:
        adj[p].append(c)
        in_degree[c] += 1

    depths: Dict[Value, int] = {}
    queue = [n for n in raw_nodes if in_degree[n] == 0]
    for n in queue:
        depths[n] = 0

    curr_idx = 0
    while curr_idx < len(queue):
        u = queue[curr_idx]
        curr_idx += 1
        d = depths.get(u, 0)
        for v in adj[u]:
            depths[v] = max(depths.get(v, 0), d + 1)
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    # Align leaf parameters (weights and biases) closer to their consumers
    # so they do not all cram into depth 0
    for n in raw_nodes:
        if len(n._prev) == 0 and not (n.label and n.label.startswith('x')):
            if adj[n]:
                depths[n] = max(0, min(depths[c] for c in adj[n]) - 1)

    # Re-normalize depths to be contiguous 0, 1, 2, ...
    unique_depths = sorted(set(depths.values()))
    depth_remap = {old_d: new_d for new_d, old_d in enumerate(unique_depths)}
    for n in raw_nodes:
        depths[n] = depth_remap.get(depths[n], 0)

    # Sort nodes by depth for consistent ordering
    sorted_nodes = sorted(list(raw_nodes), key=lambda n: (depths.get(n, 0), n.label or ''))

    # If pruning is ever needed, prune leaf nodes that are furthest from the root,
    # NEVER prune the root (Loss node) or sever active paths!
    if len(sorted_nodes) > max_nodes:
        # Keep nodes sorted from root backwards (highest depth first)
        kept_nodes = sorted(list(raw_nodes), key=lambda n: depths.get(n, 0), reverse=True)[:max_nodes]
        allowed = set(kept_nodes)
        sorted_nodes = [n for n in sorted_nodes if n in allowed]
        raw_edges = {(p, c) for (p, c) in raw_edges if p in allowed and c in allowed}

    node_to_id: Dict[Value, str] = {}
    for idx, node in enumerate(sorted_nodes):
        node_to_id[node] = f"node_{idx}"

    nodes_json = []
    for node in sorted_nodes:
        nid = node_to_id[node]
        label = node.label
        if not label:
            label = node._op if node._op else f"v_{nid[-2:]}"

        nodes_json.append({
            "id": nid,
            "label": label,
            "value": round(node.data, 4),
            "grad": round(node.grad, 4),
            "op": node._op,
            "depth": depths.get(node, 0),
        })

    edges_json = []
    for p, c in raw_edges:
        if p in node_to_id and c in node_to_id:
            # Gradient flows from child back to parent
            grad_flow = round(abs(p.grad), 4)
            edges_json.append({
                "source": node_to_id[p],
                "target": node_to_id[c],
                "grad_flow": grad_flow,
            })

    return {
        "nodes": nodes_json,
        "edges": edges_json,
    }
