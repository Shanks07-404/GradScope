import React, { useMemo, useState, useRef, useEffect } from 'react';
import { GraphData, GraphNode } from '../api/socket';

interface GraphCanvasProps {
  graph: GraphData | null;
  step: number;
}

interface PositionedNode extends GraphNode {
  x: number;
  y: number;
  radius: number;
}

export const GraphCanvas: React.FC<GraphCanvasProps> = ({ graph, step }) => {
  const [hoveredNode, setHoveredNode] = useState<PositionedNode | null>(null);
  const [tooltipPos, setTooltipPos] = useState<{
    left: number;
    top: number;
    transform: string;
  }>({ left: 0, top: 0, transform: 'translate(-50%, -100%)' });

  const containerRef = useRef<HTMLDivElement | null>(null);
  const scrollAreaRef = useRef<HTMLDivElement | null>(null);
  const [containerDim, setContainerDim] = useState({ width: 600, height: 450 });

  // Update container dimensions on window resize
  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setContainerDim({
          width: Math.max(entry.contentRect.width, 300),
          height: Math.max(entry.contentRect.height, 300),
        });
      }
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  // Compute layered layout adaptively based on depth and node count
  const { positionedNodes, nodeMap, effectiveWidth, effectiveHeight } = useMemo(() => {
    if (!graph || !graph.nodes || graph.nodes.length === 0) {
      return {
        positionedNodes: [],
        nodeMap: new Map<string, PositionedNode>(),
        effectiveWidth: containerDim.width,
        effectiveHeight: containerDim.height,
      };
    }

    // Group nodes by depth
    const depthGroups = new Map<number, GraphNode[]>();
    let maxDepth = 0;
    for (const node of graph.nodes) {
      const d = node.depth || 0;
      if (d > maxDepth) maxDepth = d;
      if (!depthGroups.has(d)) depthGroups.set(d, []);
      depthGroups.get(d)!.push(node);
    }

    let maxCountInCol = 1;
    depthGroups.forEach((group) => {
      if (group.length > maxCountInCol) maxCountInCol = group.length;
    });

    // Adaptive spacing so [4,4] or deeper architectures never cram or overlap
    const minColWidth = 72;
    const minRowHeight = 36;
    const paddingX = 50;
    const paddingY = 45;

    const neededWidth = paddingX * 2 + (maxDepth + 1) * minColWidth;
    const neededHeight = paddingY * 2 + maxCountInCol * minRowHeight;

    const calcWidth = Math.max(containerDim.width, neededWidth);
    const calcHeight = Math.max(containerDim.height, neededHeight);

    const colWidth = (calcWidth - 2 * paddingX) / Math.max(maxDepth, 1);

    const positioned: PositionedNode[] = [];
    const map = new Map<string, PositionedNode>();

    depthGroups.forEach((group, depth) => {
      const colX = paddingX + depth * colWidth;
      const count = group.length;
      const rowHeight = (calcHeight - 2 * paddingY) / Math.max(count, 1);
      const radius = Math.min(13, Math.max(8, rowHeight * 0.36));

      group.forEach((node, idx) => {
        const posY = paddingY + (idx + 0.5) * rowHeight;
        const pNode: PositionedNode = {
          ...node,
          x: colX,
          y: posY,
          radius,
        };
        positioned.push(pNode);
        map.set(node.id, pNode);
      });
    });

    return {
      positionedNodes: positioned,
      nodeMap: map,
      effectiveWidth: calcWidth,
      effectiveHeight: calcHeight,
    };
  }, [graph, containerDim]);

  // Boundary-aware tooltip positioning
  const handleNodeHover = (node: PositionedNode, event: React.MouseEvent) => {
    setHoveredNode(node);
    const container = containerRef.current;
    if (!container) return;

    const containerRect = container.getBoundingClientRect();
    const cursorX = event.clientX - containerRect.left;
    const cursorY = event.clientY - containerRect.top;

    const tooltipWidth = 180;
    const tooltipHeight = 120;

    let posX = cursorX;
    let posY = cursorY;
    let transX = '-50%';
    let transY = '-100%';

    // Horizontal boundary clamping
    if (cursorX < tooltipWidth * 0.6) {
      // Near left edge (e.g. leaf input nodes): anchor to right of cursor
      posX = cursorX + 16;
      transX = '0%';
    } else if (cursorX > containerRect.width - tooltipWidth * 0.6) {
      // Near right edge (e.g. Loss node): anchor to left of cursor
      posX = cursorX - 16;
      transX = '-100%';
    }

    // Vertical boundary clamping
    if (cursorY < tooltipHeight + 20) {
      // Near top: flip below cursor
      posY = cursorY + 18;
      transY = '0%';
    } else {
      posY = cursorY - 14;
      transY = '-100%';
    }

    setTooltipPos({
      left: posX,
      top: posY,
      transform: `translate(${transX}, ${transY})`,
    });
  };

  return (
    <div
      ref={containerRef}
      className="flex flex-col h-full bg-chalkboard-bg select-none relative overflow-hidden"
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-chalkboard-grid bg-chalkboard-card/30 z-10">
        <div className="flex items-center gap-2">
          <span className="font-serif font-semibold text-chalkboard-chalk">
            Computational DAG
          </span>
          <span className="text-[11px] font-mono text-chalkboard-sage">
            (Reverse-Mode Gradient Flow)
          </span>
        </div>
        <div className="flex items-center gap-2 text-xs font-mono">
          <span className="w-2 h-2 rounded-full bg-chalkboard-amber animate-pulse shadow-[0_0_6px_#F2B84B]" />
          <span className="text-chalkboard-amber">Amber Pulse = ∂L/∂w Flow</span>
        </div>
      </div>

      {/* Scrollable Canvas Container */}
      <div
        ref={scrollAreaRef}
        className="flex-1 relative w-full h-full overflow-auto chalk-grid"
      >
        <svg
          width={effectiveWidth}
          height={effectiveHeight}
          className="block"
        >
          <defs>
            <marker
              id="edge-arrow"
              viewBox="0 0 10 10"
              refX="18"
              refY="5"
              markerWidth="5"
              markerHeight="5"
              orient="auto-start-reverse"
            >
              <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#3F4F48" />
            </marker>
          </defs>

          {/* Edges */}
          {graph?.edges.map((edge, idx) => {
            const src = nodeMap.get(edge.source);
            const tgt = nodeMap.get(edge.target);
            if (!src || !tgt) return null;

            const hasGradFlow = Math.abs(edge.grad_flow) > 0.001;

            return (
              <g key={`edge-${edge.source}-${edge.target}-${idx}`}>
                {/* Connection line */}
                <line
                  x1={src.x}
                  y1={src.y}
                  x2={tgt.x}
                  y2={tgt.y}
                  stroke="#232D28"
                  strokeWidth="1.25"
                  markerEnd="url(#edge-arrow)"
                />

                {/* Animated gradient pulse flowing in reverse (target -> source) */}
                {hasGradFlow && (
                  <circle r="3" fill="#F2B84B" className="filter drop-shadow-[0_0_4px_#F2B84B]">
                    <animateMotion
                      key={`pulse-${step}-${idx}`}
                      path={`M ${tgt.x} ${tgt.y} L ${src.x} ${src.y}`}
                      dur="1.2s"
                      repeatCount="indefinite"
                    />
                  </circle>
                )}
              </g>
            );
          })}

          {/* Nodes */}
          {positionedNodes.map((node) => {
            const isHovered = hoveredNode?.id === node.id;
            const hasGrad = Math.abs(node.grad) > 0.0001;
            const isLoss = node.label === 'Loss';
            const r = node.radius;

            return (
              <g
                key={node.id}
                transform={`translate(${node.x}, ${node.y})`}
                className="cursor-pointer transition-transform duration-100"
                onMouseEnter={(e) => handleNodeHover(node, e)}
                onMouseMove={(e) => handleNodeHover(node, e)}
                onMouseLeave={() => setHoveredNode(null)}
              >
                {/* Outer halo when active or Loss */}
                {(hasGrad || isLoss) && (
                  <circle
                    r={isHovered ? r + 5 : r + 3}
                    fill="none"
                    stroke="#F2B84B"
                    strokeWidth={isLoss ? '1.75' : '1'}
                    strokeDasharray={isLoss ? 'none' : '2 2'}
                    opacity={isLoss ? 0.85 : 0.4}
                  />
                )}

                {/* Node circle */}
                <circle
                  r={isHovered ? r + 2 : r}
                  fill={isLoss ? '#2B2313' : '#131D19'}
                  stroke={
                    isLoss
                      ? '#F2B84B'
                      : isHovered
                      ? '#F2B84B'
                      : hasGrad
                      ? '#EDEAE0'
                      : '#3F4F48'
                  }
                  strokeWidth={isHovered || isLoss ? 2 : 1.25}
                />

                {/* Operation / label inside node */}
                <text
                  textAnchor="middle"
                  dy="3.5"
                  fill={isLoss || isHovered ? '#F2B84B' : '#EDEAE0'}
                  fontSize={node.op.length > 2 || r < 10 ? '7px' : '9px'}
                  fontFamily='"JetBrains Mono", monospace'
                  fontWeight="600"
                >
                  {isLoss ? 'L' : (node.op || 'v')}
                </text>

                {/* Subtle label underneath */}
                <text
                  textAnchor="middle"
                  dy={r + 11}
                  fill={isLoss ? '#F2B84B' : '#8FA69D'}
                  fontSize="8px"
                  fontFamily='"JetBrains Mono", monospace'
                  fontWeight={isLoss ? '600' : 'normal'}
                >
                  {node.label.length > 9 ? node.label.slice(0, 8) + '…' : node.label}
                </text>
              </g>
            );
          })}
        </svg>

        {/* Boundary-clamped Hover Tooltip */}
        {hoveredNode && (
          <div
            className="absolute z-30 pointer-events-none bg-chalkboard-card/95 border border-chalkboard-amber/80 rounded px-3 py-2 text-xs font-mono shadow-2xl backdrop-blur-md transition-all duration-75"
            style={{
              left: tooltipPos.left,
              top: tooltipPos.top,
              transform: tooltipPos.transform,
            }}
          >
            <div className="flex items-center justify-between gap-4 border-b border-chalkboard-grid pb-1 mb-1.5">
              <span className="text-chalkboard-amber font-semibold">
                {hoveredNode.label || hoveredNode.id}
              </span>
              <span className="text-[10px] bg-chalkboard-grid px-1.5 py-0.2 rounded text-chalkboard-sage">
                {hoveredNode.op || 'leaf'}
              </span>
            </div>
            <div className="flex flex-col gap-0.5 text-[11px]">
              <div className="flex justify-between gap-3">
                <span className="text-chalkboard-sage">Value:</span>
                <span className="text-chalkboard-chalk font-medium">
                  {hoveredNode.value.toFixed(4)}
                </span>
              </div>
              <div className="flex justify-between gap-3">
                <span className="text-chalkboard-sage">∂L/∂v (grad):</span>
                <span className="text-chalkboard-amber font-semibold">
                  {hoveredNode.grad.toFixed(4)}
                </span>
              </div>
              <div className="flex justify-between gap-3">
                <span className="text-chalkboard-sage">Depth:</span>
                <span className="text-chalkboard-sage font-medium">
                  {hoveredNode.depth}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
