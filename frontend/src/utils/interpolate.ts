/**
 * Interpolation and tweening helpers for smooth manifold morphing.
 */

export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

export function smoothstep(t: number): number {
  // Cubic Hermite smoothstep: 3t^2 - 2t^3
  const clamped = Math.max(0, Math.min(1, t));
  return clamped * clamped * (3 - 2 * clamped);
}

export function interpolate2D(
  p1: [number, number],
  p2: [number, number],
  t: number
): [number, number] {
  const eased = smoothstep(t);
  return [
    lerp(p1[0], p2[0], eased),
    lerp(p1[1], p2[1], eased),
  ];
}

/**
 * Given an array of 2D coordinates across layers [[x0, y0], [x1, y1], ...]
 * and a continuous layer index s (e.g. 1.4 meaning 40% between layer 1 and 2),
 * calculates the interpolated position.
 */
export function getInterpolatedLayerCoords(
  layerCoords: [number, number][],
  scrubPosition: number
): [number, number] {
  if (!layerCoords || layerCoords.length === 0) return [0, 0];
  if (layerCoords.length === 1) return layerCoords[0];

  const clamped = Math.max(0, Math.min(layerCoords.length - 1, scrubPosition));
  const baseIdx = Math.floor(clamped);
  const nextIdx = Math.min(baseIdx + 1, layerCoords.length - 1);
  const frac = clamped - baseIdx;

  if (baseIdx === nextIdx) {
    return layerCoords[baseIdx];
  }

  return interpolate2D(layerCoords[baseIdx], layerCoords[nextIdx], frac);
}
