import {GraphError} from './errors.js';
import type {InternalTensor, Shape} from './graph.js';

export const DEFAULT_MAX_TOTAL_ELEMENTS = 10_000_000;

export function sizeOf(shape: readonly number[]): number {
  let size = 1;
  for (const dimension of shape) {
    size *= dimension;
  }
  return size;
}

export function isNormalizedScalar(shape: readonly number[]): boolean {
  return shape.length === 0;
}

export function normalizeInputShape(
  shape: readonly [] | readonly [number] | readonly [number, number] | undefined
): readonly number[] {
  if (shape === undefined || shape.length === 0) {
    return [];
  }
  return [...shape];
}

export function toCanonicalShape(shape: readonly number[]): readonly number[] {
  if (shape.length === 0) {
    return [];
  }
  let first = 0;
  while (first < shape.length - 1 && shape[first] === 1) {
    first += 1;
  }
  return shape.slice(first);
}

export function validateShape(shape: readonly number[], context: string): Shape | readonly [] {
  const rank = shape.length;
  if (rank > 2) {
    throw new GraphError('INVALID_SHAPE', `${context} has unsupported rank ${rank}`);
  }
  for (const dimension of shape) {
    if (!Number.isInteger(dimension) || dimension < 1 || dimension > 2 ** 31 - 1) {
      throw new GraphError('INVALID_SHAPE', `${context} has an invalid dimension ${dimension}`);
    }
  }
  if (rank === 0) return [];
  if (rank === 1) return [shape[0]!] as Shape;
  return [shape[0]!, shape[1]!] as Shape;
}

export function broadcastShapes(
  left: readonly number[],
  right: readonly number[]
): readonly number[] {
  const rank = Math.max(left.length, right.length);
  if (rank > 2) {
    throw new GraphError('INVALID_SHAPE', 'Broadcast result has unsupported rank');
  }
  const result: number[] = [];
  for (let axis = 0; axis < rank; axis += 1) {
    const outputAxis = rank - 1 - axis;
    const leftDimension = axis < left.length ? left[left.length - 1 - axis]! : 1;
    const rightDimension = axis < right.length ? right[right.length - 1 - axis]! : 1;
    if (leftDimension !== 1 && rightDimension !== 1 && leftDimension !== rightDimension) {
      throw new GraphError(
        'INVALID_SHAPE',
        `Cannot broadcast shapes [${left.join(', ')}] and [${right.join(', ')}]`
      );
    }
    result[outputAxis] = Math.max(leftDimension, rightDimension);
  }
  return result;
}

export function linearIndex(
  shape: readonly number[],
  outputShape: readonly number[],
  outputIndex: number
): number {
  if (shape.length === 0) {
    return 0;
  }
  const offset = outputShape.length - shape.length;
  let remaining = outputIndex;
  let sourceIndex = 0;
  for (let axis = outputShape.length - 1; axis >= 0; axis -= 1) {
    const outputDimension = outputShape[axis]!;
    const coordinate = remaining % outputDimension;
    remaining = Math.floor(remaining / outputDimension);
    if (axis >= offset) {
      const sourceDimension = shape[axis - offset]!;
      sourceIndex += (coordinate % sourceDimension) * sizeOf(shape.slice(axis - offset + 1));
    }
  }
  return sourceIndex;
}

export function zeroTensor(shape: readonly number[]): InternalTensor {
  return {shape, data: new Float64Array(sizeOf(shape))};
}

export function assertFiniteTensor(tensor: InternalTensor, context: string): void {
  for (let index = 0; index < tensor.data.length; index += 1) {
    if (!Number.isFinite(tensor.data[index]!)) {
      throw new GraphError('NON_FINITE_VALUE', `${context} contains NaN or an infinite value`);
    }
  }
}
