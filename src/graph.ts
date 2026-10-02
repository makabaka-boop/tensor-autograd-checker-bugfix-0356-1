export type Shape = readonly [number] | readonly [number, number];

export interface TensorSpec {
  shape?: readonly [] | readonly [number] | readonly [number, number];
  data: readonly number[];
}

export type NodeOp =
  | 'const'
  | 'input'
  | 'add'
  | 'mul'
  | 'matmul'
  | 'sum'
  | 'relu';

export interface GraphNode {
  id: string;
  op: NodeOp;
  shape?: readonly [] | readonly [number] | readonly [number, number];
  value?: readonly number[];
  inputs?: readonly string[];
}

export interface TensorGraph {
  nodes: readonly GraphNode[];
}

export interface InternalTensor {
  readonly shape: readonly number[];
  readonly data: Float64Array;
}

export interface ComputeRequest {
  graph: TensorGraph;
  inputs: Record<string, TensorSpec>;
  outputs: readonly string[];
  gradInputs?: readonly string[];
  gradOutput?: string;
  gradSeeds?: Record<string, TensorSpec>;
  maxTotalElements?: number;
}

export interface OutputTensor {
  shape: readonly number[];
  data: readonly number[];
}

export interface ComputeResponse {
  outputs: Record<string, OutputTensor>;
  gradients: Record<string, OutputTensor>;
}
