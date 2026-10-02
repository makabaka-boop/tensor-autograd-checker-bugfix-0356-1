import {GraphError} from './errors.js';
import type {GraphNode, TensorGraph} from './graph.js';
import {
  assertFiniteTensor,
  broadcastShapes,
  DEFAULT_MAX_TOTAL_ELEMENTS,
  normalizeInputShape,
  sizeOf,
  validateShape,
} from './tensor.js';

const BINARY_OPS = new Set(['add', 'mul']);
const UNARY_OPS = new Set(['sum', 'relu']);
export const MAX_NODE_COUNT = 100;

function hasInputs(node: GraphNode): boolean {
  return Array.isArray(node.inputs);
}

function expectInputCount(node: GraphNode, opName: string, count: number): void {
  if (!Array.isArray(node.inputs) || node.inputs.length !== count) {
    throw new GraphError('INVALID_GRAPH', `${opName} node ${node.id} requires exactly ${count} input${count === 1 ? '' : 's'}`);
  }
  for (const inputId of node.inputs) {
    if (typeof inputId !== 'string' || inputId.length === 0) {
      throw new GraphError('INVALID_GRAPH', `${opName} node ${node.id} contains an invalid input id`);
    }
  }
}

export interface PreparedGraph {
  readonly nodes: ReadonlyMap<string, GraphNode>;
  readonly order: readonly GraphNode[];
  readonly shapes: ReadonlyMap<string, readonly number[]>;
  readonly totalElements: number;
}

export function prepareGraph(
  graph: TensorGraph,
  maxTotalElements: number | undefined = DEFAULT_MAX_TOTAL_ELEMENTS
): PreparedGraph {
  if (!graph || typeof graph !== 'object' || !Array.isArray(graph.nodes)) {
    throw new GraphError('INVALID_GRAPH', 'graph.nodes must be an array');
  }
  if (graph.nodes.length === 0) {
    throw new GraphError('INVALID_GRAPH', 'graph must contain at least one node');
  }
  if (graph.nodes.length > MAX_NODE_COUNT) {
    throw new GraphError('GRAPH_TOO_LARGE', `graph may contain at most ${MAX_NODE_COUNT} nodes`);
  }
  if (
    maxTotalElements !== DEFAULT_MAX_TOTAL_ELEMENTS &&
    (typeof maxTotalElements !== 'number' ||
      !Number.isSafeInteger(maxTotalElements) ||
      maxTotalElements < 1)
  ) {
    throw new GraphError('INVALID_REQUEST', 'maxTotalElements must be a positive safe integer');
  }

  const nodes = new Map<string, GraphNode>();
  for (const node of graph.nodes) {
    if (!node || typeof node !== 'object') {
      throw new GraphError('INVALID_GRAPH', 'each graph node must be an object');
    }
    if (typeof node.id !== 'string' || node.id.length === 0) {
      throw new GraphError('INVALID_GRAPH', 'each graph node requires a non-empty string id');
    }
    if (nodes.has(node.id)) {
      throw new GraphError('INVALID_GRAPH', `duplicate node id ${node.id}`);
    }
    if (typeof node.op !== 'string') {
      throw new GraphError('INVALID_GRAPH', `node ${node.id} requires a string op`);
    }
    if (node.op === 'const') {
      if (hasInputs(node)) {
        throw new GraphError('INVALID_GRAPH', `const node ${node.id} must not have inputs`);
      }
      if (!Array.isArray(node.value)) {
        throw new GraphError('INVALID_TENSOR', `const node ${node.id} requires a numeric value array`);
      }
    } else if (node.op === 'input') {
      if (hasInputs(node)) {
        throw new GraphError('INVALID_GRAPH', `input node ${node.id} must not have inputs`);
      }
      if (node.value !== undefined) {
        throw new GraphError('INVALID_GRAPH', `input node ${node.id} must not declare value`);
      }
    } else if (BINARY_OPS.has(node.op)) {
      expectInputCount(node, node.op, 2);
      if (node.value !== undefined) {
        throw new GraphError('INVALID_GRAPH', `op node ${node.id} must not declare value`);
      }
    } else if (UNARY_OPS.has(node.op)) {
      expectInputCount(node, node.op, 1);
      if (node.value !== undefined) {
        throw new GraphError('INVALID_GRAPH', `op node ${node.id} must not declare value`);
      }
    } else if (node.op === 'matmul') {
      expectInputCount(node, node.op, 2);
      if (node.value !== undefined) {
        throw new GraphError('INVALID_GRAPH', `op node ${node.id} must not declare value`);
      }
    } else {
      throw new GraphError('INVALID_GRAPH', `node ${node.id} has unsupported op ${String(node.op)}`);
    }
    if (Object.prototype.hasOwnProperty.call(node, 'shape')) {
      if (!Array.isArray(node.shape)) {
        throw new GraphError('INVALID_SHAPE', `node ${node.id} shape must be an array`);
      }
      validateShape(normalizeInputShape(node.shape), `node ${node.id}`);
    }
    nodes.set(node.id, node);
  }

  // Validate all input references before traversal.
  for (const node of graph.nodes) {
    for (const inputId of node.inputs ?? []) {
      if (typeof inputId !== 'string' || !nodes.has(inputId)) {
        throw new GraphError('INVALID_GRAPH', `node ${node.id} references unknown node ${String(inputId)}`);
      }
    }
  }

  const order = topologicalOrder(nodes);
  const shapes = new Map<string, readonly number[]>();
  let totalElements = 0;
  for (const node of order) {
    const shape = inferShape(node, nodes, shapes);
    if (node.shape !== undefined) {
      const declared = normalizeInputShape(node.shape);
      if (
        declared.length !== shape.length ||
        declared.some((dimension, index) => dimension !== shape[index])
      ) {
        throw new GraphError(
          'INVALID_SHAPE',
          `declared shape [${declared.join(',')}] on ${node.id} does not match inferred shape [${shape.join(',')}]`
        );
      }
    }
    const nodeElements = sizeOf(shape);
    if (totalElements > maxTotalElements - nodeElements) {
      throw new GraphError(
        'ELEMENT_LIMIT_EXCEEDED',
        `graph requires more than ${maxTotalElements} forward tensor elements`
      );
    }
    totalElements += nodeElements;
    shapes.set(node.id, shape);

    if (node.op === 'const') {
      const values = node.value!;
      if (values.length !== nodeElements) {
        throw new GraphError('INVALID_TENSOR', `const node ${node.id} value length does not match shape`);
      }
      for (const value of values) {
        if (typeof value !== 'number' || !Number.isFinite(value)) {
          throw new GraphError('NON_FINITE_VALUE', `const node ${node.id} must contain only finite numbers`);
        }
      }
    }
  }

  // Also check raw constant values when shape is supplied, because inferShape above
  // validates the canonical value length too.
  return {nodes, order, shapes, totalElements};
}

function topologicalOrder(nodes: ReadonlyMap<string, GraphNode>): readonly GraphNode[] {
  const indegrees = new Map<string, number>();
  const dependents = new Map<string, string[]>();
  for (const id of nodes.keys()) {
    indegrees.set(id, 0);
    dependents.set(id, []);
  }
  for (const node of nodes.values()) {
    const inputs = [...new Set(node.inputs ?? [])];
    indegrees.set(node.id, inputs.length);
    for (const inputId of inputs) {
      dependents.get(inputId)!.push(node.id);
    }
  }

  const ready = [...indegrees.entries()].filter(([, degree]) => degree === 0).map(([id]) => id);
  const order: GraphNode[] = [];
  while (ready.length > 0) {
    const id = ready.shift()!;
    const node = nodes.get(id)!;
    order.push(node);
    for (const dependentId of dependents.get(id)!) {
      indegrees.set(dependentId, indegrees.get(dependentId)! - 1);
      if (indegrees.get(dependentId) === 0) {
        ready.push(dependentId);
      }
    }
  }
  if (order.length !== nodes.size) {
    throw new GraphError('CYCLIC_GRAPH', 'graph contains a cycle');
  }
  return order;
}

function inferShape(
  node: GraphNode,
  nodes: ReadonlyMap<string, GraphNode>,
  shapes: ReadonlyMap<string, readonly number[]>
): readonly number[] {
  if (node.op === 'const' || node.op === 'input') {
    if (!Array.isArray(node.shape)) {
      throw new GraphError('INVALID_SHAPE', `node ${node.id} must declare its shape`);
    }
    const declared = validateShape(normalizeInputShape(node.shape), `node ${node.id}`);
    if (node.op === 'const') {
      if (node.value!.length !== sizeOf(declared)) {
        throw new GraphError('INVALID_TENSOR', `const node ${node.id} value length does not match shape`);
      }
    }
    return [...declared];
  }

  const inputIds = node.inputs!;
  const inputs = inputIds.map((id) => {
    const shape = shapes.get(id);
    if (!shape) {
      throw new GraphError('CYCLIC_GRAPH', `shape for ${id} is not available`);
    }
    return shape;
  });

  if (node.op === 'add' || node.op === 'mul') {
    return broadcastShapes(inputs[0]!, inputs[1]!);
  }
  if (node.op === 'relu') {
    return inputs[0]!;
  }
  if (node.op === 'sum') {
    return [];
  }
  if (node.op === 'matmul') {
    const left = inputs[0]!;
    const right = inputs[1]!;
    if (left.length !== 2 || right.length !== 2) {
      throw new GraphError('INVALID_SHAPE', `matmul node ${node.id} requires two 2D inputs`);
    }
    if (left[1] !== right[0]) {
      throw new GraphError(
        'INVALID_SHAPE',
        `matmul node ${node.id} dimensions ${left[0]}x${left[1]} and ${right[0]}x${right[1]} are incompatible`
      );
    }
    return [left[0]!, right[1]!];
  }
  throw new GraphError('INVALID_GRAPH', `unsupported op ${node.op}`);
}

export function collectReachable(
  outputIds: readonly string[],
  nodes: ReadonlyMap<string, GraphNode>
): Set<string> {
  const reachable = new Set<string>();
  const stack = [...outputIds];
  while (stack.length > 0) {
    const id = stack.pop()!;
    if (reachable.has(id)) continue;
    const node = nodes.get(id);
    if (!node) {
      throw new GraphError('INVALID_REQUEST', `unknown output node ${id}`);
    }
    reachable.add(id);
    for (const inputId of node.inputs ?? []) {
      if (!reachable.has(inputId)) stack.push(inputId);
    }
  }
  return reachable;
}

export function makeConstantTensor(node: GraphNode, shape: readonly number[]): {
  shape: readonly number[];
  data: Float64Array;
} {
  const data = Float64Array.from(node.value!);
  const tensor = {shape, data};
  assertFiniteTensor(tensor, `const node ${node.id}`);
  return tensor;
}
