import {GraphError} from './errors.js';
import type {
  ComputeRequest,
  ComputeResponse,
  GraphNode,
  InternalTensor,
  OutputTensor,
  TensorSpec,
} from './graph.js';
import {collectReachable, makeConstantTensor, prepareGraph} from './prepare.js';
import {
  assertFiniteTensor,
  broadcastShapes,
  linearIndex,
  normalizeInputShape,
  sizeOf,
  validateShape,
  zeroTensor,
} from './tensor.js';

export function compute(request: ComputeRequest): ComputeResponse {
  const prepared = prepareGraph(request.graph, request.maxTotalElements ?? undefined);
  const outputIds = validateOutputs(request.outputs, prepared.nodes);
  const reachable = collectReachable(outputIds, prepared.nodes);
  const inputTensors = validateInputs(request.inputs, prepared, reachable);

  const gradInputIds = validateGradInputs(request.gradInputs, prepared.nodes, reachable);
  const seedOutputIds = Object.keys(request.gradSeeds ?? {});
  const gradOutputId = resolveGradOutput(
    request.gradOutput ?? seedOutputIds[0], outputIds, prepared, gradInputIds
  );

  const values = forward(prepared, reachable, inputTensors);

  const outputs = Object.create(null) as Record<string, OutputTensor>;
  for (const id of outputIds) {
    outputs[id] = toOutputTensor(values.get(id)!);
  }

  let gradients = Object.create(null) as Record<string, OutputTensor>;
  if (gradInputIds.length > 0) {
    const targetAncestors = collectReachable([gradOutputId], prepared.nodes);
    const grads = backward(prepared, values, gradOutputId, targetAncestors);
    gradients = Object.create(null) as Record<string, OutputTensor>;
    for (const id of gradInputIds) {
      const gradient = grads.get(id);
      if (!gradient) {
        // This is defensive: validateGradInputs requires target reachability.
        throw new GraphError('UNSUPPORTED_GRADIENT', `could not derive gradient for input ${id}`);
      }
      gradients[id] = toOutputTensor(gradient);
    }
  }

  return {outputs, gradients};
}

function validateOutputs(outputs: readonly string[] | undefined, nodes: ReadonlyMap<string, GraphNode>): readonly string[] {
  if (!Array.isArray(outputs) || outputs.length === 0) {
    throw new GraphError('INVALID_REQUEST', 'outputs must be a non-empty array of node ids');
  }
  const seen = new Set<string>();
  for (const id of outputs) {
    if (typeof id !== 'string' || id.length === 0) {
      throw new GraphError('INVALID_REQUEST', 'output ids must be non-empty strings');
    }
    if (seen.has(id)) {
      throw new GraphError('INVALID_REQUEST', `duplicate output ${id}`);
    }
    seen.add(id);
    if (!nodes.has(id)) {
      throw new GraphError('INVALID_REQUEST', `unknown output node ${id}`);
    }
  }
  return outputs;
}

function validateInputs(
  inputs: Record<string, TensorSpec> | undefined,
  prepared: ReturnType<typeof prepareGraph>,
  reachable: Set<string>
): Map<string, InternalTensor> {
  if (!inputs || typeof inputs !== 'object' || Array.isArray(inputs)) {
    throw new GraphError('INVALID_REQUEST', 'inputs must be an object');
  }
  const result = new Map<string, InternalTensor>();
  for (const [id, spec] of Object.entries(inputs)) {
    const node = prepared.nodes.get(id);
    if (!node) {
      throw new GraphError('INVALID_REQUEST', `input ${id} is not a graph node`);
    }
    if (node.op !== 'input') {
      throw new GraphError('INVALID_REQUEST', `value for ${id} can only be supplied to an input node`);
    }
    if (!spec || typeof spec !== 'object' || Array.isArray(spec) || !Array.isArray(spec.data)) {
      throw new GraphError('INVALID_TENSOR', `input ${id} requires a numeric data array`);
    }
    const inferred = prepared.shapes.get(id)!;
    if (spec.shape !== undefined) {
      const supplied = normalizeInputShape(spec.shape);
      validateShape(supplied, `input ${id}`);
      if (
        supplied.length !== inferred.length ||
        supplied.some((dimension, axis) => dimension !== inferred[axis])
      ) {
        throw new GraphError('INVALID_SHAPE', `input ${id} shape does not match graph shape [${inferred.join(',')}]`);
      }
    }
    if (spec.data.length !== sizeOf(inferred)) {
      throw new GraphError('INVALID_TENSOR', `input ${id} data length does not match shape`);
    }
    const data = new Float64Array(spec.data.length);
    for (let index = 0; index < spec.data.length; index += 1) {
      const value = spec.data[index]!;
      if (typeof value !== 'number' || !Number.isFinite(value)) {
        throw new GraphError('NON_FINITE_VALUE', `input ${id} must contain only finite numbers`);
      }
      data[index] = value;
    }
    // Unreachable inputs are accepted but not retained for allocation/execution.
    if (reachable.has(id)) {
      result.set(id, {shape: inferred, data});
    }
  }

  for (const node of prepared.order) {
    if (node.op === 'input' && reachable.has(node.id) && !result.has(node.id)) {
      throw new GraphError('INVALID_REQUEST', `missing value for input node ${node.id}`);
    }
  }
  return result;
}

function validateGradInputs(
  gradInputs: readonly string[] | undefined,
  nodes: ReadonlyMap<string, GraphNode>,
  reachable: Set<string>
): readonly string[] {
  if (gradInputs === undefined) return [];
  if (!Array.isArray(gradInputs)) {
    throw new GraphError('INVALID_REQUEST', 'gradInputs must be an array of input node ids');
  }
  const seen = new Set<string>();
  for (const id of gradInputs) {
    if (typeof id !== 'string' || id.length === 0) {
      throw new GraphError('INVALID_REQUEST', 'gradient input ids must be non-empty strings');
    }
    if (seen.has(id)) {
      throw new GraphError('INVALID_REQUEST', `duplicate gradient input ${id}`);
    }
    seen.add(id);
    const node = nodes.get(id);
    if (!node) {
      throw new GraphError('INVALID_REQUEST', `unknown gradient input ${id}`);
    }
    if (node.op !== 'input') {
      throw new GraphError('UNSUPPORTED_GRADIENT', `gradient target ${id} must be an input node`);
    }
    if (!reachable.has(id)) {
      throw new GraphError('UNSUPPORTED_GRADIENT', `gradient input ${id} is not reachable from outputs`);
    }
  }
  return gradInputs;
}

function resolveGradOutput(
  gradOutput: string | undefined,
  outputIds: readonly string[],
  prepared: ReturnType<typeof prepareGraph>,
  gradInputIds: readonly string[]
): string {
  if (gradInputIds.length === 0) {
    if (gradOutput !== undefined) {
      throw new GraphError('INVALID_REQUEST', 'gradOutput requires at least one gradInput');
    }
    return '';
  }
  if (gradOutput !== undefined) {
    if (typeof gradOutput !== 'string' || !prepared.nodes.has(gradOutput)) {
      throw new GraphError('INVALID_REQUEST', 'gradOutput must reference a graph node');
    }
    if (!outputIds.includes(gradOutput)) {
      throw new GraphError('INVALID_REQUEST', 'gradOutput must be included in outputs');
    }
    const shape = prepared.shapes.get(gradOutput)!;
    if (shape.length !== 0) {
      throw new GraphError('UNSUPPORTED_GRADIENT', 'gradOutput must be a scalar');
    }
    const targetReachable = collectReachable([gradOutput], prepared.nodes);
    for (const id of gradInputIds) {
      if (!targetReachable.has(id)) {
        throw new GraphError('UNSUPPORTED_GRADIENT', `gradient input ${id} is not reachable from ${gradOutput}`);
      }
    }
    return gradOutput;
  }

  const scalarOutputs = outputIds.filter((id) => prepared.shapes.get(id)!.length === 0);
  if (scalarOutputs.length !== 1) {
    throw new GraphError(
      'INVALID_REQUEST',
      'gradOutput is required when outputs do not contain exactly one scalar tensor'
    );
  }
  const selected = scalarOutputs[0]!;
  const targetReachable = collectReachable([selected], prepared.nodes);
  for (const id of gradInputIds) {
    if (!targetReachable.has(id)) {
      throw new GraphError('UNSUPPORTED_GRADIENT', `gradient input ${id} is not reachable from ${selected}`);
    }
  }
  return selected;
}

function forward(
  prepared: ReturnType<typeof prepareGraph>,
  reachable: Set<string>,
  inputTensors: Map<string, InternalTensor>
): Map<string, InternalTensor> {
  const values = new Map<string, InternalTensor>();
  for (const node of prepared.order) {
    if (!reachable.has(node.id)) continue;

    if (node.op === 'const') {
      values.set(node.id, makeConstantTensor(node, prepared.shapes.get(node.id)!));
      continue;
    }
    if (node.op === 'input') {
      values.set(node.id, inputTensors.get(node.id)!);
      continue;
    }

    const inputs = node.inputs!.map((id) => values.get(id)!);
    let output: InternalTensor;
    if (node.op === 'add') {
      output = elementwise(inputs[0]!, inputs[1]!, (a, b) => a + b);
    } else if (node.op === 'mul') {
      output = elementwise(inputs[0]!, inputs[1]!, (a, b) => a * b);
    } else if (node.op === 'relu') {
      output = reluForward(inputs[0]!);
    } else if (node.op === 'sum') {
      output = sumForward(inputs[0]!);
    } else if (node.op === 'matmul') {
      output = matmulForward(inputs[0]!, inputs[1]!);
    } else {
      throw new GraphError('INVALID_GRAPH', `unsupported op ${node.op}`);
    }
    assertFiniteTensor(output, `node ${node.id}`);
    values.set(node.id, output);
  }
  return values;
}

function elementwise(
  left: InternalTensor,
  right: InternalTensor,
  operation: (left: number, right: number) => number
): InternalTensor {
  const shape = broadcastShapes(left.shape, right.shape);
  const output = zeroTensor(shape);
  for (let index = 0; index < output.data.length; index += 1) {
    const leftValue = left.data[linearIndex(left.shape, shape, index)]!;
    const rightValue = right.data[linearIndex(right.shape, shape, index)]!;
    output.data[index] = operation(leftValue, rightValue);
  }
  return output;
}

function multiplyBroadcast(gradient: InternalTensor, factor: InternalTensor): InternalTensor {
  const shape = broadcastShapes(gradient.shape, factor.shape);
  const output = zeroTensor(shape);
  for (let index = 0; index < output.data.length; index += 1) {
    const gradientValue = gradient.data[linearIndex(gradient.shape, shape, index)]!;
    const factorValue = factor.data[linearIndex(factor.shape, shape, index)]!;
    output.data[index] = gradientValue * factorValue;
  }
  return output;
}

function reluForward(input: InternalTensor): InternalTensor {
  const output = zeroTensor(input.shape);
  for (let index = 0; index < input.data.length; index += 1) {
    const value = input.data[index]!;
    output.data[index] = value > 0 ? value : 0;
  }
  return output;
}

function sumForward(input: InternalTensor): InternalTensor {
  let total = 0;
  for (const value of input.data) {
    total += value;
  }
  const output = zeroTensor([]);
  output.data[0] = total;
  return output;
}

function matmulForward(left: InternalTensor, right: InternalTensor): InternalTensor {
  const m = left.shape[0]!;
  const k = left.shape[1]!;
  const n = right.shape[1]!;
  const output = zeroTensor([m, n]);
  for (let row = 0; row < m; row += 1) {
    for (let common = 0; common < k; common += 1) {
      const leftValue = left.data[row * k + common]!;
      for (let column = 0; column < n; column += 1) {
        output.data[row * n + column]! += leftValue * right.data[common * n + column]!;
      }
    }
  }
  return output;
}

function backward(
  prepared: ReturnType<typeof prepareGraph>,
  values: Map<string, InternalTensor>,
  outputId: string,
  targetAncestors: Set<string>
): Map<string, InternalTensor> {
  const grads = new Map<string, InternalTensor>();
  for (const node of prepared.order) {
    if (node.id !== outputId && targetAncestors.has(node.id) && node.op !== 'const') {
      grads.set(node.id, zeroTensor(prepared.shapes.get(node.id)!));
    }
  }
  const seed = zeroTensor([]);
  seed.data[0] = 1;
  grads.set(outputId, seed);

  for (let orderIndex = prepared.order.length - 1; orderIndex >= 0; orderIndex -= 1) {
    const node = prepared.order[orderIndex]!;
    if (!targetAncestors.has(node.id)) continue;
    const outputGradient = grads.get(node.id);
    if (!outputGradient) continue;
    if (node.op === 'input' || node.op === 'const') continue;

    const inputIds = node.inputs!;
    const inputs = inputIds.map((id) => values.get(id)!);

    if (node.op === 'add') {
      accumulateGradient(grads, prepared.nodes, inputIds[0]!, reduceToShape(outputGradient, inputs[0]!.shape));
      accumulateGradient(grads, prepared.nodes, inputIds[1]!, reduceToShape(outputGradient, inputs[1]!.shape));
    } else if (node.op === 'mul') {
      const leftGradient = multiplyBroadcast(outputGradient, inputs[1]!);
      const rightGradient = multiplyBroadcast(outputGradient, inputs[0]!);
      assertFiniteTensor(leftGradient, `gradient for ${inputIds[0]}`);
      assertFiniteTensor(rightGradient, `gradient for ${inputIds[1]}`);
      accumulateGradient(grads, prepared.nodes, inputIds[0]!, reduceToShape(leftGradient, inputs[0]!.shape));
      accumulateGradient(grads, prepared.nodes, inputIds[1]!, reduceToShape(rightGradient, inputs[1]!.shape));
    } else if (node.op === 'relu') {
      const inputGradient = zeroTensor(inputs[0]!.shape);
      for (let index = 0; index < inputGradient.data.length; index += 1) {
        inputGradient.data[index] = inputs[0]!.data[index]! > 0 ? outputGradient.data[index]! : 0;
      }
      assertFiniteTensor(inputGradient, `gradient for ${inputIds[0]}`);
      accumulateGradient(grads, prepared.nodes, inputIds[0]!, inputGradient);
    } else if (node.op === 'sum') {
      const inputGradient = zeroTensor(inputs[0]!.shape);
      inputGradient.data.fill(outputGradient.data[0]!);
      assertFiniteTensor(inputGradient, `gradient for ${inputIds[0]}`);
      accumulateGradient(grads, prepared.nodes, inputIds[0]!, inputGradient);
    } else if (node.op === 'matmul') {
      const gradients = matmulBackward(
        inputs[0]!,
        inputs[1]!,
        outputGradient,
        inputIds,
        prepared.nodes
      );
      if (gradients.left) accumulateGradient(grads, prepared.nodes, inputIds[0]!, gradients.left);
      if (gradients.right) accumulateGradient(grads, prepared.nodes, inputIds[1]!, gradients.right);
    }
  }

  return grads;
}

function accumulateGradient(
  grads: Map<string, InternalTensor>,
  nodes: ReadonlyMap<string, GraphNode>,
  id: string,
  contribution: InternalTensor
): void {
  if (nodes.get(id)!.op === 'const') return;
  const existing = grads.get(id);
  if (!existing) {
    grads.set(id, contribution);
    return;
  }
  for (let index = 0; index < existing.data.length; index += 1) {
    existing.data[index] = existing.data[index]! + contribution.data[index]!;
  }
  assertFiniteTensor(existing, `accumulated gradient for ${id}`);
}

function reduceToShape(gradient: InternalTensor, targetShape: readonly number[]): InternalTensor {
  const reduced = zeroTensor(targetShape);
  for (let sourceIndex = 0; sourceIndex < gradient.data.length; sourceIndex += 1) {
    const targetIndex = linearIndex(targetShape, gradient.shape, sourceIndex);
    reduced.data[targetIndex] = reduced.data[targetIndex]! + gradient.data[sourceIndex]!;
  }
  assertFiniteTensor(reduced, 'broadcast gradient reduction');
  return reduced;
}

function matmulBackward(
  left: InternalTensor,
  right: InternalTensor,
  outputGradient: InternalTensor,
  inputIds: readonly string[],
  nodes: ReadonlyMap<string, GraphNode>
): {left?: InternalTensor; right?: InternalTensor} {
  const m = left.shape[0]!;
  const k = left.shape[1]!;
  const n = right.shape[1]!;
  const leftNeeded = nodes.get(inputIds[0]!)!.op !== 'const';
  const rightNeeded = nodes.get(inputIds[1]!)!.op !== 'const';
  const leftGradient = leftNeeded ? zeroTensor(left.shape) : undefined;
  const rightGradient = rightNeeded ? zeroTensor(right.shape) : undefined;

  for (let row = 0; row < m; row += 1) {
    for (let common = 0; common < k; common += 1) {
      let leftValue = 0;
      for (let column = 0; column < n; column += 1) {
        const gradient = outputGradient.data[row * n + column]!;
        leftValue += gradient * right.data[common * n + column]!;
        if (rightGradient) {
          const target = common * n + column;
          rightGradient.data[target] = rightGradient.data[target]! + left.data[row * k + common]! * gradient;
        }
      }
      if (leftGradient) {
        leftGradient.data[row * k + common] = leftValue;
      }
    }
  }
  if (leftGradient) assertFiniteTensor(leftGradient, `matmul gradient for ${inputIds[0]}`);
  if (rightGradient) assertFiniteTensor(rightGradient, `matmul gradient for ${inputIds[1]}`);
  return {
    ...(leftGradient ? {left: leftGradient} : {}),
    ...(rightGradient ? {right: rightGradient} : {}),
  };
}

function toOutputTensor(tensor: InternalTensor): OutputTensor {
  return {
    shape: [...tensor.shape],
    data: Array.from(tensor.data),
  };
}
