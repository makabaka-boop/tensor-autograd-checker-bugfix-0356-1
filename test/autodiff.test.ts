import assert from 'node:assert/strict';
import test from 'node:test';
import {compute} from '../src/service.js';
import {GraphError} from '../src/errors.js';
import type {ComputeRequest, GraphNode} from '../src/graph.js';

function node(
  id: string,
  op: GraphNode['op'],
  options: Partial<GraphNode> = {}
): GraphNode {
  const result: GraphNode = {id, op};
  if (options.shape !== undefined) result.shape = options.shape;
  if (options.value !== undefined) result.value = options.value;
  if (options.inputs !== undefined) result.inputs = options.inputs;
  return result;
}

function scalarLoss(request: ComputeRequest, output: string): number {
  const result = compute(request);
  const tensor = result.outputs[output]!;
  assert.equal(tensor.shape.length, 0);
  return tensor.data[0]!;
}

function checkFiniteDifferences(
  makeRequest: () => ComputeRequest,
  output: string,
  target: string,
  epsilon = 1e-6
): void {
  const baseRequest = makeRequest();
  const analytical = compute(baseRequest).gradients[target]!;
  const spec = baseRequest.inputs[target]!;
  const original = [...spec.data];

  for (let index = 0; index < original.length; index += 1) {
    const positive = makeRequest();
    positive.inputs[target] = {
      ...(spec.shape ? {shape: spec.shape} : {}),
      data: original.map((value, valueIndex) => (valueIndex === index ? value + epsilon : value)),
    };
    const negative = makeRequest();
    negative.inputs[target] = {
      ...(spec.shape ? {shape: spec.shape} : {}),
      data: original.map((value, valueIndex) => (valueIndex === index ? value - epsilon : value)),
    };

    const numerical =
      (scalarLoss(positive, output) - scalarLoss(negative, output)) / (2 * epsilon);
    const expected = analytical.data[index]!;
    assert.ok(
      Math.abs(numerical - expected) <= 1e-6 + 1e-6 * Math.abs(expected),
      `gradient mismatch for ${target}[${index}]: analytical=${expected}, numerical=${numerical}`
    );
  }
}

test('accumulates gradients through a shared subgraph and matches finite differences', () => {
  const request: ComputeRequest = {
    graph: {
      nodes: [
        node('x', 'input', {shape: [2]}),
        node('c1', 'const', {shape: [2], value: [2, 3]}),
        node('z', 'mul', {inputs: ['x', 'c1']}),
        node('c2', 'const', {shape: [2], value: [0.5, -0.2]}),
        node('add0', 'add', {inputs: ['z', 'c2']}),
        node('a', 'relu', {inputs: ['add0']}),
        node('c3', 'const', {shape: [2], value: [2, 1]}),
        node('b', 'mul', {inputs: ['z', 'c3']}),
        node('sumInput', 'add', {inputs: ['a', 'b']}),
        node('y', 'sum', {inputs: ['sumInput']}),
      ],
    },
    inputs: {x: {shape: [2], data: [0.3, 1.2]}},
    outputs: ['y'],
    gradInputs: ['x'],
  };
  const result = compute(request);
  assert.deepEqual(result.outputs.y!.shape, []);
  // z=(.6,3.6); a=(1.1,3.4); b=(1.2,3.6); dx=(2*3, 3*2)=(6,6)
  assert.deepEqual(result.gradients.x!.data, [6, 6]);
  checkFiniteDifferences(
    () => structuredClone(request),
    'y',
    'x'
  );
});

test('matrix multiplication gradients match finite differences for both operands', () => {
  const request: ComputeRequest = {
    graph: {
      nodes: [
        node('a', 'input', {shape: [2, 3]}),
        node('w', 'input', {shape: [3, 2]}),
        node('product', 'matmul', {inputs: ['a', 'w']}),
        node('bias', 'input', {shape: [2]}),
        node('biased', 'add', {inputs: ['product', 'bias']}),
        node('activated', 'relu', {inputs: ['biased']}),
        node('y', 'sum', {inputs: ['activated']}),
      ],
    },
    inputs: {
      a: {shape: [2, 3], data: [0.2, 0.4, 0.1, 0.5, 0.1, 0.3]},
      w: {shape: [3, 2], data: [0.3, 0.1, 0.2, 0.4, 0.1, 0.2]},
      bias: {shape: [2], data: [0.2, 0.3]},
    },
    outputs: ['y'],
    gradInputs: ['a', 'w'],
  };

  const result = compute(request);
  assert.deepEqual(result.outputs.y!.shape, []);
  checkFiniteDifferences(() => structuredClone(request), 'y', 'a');
  checkFiniteDifferences(() => structuredClone(request), 'y', 'w');
});

test('two-axis broadcasting sums expanded gradient axes and matches finite differences', () => {
  const request: ComputeRequest = {
    graph: {
      nodes: [
        node('x', 'input', {shape: [1, 1]}),
        node('y', 'input', {shape: [2, 3]}),
        node('scale', 'input', {shape: []}),
        node('product', 'mul', {inputs: ['x', 'y']}),
        node('shifted', 'add', {inputs: ['product', 'scale']}),
        node('activated', 'relu', {inputs: ['shifted']}),
        node('loss', 'sum', {inputs: ['activated']}),
      ],
    },
    inputs: {
      x: {shape: [1, 1], data: [0.7]},
      y: {shape: [2, 3], data: [0.2, 0.4, 0.1, 0.5, 0.1, 0.3]},
      scale: {shape: [], data: [0.25]},
    },
    outputs: ['loss'],
    gradInputs: ['x', 'y', 'scale'],
  };

  const result = compute(request);
  assert.deepEqual(result.outputs.loss!.shape, []);
  // x is multiplied by all six y values, then expanded to all six output positions.
  assert.ok(Math.abs(result.gradients.x!.data[0]! - 1.6) < 1e-12);
  checkFiniteDifferences(() => structuredClone(request), 'loss', 'x');
  checkFiniteDifferences(() => structuredClone(request), 'loss', 'y');
  checkFiniteDifferences(() => structuredClone(request), 'loss', 'scale');
});

test('defines ReLU derivative at zero as zero', () => {
  const request: ComputeRequest = {
    graph: {
      nodes: [
        node('x', 'input', {shape: [3]}),
        node('r', 'relu', {inputs: ['x']}),
        node('y', 'sum', {inputs: ['r']}),
      ],
    },
    inputs: {x: {shape: [3], data: [-0.7, 0, 1.25]}},
    outputs: ['y'],
    gradInputs: ['x'],
  };
  assert.deepEqual(compute(request).gradients.x!.data, [0, 0, 1]);
});

test('rejects NaN and infinite input values', () => {
  const request: ComputeRequest = {
    graph: {nodes: [node('x', 'input', {shape: [2]}), node('y', 'sum', {inputs: ['x']})]},
    inputs: {x: {shape: [2], data: [1, Number.NaN]}},
    outputs: ['y'],
    gradInputs: ['x'],
  };
  assert.throws(() => compute(request), (error: unknown) => error instanceof GraphError && error.code === 'NON_FINITE_VALUE'
  );
});

test('rejects non-finite values generated during forward computation', () => {
  const request: ComputeRequest = {
    graph: {
      nodes: [
        node('x', 'input', {shape: [1]}),
        node('large', 'const', {shape: [1], value: [1e308]}),
        node('over', 'mul', {inputs: ['x', 'large']}),
        node('y', 'sum', {inputs: ['over']}),
      ],
    },
    inputs: {x: {shape: [1], data: [10]}},
    outputs: ['y'],
    gradInputs: ['x'],
  };
  assert.throws(() => compute(request), (error: unknown) => error instanceof GraphError && error.code === 'NON_FINITE_VALUE'
  );
});

test('rejects incompatible matrix dimensions', () => {
  const request: ComputeRequest = {
    graph: {
      nodes: [
        node('a', 'input', {shape: [2, 3]}),
        node('b', 'input', {shape: [2, 2]}),
        node('c', 'matmul', {inputs: ['a', 'b']}),
      ],
    },
    inputs: {
      a: {shape: [2, 3], data: [1, 2, 3, 4, 5, 6]},
      b: {shape: [2, 2], data: [1, 2, 3, 4]},
    },
    outputs: ['c'],
  };
  assert.throws(() => compute(request), (error: unknown) => error instanceof GraphError && error.code === 'INVALID_SHAPE'
  );
});

test('rejects cyclic graphs', () => {
  const request: ComputeRequest = {
    graph: {
      nodes: [
        node('a', 'add', {inputs: ['x', 'b']}),
        node('b', 'add', {inputs: ['a', 'x']}),
        node('x', 'input', {shape: [1]}),
      ],
    },
    inputs: {x: {shape: [1], data: [1]}},
    outputs: ['a'],
  };
  assert.throws(() => compute(request), (error: unknown) => error instanceof GraphError && error.code === 'CYCLIC_GRAPH'
  );
});

test('enforces total forward tensor element budget before allocation', () => {
  const request: ComputeRequest = {
    graph: {
      nodes: [
        node('x', 'input', {shape: [3, 4]}),
        node('y', 'relu', {inputs: ['x']}),
      ],
    },
    inputs: {x: {shape: [3, 4], data: Array(12).fill(0.1)}},
    outputs: ['y'],
    maxTotalElements: 11,
  };
  assert.throws(() => compute(request), (error: unknown) => error instanceof GraphError && error.code === 'ELEMENT_LIMIT_EXCEEDED'
  );
});

test('does not return gradients when graph validation fails', () => {
  const request: ComputeRequest = {
    graph: {
      nodes: [
        node('x', 'input', {shape: [2]}),
        node('w', 'const', {shape: [2], value: [1, Number.POSITIVE_INFINITY]}),
        node('p', 'mul', {inputs: ['x', 'w']}),
        node('y', 'sum', {inputs: ['p']}),
      ],
    },
    inputs: {x: {shape: [2], data: [1, 2]}},
    outputs: ['y'],
    gradInputs: ['x'],
  };
  let result: unknown;
  assert.throws(() => {
    try {
      result = compute(request);
    } catch (error) {
      throw error;
    }
  }, GraphError);
  assert.equal(result, undefined);
});
