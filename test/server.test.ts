import assert from 'node:assert/strict';
import test from 'node:test';
import type {AddressInfo} from 'node:net';
import {createTensorGraphServer} from '../src/server.js';

async function postJson(url: string, value: unknown): Promise<Response> {
  return fetch(url, {
    method: 'POST',
    headers: {'content-type': 'application/json'},
    body: JSON.stringify(value),
  });
}

test('HTTP service computes output and gradients', async () => {
  const server = createTensorGraphServer();
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  try {
    const {port} = server.address() as AddressInfo;
    const response = await postJson(`http://127.0.0.1:${port}/compute`, {
      graph: {
        nodes: [
          {id: 'x', op: 'input', shape: [2]},
          {id: 'two', op: 'const', shape: [2], value: [2, 2]},
          {id: 'p', op: 'mul', inputs: ['x', 'two']},
          {id: 'y', op: 'sum', inputs: ['p']},
        ],
      },
      inputs: {x: {shape: [2], data: [0.5, 0.25]}},
      outputs: ['y'],
      gradInputs: ['x'],
    });
    assert.equal(response.status, 200);
    const body = await response.json();
    assert.deepEqual(body.outputs.y.data, [1.5]);
    assert.deepEqual(body.gradients.x.data, [2, 2]);
  } finally {
    await new Promise<void>((resolve, reject) => server.close((error?: Error) => error ? reject(error) : resolve()));
  }
});

test('HTTP service returns structured validation errors', async () => {
  const server = createTensorGraphServer();
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  try {
    const {port} = server.address() as AddressInfo;
    const response = await postJson(`http://127.0.0.1:${port}/compute`, {
      graph: {nodes: [{id: 'x', op: 'input', shape: [1]}]},
      inputs: {},
      outputs: ['missing'],
    });
    assert.equal(response.status, 400);
    const body = await response.json();
    assert.equal(body.error.code, 'INVALID_REQUEST');
  } finally {
    await new Promise<void>((resolve, reject) => server.close((error?: Error) => error ? reject(error) : resolve()));
  }
});

test('HTTP service rejects oversized payloads', async () => {
  const server = createTensorGraphServer({maxBodyBytes: 64});
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  try {
    const {port} = server.address() as AddressInfo;
    const response = await postJson(`http://127.0.0.1:${port}/compute`, {
      graph: {nodes: [{id: 'x', op: 'input', shape: [100]}]},
      inputs: {x: {shape: [100], data: Array(100).fill(1)}},
      outputs: ['x'],
    });
    assert.equal(response.status, 413);
    const body = await response.json();
    assert.equal(body.error.code, 'PAYLOAD_TOO_LARGE');
  } finally {
    await new Promise<void>((resolve, reject) => server.close((error?: Error) => error ? reject(error) : resolve()));
  }
});
