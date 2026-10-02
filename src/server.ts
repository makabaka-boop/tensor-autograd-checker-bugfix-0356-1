import {createServer, type IncomingMessage, type Server, type ServerResponse} from 'node:http';
import {GraphError} from './errors.js';
import type {ComputeRequest} from './graph.js';
import {compute} from './service.js';

export interface ServerOptions {
  maxBodyBytes?: number;
}

export function createTensorGraphServer(options: ServerOptions = {}): Server {
  const maxBodyBytes = options.maxBodyBytes ?? 1024 * 1024;
  if (!Number.isSafeInteger(maxBodyBytes) || maxBodyBytes < 1) {
    throw new GraphError('INVALID_REQUEST', 'maxBodyBytes must be a positive safe integer');
  }
  return createServer((request, response) => {
    if (request.method === 'OPTIONS' && request.url === '/compute') {
      response.writeHead(204);
      response.end();
      return;
    }
    if (request.method !== 'POST' || request.url !== '/compute') {
      sendJson(response, 404, {error: {code: 'NOT_FOUND', message: 'POST /compute is the only available route'}});
      return;
    }
    void handleCompute(request, response, maxBodyBytes);
  });
}

async function handleCompute(
  request: IncomingMessage,
  response: ServerResponse,
  maxBodyBytes: number
): Promise<void> {
  try {
    const body = await readBody(request, maxBodyBytes);
    let parsed: unknown;
    try {
      parsed = JSON.parse(body.toString('utf8'));
    } catch {
      throw new GraphError('INVALID_REQUEST', 'request body must be valid JSON');
    }
    if (!parsed || typeof parsed !== 'object') {
      throw new GraphError('INVALID_REQUEST', 'request body must be a JSON object');
    }
    const result = compute(parsed as ComputeRequest);
    sendJson(response, 200, result);
  } catch (error) {
    if (error instanceof GraphError) {
      sendJson(response, statusForCode(error.code), {error: {code: error.code, message: error.message}});
      return;
    }
    sendJson(response, 500, {error: {code: 'INTERNAL', message: 'internal server error'}});
  }
}

function readBody(request: IncomingMessage, maxBodyBytes: number): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let length = 0;
    let tooLarge = false;
    request.on('data', (chunk: Buffer) => {
      length += chunk.length;
      if (length > maxBodyBytes) {
        tooLarge = true;
        return;
      }
      chunks.push(chunk);
    });
    request.on('end', () => {
      if (tooLarge) {
        reject(new GraphError('PAYLOAD_TOO_LARGE', `request body exceeds ${maxBodyBytes} bytes`));
      } else {
        resolve(Buffer.concat(chunks));
      }
    });
    request.on('error', (error) => reject(error));
  });
}

function statusForCode(code: string): number {
  if (code === 'PAYLOAD_TOO_LARGE') return 413;
  return 400;
}

function sendJson(response: ServerResponse, status: number, value: unknown): void {
  const body = JSON.stringify(value);
  response.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'content-length': Buffer.byteLength(body),
  });
  response.end(body);
}

const isDirectRun = process.argv[1]?.endsWith('/dist/src/server.js');
if (isDirectRun) {
  const port = Number.parseInt(process.env.PORT ?? '3000', 10);
  createTensorGraphServer().listen(port, () => {
    console.log(`tensor graph service listening on port ${port}`);
  });
}
