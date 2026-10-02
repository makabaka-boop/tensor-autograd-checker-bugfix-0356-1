export type ErrorCode =
  | 'INVALID_REQUEST'
  | 'INVALID_GRAPH'
  | 'INVALID_TENSOR'
  | 'INVALID_SHAPE'
  | 'CYCLIC_GRAPH'
  | 'GRAPH_TOO_LARGE'
  | 'ELEMENT_LIMIT_EXCEEDED'
  | 'UNSUPPORTED_GRADIENT'
  | 'NON_FINITE_VALUE'
  | 'PAYLOAD_TOO_LARGE';

export class GraphError extends Error {
  readonly code: ErrorCode;

  constructor(code: ErrorCode, message: string) {
    super(message);
    this.name = 'GraphError';
    this.code = code;
  }
}
