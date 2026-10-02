# Tensor Graph Service

A small TypeScript/Node service for shape-checked scalar, rank-1, and rank-2 tensor DAGs.

## Operations

- `const` and `input`
- element-wise `add` and `mul` with NumPy-style trailing-dimension broadcasting
- rank-2 `matmul`
- full reduction `sum` to a scalar
- `relu`

The service validates references, acyclicity, rank/dimension compatibility, finite values, constant lengths, and a forward-allocation element budget. Graphs may contain at most 100 nodes. Reverse-mode gradients accumulate through shared children, sum broadcast axes, and use derivative `0` for ReLU at exactly zero.

## Build and test

```bash
npm install
npm test
npm run build
```

## HTTP service

```bash
npm start
# POST http://localhost:3000/compute
```

Request shape:

```json
{
  "graph": {
    "nodes": [
      {"id": "x", "op": "input", "shape": [2]},
      {"id": "two", "op": "const", "shape": [2], "value": [2, 2]},
      {"id": "product", "op": "mul", "inputs": ["x", "two"]},
      {"id": "y", "op": "sum", "inputs": ["product"]}
    ]
  },
  "inputs": {
    "x": {"shape": [2], "data": [0.5, 0.25]}
  },
  "outputs": ["y"],
  "gradInputs": ["x"],
  "maxTotalElements": 10000000
}
```

`gradOutput` is optional when exactly one requested output is scalar; otherwise provide the scalar output id from which gradients should be taken. Errors return structured responses such as:

For a weighted objective over several output tensors, callers may pass `gradSeeds`, keyed by requested output id. Each seed tensor has the output's shape and supplies that output's contribution to the reverse pass. When `gradSeeds` is absent, the original single-scalar `gradOutput` convention applies.

```json
{"error": {"code": "INVALID_SHAPE", "message": "..."}}
```

The service does not return partial gradients: any graph, shape, finite-value, or request failure rejects the complete computation.
