declare module 'onnxruntime-web' {
  export const env: { wasm: { numThreads: number; wasmPaths: string } }
  export class Tensor {
    constructor(type: 'float32', data: Float32Array, dims: number[])
    data: Float32Array
  }
  export type OrtSession = {
    run(feeds: Record<string, Tensor>): Promise<Record<string, Tensor>>
    release(): Promise<void>
  }
  export const InferenceSession: {
    create(url: string, opts?: { executionProviders?: string[] }): Promise<OrtSession>
  }
}
