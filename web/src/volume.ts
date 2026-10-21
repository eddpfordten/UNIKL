export type Volume = {
  nx: number
  ny: number
  nz: number
  nc: number
  data: Uint8Array
}

export async function loadVolume(source: string | ArrayBuffer): Promise<Volume> {
  const buf = typeof source === 'string' ? await (await fetch(source)).arrayBuffer() : source
  const bytes = new Uint8Array(buf)
  const nifti = bytes[0] === 0x1f && bytes[1] === 0x8b ? await gunzip(buf) : buf
  return parseNifti(nifti)
}

async function gunzip(buf: ArrayBuffer): Promise<ArrayBuffer> {
  const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'))
  return new Response(stream).arrayBuffer()
}

export function parseNifti(buf: ArrayBuffer): Volume {
  const view = new DataView(buf)
  const dim0 = view.getInt16(40, true)
  const nx = view.getInt16(42, true)
  const ny = view.getInt16(44, true)
  const nz = view.getInt16(46, true)
  const nc = dim0 >= 4 ? Math.max(1, view.getInt16(48, true)) : 1
  const datatype = view.getInt16(70, true)
  const slope = view.getFloat32(112, true)
  const inter = view.getFloat32(116, true)
  const vox = view.getFloat32(108, true)
  const offset = Math.max(0, Math.round(Number.isFinite(vox) && vox > 0 ? vox : 352))
  if (nx < 1 || ny < 1 || nz < 1) throw new Error('NIfTI has no volume')
  const count = nx * ny * nz * nc
  const scale = slope === 0 ? 1 : slope
  const raw = toBytes(view, offset, count, datatype, scale, inter)
  return { nx, ny, nz, nc, data: raw }
}

function toBytes(
  view: DataView,
  offset: number,
  count: number,
  datatype: number,
  scale: number,
  inter: number,
): Uint8Array {
  if (datatype === 2) {
    const src = new Uint8Array(view.buffer, view.byteOffset + offset, count)
    return new Uint8Array(src)
  }
  const out = new Uint8Array(count)
  if (datatype === 4) {
    let min = Infinity
    let max = -Infinity
    const vals = new Float32Array(count)
    for (let i = 0; i < count; i++) {
      const v = view.getInt16(offset + i * 2, true) * scale + inter
      vals[i] = v
      if (v < min) min = v
      if (v > max) max = v
    }
    return scaleToBytes(vals, out, min, max)
  }
  if (datatype === 16) {
    let min = Infinity
    let max = -Infinity
    const vals = new Float32Array(count)
    for (let i = 0; i < count; i++) {
      const v = view.getFloat32(offset + i * 4, true) * scale + inter
      vals[i] = Number.isFinite(v) ? v : 0
      if (vals[i] < min) min = vals[i]
      if (vals[i] > max) max = vals[i]
    }
    return scaleToBytes(vals, out, min, max)
  }
  throw new Error(`Unsupported NIfTI datatype ${datatype}`)
}

function scaleToBytes(vals: Float32Array, out: Uint8Array, min: number, max: number): Uint8Array {
  const span = max - min || 1
  for (let i = 0; i < vals.length; i++) out[i] = Math.round(((vals[i] - min) / span) * 255)
  return out
}

export function at(vol: Volume, x: number, y: number, z: number, c = 0): number {
  if (x < 0 || y < 0 || z < 0 || x >= vol.nx || y >= vol.ny || z >= vol.nz) return 0
  const plane = vol.nx * vol.ny
  const channel = Math.min(c, vol.nc - 1) * plane * vol.nz
  return vol.data[channel + z * plane + y * vol.nx + x]
}
