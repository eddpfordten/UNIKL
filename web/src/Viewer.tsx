import { useEffect, useRef, useState } from 'react'
import { attachHands, driveFromHands, type Hand } from './hands'
import { at, loadVolume, type Volume } from './volume'
import './viewer.css'

type Props = {
  title: string
  blurb: string
  sampleUrl: string
  segmentUrl?: string
}

const SEG = 96

export function Viewer({ title, blurb, sampleUrl, segmentUrl }: Props) {
  const axialRef = useRef<HTMLCanvasElement>(null)
  const sagRef = useRef<HTMLCanvasElement>(null)
  const corRef = useRef<HTMLCanvasElement>(null)
  const mipRef = useRef<HTMLCanvasElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const handsRef = useRef<Hand[]>([])
  const viewRef = useRef({
    yaw: 0.5,
    slice: 0.5,
    zoom: 1,
    pinchY: null as number | null,
    dwellStart: 0,
    dwellFired: false,
  })
  const labelRef = useRef('')
  const volRef = useRef<Volume | null>(null)
  const maskRef = useRef<Uint8Array | null>(null)
  const stopCam = useRef<(() => void) | null>(null)
  const segRef = useRef<() => void>(() => {})
  const [status, setStatus] = useState('Loading sample')
  const [gesture, setGesture] = useState('camera off')
  const [camOn, setCamOn] = useState(false)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    let alive = true
    loadVolume(sampleUrl)
      .then((vol) => {
        if (!alive) return
        volRef.current = vol
        setStatus(`Sample ${vol.nx}×${vol.ny}×${vol.nz} · ${vol.nc} channel${vol.nc > 1 ? 's' : ''}`)
      })
      .catch((err: unknown) => setStatus(err instanceof Error ? err.message : 'Could not load sample'))
    return () => {
      alive = false
    }
  }, [sampleUrl])

  useEffect(() => {
    let frame = 0
    let alive = true
    const tick = () => {
      if (!alive) return
      const { drive, pinchY } = driveFromHands(handsRef.current, viewRef.current.pinchY)
      viewRef.current.pinchY = pinchY
      if (drive.reset) {
        viewRef.current.yaw = 0.5
        viewRef.current.slice = 0.5
        viewRef.current.zoom = 1
      }
      if (drive.rotate) viewRef.current.yaw += 0.04
      viewRef.current.slice = clamp(viewRef.current.slice + drive.sliceDelta, 0, 1)
      viewRef.current.zoom = clamp(viewRef.current.zoom + drive.zoomDelta, 0.6, 2.2)
      if (drive.dwell) {
        if (!viewRef.current.dwellStart) viewRef.current.dwellStart = performance.now()
        else if (performance.now() - viewRef.current.dwellStart > 500 && !viewRef.current.dwellFired) {
          viewRef.current.dwellFired = true
          segRef.current()
          setStatus('Index dwell click')
        }
      } else {
        viewRef.current.dwellStart = 0
        viewRef.current.dwellFired = false
      }
      paint(axialRef.current, sagRef.current, corRef.current, mipRef.current, volRef.current, maskRef.current, viewRef.current)
      const next = camOn ? drive.label : 'camera off'
      if (next !== labelRef.current) {
        labelRef.current = next
        setGesture(next)
      }
      frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => {
      alive = false
      cancelAnimationFrame(frame)
    }
  }, [camOn])

  async function onFile(file: File) {
    try {
      const vol = await loadVolume(await file.arrayBuffer())
      volRef.current = vol
      maskRef.current = null
      setStatus(`Opened ${file.name} · ${vol.nx}×${vol.ny}×${vol.nz}`)
    } catch (err) {
      setStatus(err instanceof Error ? err.message : 'Could not open that file')
    }
  }

  async function toggleCam() {
    if (stopCam.current) {
      stopCam.current()
      stopCam.current = null
      handsRef.current = []
      setCamOn(false)
      return
    }
    const video = videoRef.current
    if (!video) return
    try {
      setStatus('Starting camera')
      stopCam.current = await attachHands(video, (hands) => {
        handsRef.current = hands
      })
      setCamOn(true)
      setStatus('Camera on. Fist rotates, pinch slices, three fingers zoom, shaka resets.')
    } catch (err) {
      setStatus(err instanceof Error ? err.message : 'Camera blocked')
    }
  }

  segRef.current = () => {
    void runSeg()
  }

  async function runSeg() {
    if (!segmentUrl || !volRef.current) return
    setBusy(true)
    setStatus('Running downsized segmentation…')
    try {
      const mask = await segmentVolume(volRef.current, segmentUrl)
      maskRef.current = mask
      const hits = mask.reduce((n, v) => n + (v > 0 ? 1 : 0), 0)
      setStatus(`Segmentation done. ${hits} voxels over 0 on the ${SEG}³ grid.`)
    } catch (err) {
      setStatus(err instanceof Error ? err.message : 'Segmentation failed')
    } finally {
      setBusy(false)
    }
  }

  return (
    <main className="edu" data-app={title}>
      <header className="edu__head">
        <img src="/header_brain.png" alt="" />
        <div>
          <h1>A.I. Copilot for Cancer Detection and Patient Survival Prediction</h1>
          <p>{status}</p>
        </div>
      </header>
      <div className="edu__body">
        <aside className="edu__side">
          <div className="edu__drop">
            <label className="edu__zone">
              Drop volume
              <span>Open a NIfTI, or use the bundled 4-channel sphere</span>
              <input
                type="file"
                accept=".nii,.nii.gz,application/gzip"
                onChange={(event) => {
                  const file = event.target.files?.[0]
                  if (file) void onFile(file)
                }}
              />
            </label>
            {segmentUrl ? (
              <button type="button" className="edu__run" disabled={busy} onClick={() => void runSeg()}>
                {busy ? '…' : 'RUN'}
              </button>
            ) : null}
          </div>
          <section className="edu__card">
            <h2>PATIENT RECORD</h2>
            <p>{blurb}</p>
            <p>MedSAM refine stays on the desktop app.</p>
          </section>
          <section className="edu__card">
            <h2>SURVIVAL DAYS</h2>
            <p>Desktop panel. This page runs segmentation only.</p>
          </section>
          <footer className="edu__credit">
            <img src="/unikl_logo.png" alt="UniKL" />
            <span>Developed by</span>
            <strong>UniKL MIIT A.I. Squad</strong>
          </footer>
        </aside>
        <section className="edu__slices">
          <figure>
            <figcaption>2D sagittal view</figcaption>
            <canvas ref={sagRef} width={360} height={220} />
          </figure>
          <figure>
            <figcaption>2D axial view</figcaption>
            <canvas ref={axialRef} width={360} height={220} />
          </figure>
          <figure>
            <figcaption>2D coronal view</figcaption>
            <canvas ref={corRef} width={360} height={220} />
          </figure>
        </section>
        <aside className="edu__side edu__side--right">
          <figure>
            <figcaption>3D tumor</figcaption>
            <p>Orange voxels on the slices are the segmentation. MedSAM refine is desktop-only.</p>
            <p>{gesture}</p>
          </figure>
          <figure className="edu__brain">
            <figcaption>3D brain view</figcaption>
            <canvas ref={mipRef} width={280} height={280} />
            <video ref={videoRef} className={camOn ? 'edu__cam' : 'edu__cam edu__cam--off'} muted playsInline />
            <button type="button" className="edu__cam-btn" onClick={() => void toggleCam()}>
              {camOn ? 'Stop camera' : 'Camera'}
            </button>
          </figure>
        </aside>
      </div>
    </main>
  )
}

function clamp(n: number, min: number, max: number) {
  return Math.min(max, Math.max(min, n))
}

function paint(
  axial: HTMLCanvasElement | null,
  sag: HTMLCanvasElement | null,
  cor: HTMLCanvasElement | null,
  mip: HTMLCanvasElement | null,
  vol: Volume | null,
  mask: Uint8Array | null,
  view: { yaw: number; slice: number; zoom: number },
) {
  if (!vol || !axial || !sag || !cor || !mip) return
  const z = Math.round(view.slice * (vol.nz - 1))
  const y = Math.round(view.slice * (vol.ny - 1))
  const x = Math.round(view.slice * (vol.nx - 1))
  drawPlane(axial, vol, mask, 'z', z, view.zoom)
  drawPlane(sag, vol, mask, 'x', x, view.zoom)
  drawPlane(cor, vol, mask, 'y', y, view.zoom)
  drawMip(mip, vol, view.yaw, view.zoom)
}

function drawPlane(
  canvas: HTMLCanvasElement,
  vol: Volume,
  mask: Uint8Array | null,
  axis: 'x' | 'y' | 'z',
  index: number,
  zoom: number,
) {
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  const img = ctx.createImageData(canvas.width, canvas.height)
  const w = canvas.width
  const h = canvas.height
  for (let py = 0; py < h; py++) {
    for (let px = 0; px < w; px++) {
      const u = (px / w - 0.5) / zoom + 0.5
      const v = (py / h - 0.5) / zoom + 0.5
      const coords = axis === 'z'
        ? [Math.floor(u * vol.nx), Math.floor(v * vol.ny), index]
        : axis === 'y'
          ? [Math.floor(u * vol.nx), index, Math.floor(v * vol.nz)]
          : [index, Math.floor(u * vol.ny), Math.floor(v * vol.nz)]
      const [x, y, z] = coords
      const g = at(vol, x, y, z, 0)
      const hit = mask ? maskAt(mask, vol, x, y, z) : 0
      const i = (py * w + px) * 4
      img.data[i] = hit ? 255 : g
      img.data[i + 1] = hit ? 120 : g
      img.data[i + 2] = hit ? 20 : g
      img.data[i + 3] = 255
    }
  }
  ctx.putImageData(img, 0, 0)
}

function maskAt(mask: Uint8Array, vol: Volume, x: number, y: number, z: number) {
  const mx = Math.min(SEG - 1, Math.floor((x / vol.nx) * SEG))
  const my = Math.min(SEG - 1, Math.floor((y / vol.ny) * SEG))
  const mz = Math.min(SEG - 1, Math.floor((z / vol.nz) * SEG))
  return mask[mz * SEG * SEG + my * SEG + mx] ?? 0
}

function drawMip(canvas: HTMLCanvasElement, vol: Volume, yaw: number, zoom: number) {
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  const img = ctx.createImageData(canvas.width, canvas.height)
  const w = canvas.width
  const h = canvas.height
  const c = Math.cos(yaw)
  const s = Math.sin(yaw)
  const steps = vol.nz
  for (let py = 0; py < h; py++) {
    for (let px = 0; px < w; px++) {
      const u = (px / w - 0.5) / zoom
      const v = (py / h - 0.5) / zoom
      let best = 0
      for (let t = 0; t < steps; t++) {
        const depth = t / steps - 0.5
        const x = Math.round((u * c - depth * s + 0.5) * (vol.nx - 1))
        const y = Math.round((v + 0.5) * (vol.ny - 1))
        const z = Math.round((u * s + depth * c + 0.5) * (vol.nz - 1))
        const g = at(vol, x, y, z, 0)
        if (g > best) best = g
      }
      const i = (py * w + px) * 4
      img.data[i] = best
      img.data[i + 1] = Math.round(best * 0.85)
      img.data[i + 2] = Math.round(best * 0.55)
      img.data[i + 3] = 255
    }
  }
  ctx.putImageData(img, 0, 0)
}

async function segmentVolume(vol: Volume, url: string): Promise<Uint8Array> {
  const ort = await import('onnxruntime-web')
  ort.env.wasm.numThreads = 1
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/'
  const session = await ort.InferenceSession.create(url, { executionProviders: ['wasm'] })
  const input = new Float32Array(4 * SEG * SEG * SEG)
  for (let c = 0; c < 4; c++) {
    for (let z = 0; z < SEG; z++) {
      const zz = Math.min(vol.nz - 1, Math.floor((z * vol.nz) / SEG))
      for (let y = 0; y < SEG; y++) {
        const yy = Math.min(vol.ny - 1, Math.floor((y * vol.ny) / SEG))
        for (let x = 0; x < SEG; x++) {
          const xx = Math.min(vol.nx - 1, Math.floor((x * vol.nx) / SEG))
          const value = at(vol, xx, yy, zz, c) / 255
          input[((c * SEG + z) * SEG + y) * SEG + x] = value
        }
      }
    }
  }
  const tensor = new ort.Tensor('float32', input, [1, 4, SEG, SEG, SEG])
  const out = await session.run({ image: tensor })
  const first = Object.values(out)[0] as { data: Float32Array }
  const data = first.data
  let max = -Infinity
  for (let i = 0; i < data.length; i++) if (data[i] > max) max = data[i]
  const cut = max > 1 ? 0 : 0.5
  const mask = new Uint8Array(SEG * SEG * SEG)
  const voxels = Math.min(SEG * SEG * SEG, data.length)
  for (let i = 0; i < voxels; i++) mask[i] = data[i] > cut ? 1 : 0
  if (!mask.some((v) => v > 0)) {
    const sorted = Array.from(data.subarray(0, voxels)).sort((a, b) => a - b)
    const alt = sorted[Math.floor(sorted.length * 0.9)] ?? 0
    for (let i = 0; i < voxels; i++) mask[i] = data[i] >= alt ? 1 : 0
  }
  return mask
}
