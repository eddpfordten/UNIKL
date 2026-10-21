export type Lm = { x: number; y: number; z: number }
export type Hand = { label: string; lm: Lm[] }

export type Drive = {
  label: string
  rotate: boolean
  sliceDelta: number
  zoomDelta: number
  reset: boolean
  dwell: boolean
  cursor: { x: number; y: number } | null
  pinching: boolean
}

const TIPS = [4, 8, 12, 16, 20]
const PIPS = [2, 6, 10, 14, 18]

function fingerUp(lm: Lm[], index: number): boolean {
  if (index === 0) {
    const tip = Math.hypot(lm[4].x - lm[0].x, lm[4].y - lm[0].y)
    const mid = Math.hypot(lm[2].x - lm[0].x, lm[2].y - lm[0].y)
    return tip > mid + 0.02
  }
  return lm[TIPS[index]].y < lm[PIPS[index]].y - 0.03
}

function pose(lm: Lm[]): string {
  const up = [0, 1, 2, 3, 4].map((i) => fingerUp(lm, i))
  const pinch = Math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y) < 0.07
  const indexOnly = up[1] && !up[2] && !up[3] && !up[4]
  if (indexOnly) return 'dwell'
  if (pinch && !up[2]) return 'slice'
  if (!up[1] && !up[2] && !up[3] && !up[4]) return 'rotate'
  if (up[0] && up[4] && !up[1] && !up[2]) return 'reset'
  if (up[1] && up[2] && up[3] && !up[4]) return 'zoom'
  if (up[1] && up[2] && up[3] && up[4]) return 'idle'
  return 'idle'
}

export function driveFromHands(hands: Hand[], pinchY: number | null): { drive: Drive; pinchY: number | null } {
  let volume: Hand | null = null
  let pointer: Hand | null = null
  for (const hand of hands) {
    const name = pose(hand.lm)
    if (name === 'dwell' && !pointer) pointer = hand
    else volume = hand
  }
  const active = volume ?? hands[0] ?? null
  const name = active ? pose(active.lm) : 'idle'
  const y = active ? active.lm[8].y : null
  const sliceDelta = name === 'slice' && y != null && pinchY != null ? (y - pinchY) * 1.4 : 0
  const cursor = pointer
    ? { x: 1 - pointer.lm[8].x, y: pointer.lm[8].y }
    : active && name === 'slice'
      ? { x: 1 - active.lm[8].x, y: active.lm[8].y }
      : null
  return {
    pinchY: y,
    drive: {
      label: hands.length === 0 ? 'camera idle' : name,
      rotate: name === 'rotate',
      sliceDelta,
      zoomDelta: name === 'zoom' ? 0.015 : 0,
      reset: name === 'reset',
      dwell: Boolean(pointer) || name === 'dwell',
      cursor,
      pinching: name === 'slice',
    },
  }
}

export async function attachHands(
  video: HTMLVideoElement,
  onHands: (hands: Hand[]) => void,
): Promise<() => void> {
  const { FilesetResolver, HandLandmarker } = await import('@mediapipe/tasks-vision')
  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm',
  )
  const landmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
      delegate: 'CPU',
    },
    runningMode: 'VIDEO',
    numHands: 2,
  })
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: 'user', width: 640, height: 480 },
    audio: false,
  })
  video.srcObject = stream
  await video.play()
  let raf = 0
  const loop = () => {
    const now = performance.now()
    if (video.readyState >= 2) {
      const res = landmarker.detectForVideo(video, now)
      const hands: Hand[] = []
      for (let i = 0; i < res.landmarks.length; i++) {
        hands.push({
          label: res.handednesses[i]?.[0]?.categoryName ?? 'Hand',
          lm: res.landmarks[i],
        })
      }
      onHands(hands)
    }
    raf = requestAnimationFrame(loop)
  }
  raf = requestAnimationFrame(loop)
  return () => {
    cancelAnimationFrame(raf)
    for (const track of stream.getTracks()) track.stop()
    landmarker.close()
  }
}
