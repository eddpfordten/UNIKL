import { Viewer } from './Viewer'

export default function App() {
  return (
    <Viewer
      title="UNIKL edu-cam"
      blurb="Browser slice of the desktop viewer. The sample is a 32³ sphere copied into 4 channels. Segment resizes it to 96³ and runs the int8 ONNX. MedSAM refine is not on this page. Not a patient scan."
      sampleUrl="/sphere4.nii"
      segmentUrl="/seg.onnx"
    />
  )
}
