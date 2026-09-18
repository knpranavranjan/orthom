/**
 * In-browser pose estimation for the camera-based gait walk — the JS/WASM
 * counterpart of OA_Computer_Vision/core/pose_detector.py. Same standard
 * MediaPipe Pose landmark indices, same hip-knee-ankle angle formula, so a
 * frame extracted here means the same thing to gaitAnalysis.ts's
 * /api/gait/analyze as one the Python pipeline would have produced.
 *
 * Runs fully offline: the WASM runtime and the pose_landmarker_lite model
 * are bundled under public/mediapipe/ (copied from
 * node_modules/@mediapipe/tasks-vision/wasm and downloaded from Google's
 * model store respectively — see oa copy/README or the setup notes), not
 * fetched from a CDN at runtime. If those files are missing, `init()`
 * rejects with a clear error rather than silently falling back to a network
 * fetch that would break the offline-first requirement.
 */
import { FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision';

// Standard MediaPipe Pose landmark indices — identical to
// mp.solutions.pose.PoseLandmark in pose_detector.py.
const LEFT_HIP = 23, RIGHT_HIP = 24;
const LEFT_KNEE = 25, RIGHT_KNEE = 26;
const LEFT_ANKLE = 27, RIGHT_ANKLE = 28;

export interface KneeAngleFrame {
  time_sec: number;
  left_knee_angle: number | null;
  right_knee_angle: number | null;
}

/** Angle ABC in degrees, B the vertex — same formula as
 *  pose_detector.py / gait_engine.py's calculate_angle(). */
function angleDeg(a: [number, number], b: [number, number], c: [number, number]): number | null {
  const ba: [number, number] = [a[0] - b[0], a[1] - b[1]];
  const bc: [number, number] = [c[0] - b[0], c[1] - b[1]];
  const nBa = Math.hypot(ba[0], ba[1]);
  const nBc = Math.hypot(bc[0], bc[1]);
  if (nBa === 0 || nBc === 0) return null;
  const cos = Math.max(-1, Math.min(1, (ba[0] * bc[0] + ba[1] * bc[1]) / (nBa * nBc)));
  return (Math.acos(cos) * 180) / Math.PI;
}

const MIN_VISIBILITY = 0.5;

export interface PoseCaptureOptions {
  wasmBase?: string;
  modelUrl?: string;
  /** 'CPU' is the safer default across unknown kiosk hardware/browser GPU
   *  driver support; 'GPU' is faster where available. */
  delegate?: 'CPU' | 'GPU';
}

function defaults(): Required<PoseCaptureOptions> {
  const base = import.meta.env.BASE_URL ?? '/';
  return {
    wasmBase: `${base}mediapipe/wasm`,
    modelUrl: `${base}mediapipe/models/pose_landmarker_lite.task`,
    delegate: 'CPU',
  };
}

/**
 * Wraps PoseLandmarker in VIDEO running mode. One instance per capture
 * session — call `dispose()` when done to free the WASM-side model.
 */
export class PoseCapture {
  private landmarker: PoseLandmarker | null = null;
  private readonly opts: Required<PoseCaptureOptions>;

  constructor(opts: PoseCaptureOptions = {}) {
    this.opts = { ...defaults(), ...opts };
  }

  async init(): Promise<void> {
    let fileset;
    try {
      fileset = await FilesetResolver.forVisionTasks(this.opts.wasmBase);
    } catch (e) {
      throw new Error(`pose model runtime failed to load from ${this.opts.wasmBase} — is public/mediapipe/wasm present? (${e})`);
    }
    try {
      this.landmarker = await PoseLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: this.opts.modelUrl, delegate: this.opts.delegate },
        runningMode: 'VIDEO',
        numPoses: 1,
      });
    } catch (e) {
      throw new Error(`pose model failed to load from ${this.opts.modelUrl} — is public/mediapipe/models/pose_landmarker_lite.task present? (${e})`);
    }
  }

  get ready(): boolean {
    return this.landmarker !== null;
  }

  /** One video frame in -> one knee-angle sample out (nulls if no pose, a
   *  joint's visibility is below threshold, or the underlying MediaPipe
   *  graph throws on this particular frame).
   *
   *  MediaPipe's WASM graph can throw an internal error on a single frame
   *  (observed: "ROI width and height must be > 0" when no person is in
   *  frame under certain tracking states) — verified against a fake video
   *  device, not hypothetical. Left unguarded, that exception propagates
   *  out of the recording loop's requestAnimationFrame callback and kills
   *  it silently: the UI keeps showing "recording" for the rest of the
   *  walk while zero further frames are ever collected. A bad frame must
   *  degrade to a null-angle sample, never abort the capture. */
  detect(video: HTMLVideoElement, timeSecSinceStart: number, timestampMs: number): KneeAngleFrame {
    if (!this.landmarker) throw new Error('PoseCapture.init() was not called');
    const empty: KneeAngleFrame = { time_sec: timeSecSinceStart, left_knee_angle: null, right_knee_angle: null };
    let result;
    try {
      result = this.landmarker.detectForVideo(video, timestampMs);
    } catch {
      return empty;
    }
    const lm = result.landmarks[0];
    if (!lm) return empty;

    const w = video.videoWidth, h = video.videoHeight;
    const px = (i: number): [number, number] => [lm[i].x * w, lm[i].y * h];
    const visOk = (...idx: number[]) => idx.every((i) => lm[i].visibility >= MIN_VISIBILITY);

    const left = visOk(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE) ? angleDeg(px(LEFT_HIP), px(LEFT_KNEE), px(LEFT_ANKLE)) : null;
    const right = visOk(RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE) ? angleDeg(px(RIGHT_HIP), px(RIGHT_KNEE), px(RIGHT_ANKLE)) : null;
    return { time_sec: timeSecSinceStart, left_knee_angle: left, right_knee_angle: right };
  }

  dispose(): void {
    this.landmarker?.close();
    this.landmarker = null;
  }
}
