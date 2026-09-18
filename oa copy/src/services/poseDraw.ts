/**
 * Live skeleton + HUD overlay for the gait capture screen — the Canvas 2D
 * counterpart of OA_Computer_Vision/core/pose_detector.py's draw_skeleton()
 * and draw_non_overlapping_hud(). Same colour language, same layout, so the
 * feature reads as "the same tool" as the original Python kiosk:
 *   - LEFT leg: BLUE stick    - RIGHT leg: RED stick
 *   - Top-left/top-right HUD cards: live LEFT KNEE / RIGHT KNEE angle
 *   - Bottom legend: which colour is which side
 *
 * Colours are the Python COLOR_* constants (pose_detector.py) converted from
 * OpenCV's BGR tuples to CSS hex, not re-picked — e.g. COLOR_LEFT_LEG =
 * (255,120,0) BGR -> #0078FF.
 */
import type { PoseJoints, JointPoint } from './poseCapture';

const LEFT_LEG = '#0078FF';
const LEFT_JOINT = '#50C8FF';
const RIGHT_LEG = '#FF2814';
const RIGHT_JOINT = '#FF8C64';
const TORSO = '#969696';
const PELVIS = '#D2D2D2';
const HUD_BG = 'rgba(18,22,30,0.82)';
const TEXT = '#FFFFFF';

/** Maps a video's own normalized (0..1) landmark coords onto a canvas box,
 *  reproducing `object-fit:contain`'s letterbox math so the overlay lines
 *  up with the video pixels underneath regardless of aspect-ratio mismatch. */
export function containMap(
  videoW: number, videoH: number, boxW: number, boxH: number,
): (nx: number, ny: number) => [number, number] {
  if (videoW <= 0 || videoH <= 0 || boxW <= 0 || boxH <= 0) {
    return (nx, ny) => [nx * boxW, ny * boxH];
  }
  const scale = Math.min(boxW / videoW, boxH / videoH);
  const drawW = videoW * scale, drawH = videoH * scale;
  const offsetX = (boxW - drawW) / 2, offsetY = (boxH - drawH) / 2;
  return (nx, ny) => [offsetX + nx * drawW, offsetY + ny * drawH];
}

function seg(
  ctx: CanvasRenderingContext2D, a: JointPoint, b: JointPoint,
  toPx: (nx: number, ny: number) => [number, number], color: string, width: number,
): void {
  if (!a.visible || !b.visible) return;
  const [ax, ay] = toPx(a.x, a.y);
  const [bx, by] = toPx(b.x, b.y);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(ax, ay);
  ctx.lineTo(bx, by);
  ctx.stroke();
}

function joint(
  ctx: CanvasRenderingContext2D, p: JointPoint,
  toPx: (nx: number, ny: number) => [number, number], fill: string, border: string, r = 5,
): void {
  if (!p.visible) return;
  const [x, y] = toPx(p.x, p.y);
  ctx.beginPath();
  ctx.arc(x, y, r + 2, 0, Math.PI * 2);
  ctx.fillStyle = border;
  ctx.fill();
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = fill;
  ctx.fill();
}

/** Blue left leg / red right leg stick figure, same joints as
 *  pose_detector.py's draw_skeleton(). */
export function drawSkeleton(
  ctx: CanvasRenderingContext2D, j: PoseJoints,
  toPx: (nx: number, ny: number) => [number, number],
): void {
  seg(ctx, j.leftShoulder, j.rightShoulder, toPx, TORSO, 2);
  seg(ctx, j.leftShoulder, j.leftHip, toPx, TORSO, 2);
  seg(ctx, j.rightShoulder, j.rightHip, toPx, TORSO, 2);
  seg(ctx, j.leftHip, j.rightHip, toPx, PELVIS, 3);

  seg(ctx, j.leftHip, j.leftKnee, toPx, LEFT_LEG, 5);
  seg(ctx, j.leftKnee, j.leftAnkle, toPx, LEFT_LEG, 5);
  seg(ctx, j.leftAnkle, j.leftHeel, toPx, LEFT_LEG, 3);
  seg(ctx, j.leftHeel, j.leftFoot, toPx, LEFT_LEG, 3);
  seg(ctx, j.leftAnkle, j.leftFoot, toPx, LEFT_LEG, 2);
  for (const p of [j.leftHip, j.leftKnee, j.leftAnkle, j.leftHeel, j.leftFoot]) {
    joint(ctx, p, toPx, LEFT_JOINT, LEFT_LEG);
  }

  seg(ctx, j.rightHip, j.rightKnee, toPx, RIGHT_LEG, 5);
  seg(ctx, j.rightKnee, j.rightAnkle, toPx, RIGHT_LEG, 5);
  seg(ctx, j.rightAnkle, j.rightHeel, toPx, RIGHT_LEG, 3);
  seg(ctx, j.rightHeel, j.rightFoot, toPx, RIGHT_LEG, 3);
  seg(ctx, j.rightAnkle, j.rightFoot, toPx, RIGHT_LEG, 2);
  for (const p of [j.rightHip, j.rightKnee, j.rightAnkle, j.rightHeel, j.rightFoot]) {
    joint(ctx, p, toPx, RIGHT_JOINT, RIGHT_LEG);
  }
}

function pill(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, border: string): void {
  ctx.fillStyle = HUD_BG;
  ctx.fillRect(x, y, w, h);
  ctx.strokeStyle = border;
  ctx.lineWidth = 1;
  ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
}

/** Top-left "LEFT KNEE" / top-right "RIGHT KNEE" angle cards + a bottom
 *  legend, same layout as pose_detector.py's draw_non_overlapping_hud() —
 *  the sit-to-stand/calibration-specific badges from that function aren't
 *  reproduced here since this screen only ever runs the walking test. */
export function drawHud(
  ctx: CanvasRenderingContext2D, w: number, h: number,
  leftAngle: number | null, rightAngle: number | null, tracking: boolean,
): void {
  ctx.textBaseline = 'alphabetic';
  ctx.font = '600 11px system-ui, sans-serif';

  // top-left: LEFT KNEE (blue)
  pill(ctx, 10, 10, 130, 44, LEFT_LEG);
  ctx.fillStyle = TEXT;
  ctx.fillText('LEFT KNEE', 20, 26);
  ctx.fillStyle = LEFT_LEG;
  ctx.font = '700 16px system-ui, sans-serif';
  ctx.fillText(leftAngle != null ? `${leftAngle.toFixed(0)}°` : '—', 20, 46);

  // top-right: RIGHT KNEE (red)
  const rw = 130;
  pill(ctx, w - rw - 10, 10, rw, 44, RIGHT_LEG);
  ctx.fillStyle = TEXT;
  ctx.font = '600 11px system-ui, sans-serif';
  ctx.fillText('RIGHT KNEE', w - rw, 26);
  ctx.fillStyle = RIGHT_LEG;
  ctx.font = '700 16px system-ui, sans-serif';
  const rightStr = rightAngle != null ? `${rightAngle.toFixed(0)}°` : '—';
  ctx.fillText(rightStr, w - rw, 46);

  // bottom legend
  const lw = 210, lx = (w - lw) / 2, ly = h - 30;
  pill(ctx, lx, ly, lw, 22, tracking ? '#50C87850' : '#96969650');
  ctx.font = '600 10px system-ui, sans-serif';
  ctx.fillStyle = LEFT_JOINT;
  ctx.beginPath(); ctx.arc(lx + 16, ly + 11, 4, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = TEXT;
  ctx.fillText('Left', lx + 26, ly + 15);
  ctx.fillStyle = RIGHT_JOINT;
  ctx.beginPath(); ctx.arc(lx + 90, ly + 11, 4, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = TEXT;
  ctx.fillText('Right', lx + 100, ly + 15);
  ctx.fillStyle = tracking ? '#78D250' : '#FFB400';
  ctx.fillText(tracking ? '● tracking' : '○ no person', lx + 150, ly + 15);
}
