import { useEffect, useRef } from 'react';
import type { CaptureState, MediaSlot } from '../movements';

interface Props {
  media: MediaSlot;
  state: CaptureState;
  /** live knee angle in degrees, drives the mirror during capture */
  angle?: number;
}

/**
 * The media slot is polymorphic on purpose: swap `kind` in movements.ts to
 * 'video' or 'image' and real phone footage drops in with no code change.
 * SVG stays the default because it is the only version guaranteed to render
 * fast on an SPI panel.
 */
export default function MovementFigure({ media, state, angle }: Props) {
  const looping = state === 'idle' || state === 'ready';

  if (media.kind === 'video') return <VideoFigure media={media} looping={looping} />;
  if (media.kind === 'image') return <img className="mvsvg mvmedia" src={media.src} alt="" />;

  switch (media.src) {
    case 'flexion':       return <Flexion looping={looping} state={state} angle={angle} />;
    case 'sit_to_stand':  return <SitToStand looping={looping} />;
    case 'gait':          return <Gait looping={looping} />;
    default:              return <div className="empty">—</div>;
  }
}

/**
 * Real footage. Loops while briefing, pauses the moment capture starts —
 * a looping demonstration during capture makes the patient synchronise to
 * it, which is no longer their natural movement.
 */
function VideoFigure({ media, looping }: { media: MediaSlot; looping: boolean }) {
  const ref = useRef<HTMLVideoElement>(null);
  const sources = media.sources ?? [{ src: media.src, type: 'video/mp4' }];
  const srcKey = sources.map((s) => s.src).join('|');

  // All three movement screens are the same component, so React reuses this
  // <video> element when you advance between them. Swapping <source> children
  // does NOT reload a video — without an explicit load() the element keeps
  // playing the previous movement's clip under the new movement's title.
  useEffect(() => {
    ref.current?.load();
  }, [srcKey]);

  useEffect(() => {
    const v = ref.current;
    if (!v) return;
    if (looping) void v.play().catch(() => undefined);
    else v.pause();
  }, [looping, srcKey]);

  return (
    <video
      ref={ref}
      key={srcKey}
      className="mvsvg mvmedia"
      poster={media.poster}
      autoPlay loop muted playsInline preload="auto"
      disablePictureInPicture
      aria-label="Seated knee flexion and extension"
    >
      {sources.map((s) => <source key={s.src} src={s.src} type={s.type} />)}
    </video>
  );
}

/* Movement 1 — the hard part is the RANGE, so the arc is the instruction
   and the measurement at the same time. Bounded by construction: it cannot
   draw a hyperextending knee. */
function Flexion({ looping, state, angle }: { looping: boolean; state: CaptureState; angle?: number }) {
  // 0 deg = shin hanging (flexed), -86 deg = leg extended
  const live = typeof angle === 'number' ? -Math.max(0, Math.min(86, (angle / 130) * 86)) : null;
  const style = !looping && live !== null ? { transform: `rotate(${live}deg)` } : undefined;
  return (
    <svg className="mvsvg" viewBox="0 0 220 150" role="img" aria-label="Seated knee flexion and extension">
      <path className="chair" d="M36 88 L36 40 M36 88 L88 88 M42 88 L42 126 M84 88 L84 126" />
      <path className="arc" d="M126 126 A46 46 0 0 0 172 80" />
      <path className={looping ? 'arc live fig-sweep anim' : 'arc live fig-sweep'}
            d="M126 126 A46 46 0 0 0 172 80"
            style={!looping && state === 'review' ? { strokeDashoffset: 0 } : undefined} />
      <circle className="hd" cx="66" cy="36" r="9" />
      <path className="lmb" d="M70 46 L77 80" />
      <path className="lmb thin" d="M70 50 L88 70" />
      <path className="lmb" d="M77 80 L126 80" />
      <g className={looping ? 'fig-shin anim' : 'fig-shin'} style={style}>
        <path className="lmb" d="M126 80 L126 122" />
        <path className="lmb thin" d="M126 122 L143 122" />
      </g>
      <circle cx="126" cy="80" r="4.5" fill="var(--accent)" />
    </svg>
  );
}

/* Movement 2 — the hard part is NOT USING YOUR HANDS, so the folded arms
   are the only thing in accent colour. One glance lands on the thing that
   invalidates the test if they get it wrong. */
function SitToStand({ looping }: { looping: boolean }) {
  const cls = (n: number) => `pose ${looping ? 'anim' : 'still'} p${n}`;
  return (
    <svg className="mvsvg" viewBox="0 0 220 150" role="img" aria-label="Sit to stand without using hands">
      <path className="chair" d="M36 88 L36 40 M36 88 L88 88 M42 88 L42 126 M84 88 L84 126" />
      <g className={cls(1)}>
        <circle className="hd" cx="67" cy="36" r="9" />
        <path className="lmb" d="M71 46 L78 80 M78 80 L126 80 M126 80 L126 122" />
        <path className="lmb thin" d="M126 122 L143 122" />
        <path className="arms" d="M62 56 L86 68 M86 56 L62 68" />
      </g>
      <g className={cls(2)}>
        <circle className="hd" cx="112" cy="34" r="9" />
        <path className="lmb" d="M108 43 L88 74 M88 74 L128 86 M128 86 L126 122" />
        <path className="lmb thin" d="M126 122 L143 122" />
        <path className="arms" d="M96 50 L120 62 M120 50 L96 62" />
      </g>
      <g className={cls(3)}>
        <circle className="hd" cx="126" cy="20" r="9" />
        <path className="lmb" d="M126 29 L126 122" />
        <path className="lmb thin" d="M126 122 L143 122" />
        <path className="arms" d="M114 36 L138 48 M138 36 L114 48" />
      </g>
    </svg>
  );
}

/* Movement 3 — nobody needs to be taught to walk. Show WHERE and HOW FAST.
   The travelling marker moves at normal cadence, so the animation is the
   pace instruction. Doubles as the floor-marker setup diagram. */
function Gait({ looping }: { looping: boolean }) {
  const feet = [50, 70, 90, 110, 130, 150, 170];
  return (
    <svg className="mvsvg" viewBox="0 0 220 150" role="img" aria-label="Overhead view of the marked walking path">
      <path className="mark" d="M30 44 L30 106 M190 44 L190 106" />
      <path className="trk" d="M30 75 L190 75" />
      {feet.map((x, i) => (
        <ellipse key={x} className={i % 2 ? 'fp b' : 'fp'} cx={x} cy={i % 2 ? 88 : 62} rx="5" ry="8" />
      ))}
      <g className={looping ? 'fig-walker anim' : 'fig-walker'}>
        <circle cx="30" cy="75" r="7" fill="var(--accent)" />
        <circle cx="30" cy="75" r="12" fill="none" stroke="var(--accent)" strokeWidth="2" opacity=".4" />
      </g>
      <text className="svglbl" x="18" y="122">START</text>
      <text className="svglbl" x="164" y="122">FINISH</text>
    </svg>
  );
}
