import math
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from core.oa_risk_engine import evaluate_oa_preventive_risk

# ==============================================================================
# AETHER-OA — CORE GAIT & FUNCTIONAL BIOMECHANICS ENGINE
# ==============================================================================


def calculate_angle(a, b, c):
    """Calculate angle ABC in degrees where b is the vertex."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0 or np.isnan(norm_ba) or np.isnan(norm_bc):
        return np.nan

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine_angle)))


def smooth_signal(values, window_length=15, polyorder=2):
    """Smooth joint angle trajectory with Savitzky-Golay filter."""
    s = pd.Series(values, dtype=float)
    s = s.interpolate(method="linear", limit=7, limit_direction="both").bfill().ffill()
    clean_vals = s.to_numpy(dtype=float)

    if len(clean_vals) < 7:
        return clean_vals

    window_length = min(window_length, len(clean_vals))
    if window_length % 2 == 0:
        window_length -= 1

    if window_length <= polyorder:
        return clean_vals

    try:
        return savgol_filter(clean_vals, window_length=window_length, polyorder=polyorder)
    except Exception:
        return clean_vals


class SitToStandTracker:
    """
    Real-time State Machine for OARSI 30-second Chair Stand Test (30s CST) / 5XSTS.
    States: SEATED -> ASCENDING -> STANDING -> DESCENDING -> SEATED
    """
    def __init__(self, fps=30.0):
        self.fps = fps
        self.state = "SEATED"
        self.reps = 0
        self.ascent_start_time = None
        self.ascent_durations = []
        self.trunk_angles = []
        self.peak_trunk_lean = 0.0

    def update(self, knee_angle, trunk_angle, timestamp):
        """Processes a single frame and updates the state machine."""
        if not np.isfinite(knee_angle):
            return self.reps, self.state

        if np.isfinite(trunk_angle):
            self.trunk_angles.append(trunk_angle)
            if trunk_angle > self.peak_trunk_lean:
                self.peak_trunk_lean = float(trunk_angle)

        # Transition Logic based on Knee Angle
        if self.state == "SEATED":
            # Rising started: knee angle begins increasing past 105°
            if knee_angle > 105.0:
                self.state = "ASCENDING"
                self.ascent_start_time = timestamp

        elif self.state == "ASCENDING":
            # Reached full standing extension
            if knee_angle >= 165.0:
                self.state = "STANDING"
                if self.ascent_start_time is not None:
                    duration = float(timestamp - self.ascent_start_time)
                    if 0.4 <= duration <= 4.0:
                        self.ascent_durations.append(duration)

        elif self.state == "STANDING":
            # Descending back to chair
            if knee_angle < 150.0:
                self.state = "DESCENDING"

        elif self.state == "DESCENDING":
            # Returned to seated position
            if knee_angle <= 105.0:
                self.state = "SEATED"
                self.reps += 1

        return self.reps, self.state

    def get_summary(self):
        mean_ascent = float(np.mean(self.ascent_durations)) if self.ascent_durations else 1.8
        mean_trunk = float(np.mean(self.trunk_angles)) if self.trunk_angles else 22.0
        return {
            "completed_reps": int(self.reps),
            "mean_ascent_sec": mean_ascent,
            "peak_trunk_lean_deg": float(self.peak_trunk_lean),
            "mean_trunk_lean_deg": mean_trunk
        }


def detect_gait_cycles(time_values, knee_angles, fps=30.0, min_distance_sec=0.5, prominence=3.0):
    """Detect individual gait cycles by identifying peak flexion valleys."""
    time_values = np.asarray(time_values, dtype=float)
    knee_angles = np.asarray(knee_angles, dtype=float)

    if len(knee_angles) < 15 or fps <= 0:
        return []

    valid = np.isfinite(knee_angles)
    if np.sum(valid) < 15:
        return []

    min_distance_frames = max(int(fps * min_distance_sec), 6)
    std_dev = float(np.nanstd(knee_angles))
    prom = max(prominence, std_dev * 0.2)

    peaks, _ = find_peaks(
        -knee_angles,
        distance=min_distance_frames,
        prominence=prom
    )

    cycles = []
    for i in range(len(peaks) - 1):
        start_idx = peaks[i]
        end_idx = peaks[i + 1]

        segment = knee_angles[start_idx:end_idx + 1]
        t_segment = time_values[start_idx:end_idx + 1]

        if len(segment) < 3:
            continue

        cycle_duration = float(t_segment[-1] - t_segment[0])
        if not (0.4 <= cycle_duration <= 3.2):
            continue

        max_extension = float(np.max(segment))
        max_flexion = float(np.min(segment))
        rom = float(max_extension - max_flexion)

        vel = np.diff(segment) * fps
        max_vel = float(np.max(np.abs(vel))) if len(vel) > 0 else 0.0

        cycles.append({
            "cycle_number": len(cycles) + 1,
            "start_time": float(t_segment[0]),
            "end_time": float(t_segment[-1]),
            "duration_sec": cycle_duration,
            "max_extension_deg": max_extension,
            "max_flexion_deg": max_flexion,
            "rom_deg": rom,
            "peak_velocity_deg_per_sec": max_vel
        })

    return cycles


def calculate_cycle_metrics(cycles):
    """Computes summary statistics across detected gait cycles."""
    if not cycles:
        return {
            "cycle_count": 0,
            "mean_cycle_duration": np.nan,
            "cycle_duration_std": np.nan,
            "cycle_duration_cv": np.nan,
            "mean_cycle_rom": np.nan,
            "cycle_rom_std": np.nan,
            "cycle_rom_cv": np.nan
        }

    durations = np.array([c["duration_sec"] for c in cycles], dtype=float)
    roms = np.array([c["rom_deg"] for c in cycles], dtype=float)

    mean_dur = float(np.mean(durations))
    std_dur = float(np.std(durations, ddof=1)) if len(durations) > 1 else 0.0
    cv_dur = (std_dur / mean_dur * 100.0) if mean_dur > 0 else np.nan

    mean_rom = float(np.mean(roms))
    std_rom = float(np.std(roms, ddof=1)) if len(roms) > 1 else 0.0
    cv_rom = (std_rom / mean_rom * 100.0) if mean_rom > 0 else np.nan

    return {
        "cycle_count": int(len(cycles)),
        "mean_cycle_duration": mean_dur,
        "cycle_duration_std": std_dur,
        "cycle_duration_cv": cv_dur,
        "mean_cycle_rom": mean_rom,
        "cycle_rom_std": std_rom,
        "cycle_rom_cv": cv_rom
    }


def calculate_angular_velocities(angles, fps=30.0):
    """Calculates mean flexion and extension angular velocity (°/s)."""
    angles = np.asarray(angles, dtype=float)
    valid = angles[np.isfinite(angles)]

    if len(valid) < 3 or fps <= 0:
        return np.nan, np.nan

    vel = np.diff(valid) * fps
    flexion = np.abs(vel[vel < 0])
    extension = np.abs(vel[vel > 0])

    mean_flex = float(np.mean(flexion)) if len(flexion) else np.nan
    mean_ext = float(np.mean(extension)) if len(extension) else np.nan
    return mean_flex, mean_ext


def analyze_session(
    df,
    fps=30.0,
    seated_left_flexion=None,
    seated_right_flexion=None,
    seated_left_extension=None,
    seated_right_extension=None,
    sts_completed_reps=None,
    sts_mean_ascent_sec=None,
    sts_peak_trunk_lean_deg=None,
    standing_knee_ankle_ratio=None,
    standing_baseline_ext=None
):
    """
    Comprehensive Multi-Domain Gait & Functional Kinematics Analysis.
    Integrates standing calibration, seated profiles, sit-to-stand, and dynamic walking.
    """
    if df is None or len(df) < 10:
        return {"status": "INSUFFICIENT_DATA"}

    df = df.copy()

    if "time_sec" not in df.columns:
        if "timestamp" in df.columns:
            df["time_sec"] = df["timestamp"]
        else:
            df["time_sec"] = np.arange(len(df)) / float(fps)

    time_values = df["time_sec"].to_numpy(dtype=float)

    if len(time_values) > 1 and (time_values[-1] - time_values[0]) > 0:
        effective_fps = float((len(time_values) - 1) / (time_values[-1] - time_values[0]))
    else:
        effective_fps = float(fps)

    left_raw = df["left_knee_angle"].to_numpy(dtype=float)
    right_raw = df["right_knee_angle"].to_numpy(dtype=float)

    left_smooth = smooth_signal(left_raw)
    right_smooth = smooth_signal(right_raw)

    df["left_knee_smooth"] = left_smooth
    df["right_knee_smooth"] = right_smooth

    # Signal noise
    l_mask = np.isfinite(left_raw) & np.isfinite(left_smooth)
    r_mask = np.isfinite(right_raw) & np.isfinite(right_smooth)
    left_noise = float(np.std(left_raw[l_mask] - left_smooth[l_mask])) if np.any(l_mask) else 0.0
    right_noise = float(np.std(right_raw[r_mask] - right_smooth[r_mask])) if np.any(r_mask) else 0.0

    # Range of motion
    left_valid = left_smooth[np.isfinite(left_smooth)]
    right_valid = right_smooth[np.isfinite(right_smooth)]

    left_max_ext = float(np.max(left_valid)) if len(left_valid) else 170.0
    left_min_flex = float(np.min(left_valid)) if len(left_valid) else 110.0
    left_rom = float(left_max_ext - left_min_flex)

    right_max_ext = float(np.max(right_valid)) if len(right_valid) else 170.0
    right_min_flex = float(np.min(right_valid)) if len(right_valid) else 110.0
    right_rom = float(right_max_ext - right_min_flex)

    # ROM asymmetry (%)
    denom = max(left_rom, right_rom)
    rom_asymmetry = float(abs(left_rom - right_rom) / denom * 100.0) if denom > 0 else 0.0

    # Gait cycle detection
    left_cycles = detect_gait_cycles(time_values, left_smooth, fps=effective_fps)
    right_cycles = detect_gait_cycles(time_values, right_smooth, fps=effective_fps)

    left_metrics = calculate_cycle_metrics(left_cycles)
    right_metrics = calculate_cycle_metrics(right_cycles)

    # Velocities
    left_flex_vel, left_ext_vel = calculate_angular_velocities(left_smooth, fps=effective_fps)
    right_flex_vel, right_ext_vel = calculate_angular_velocities(right_smooth, fps=effective_fps)

    duration = float(time_values[-1] - time_values[0]) if len(time_values) > 1 else 0.0
    total_cycles = len(left_cycles) + len(right_cycles)
    cadence = float(total_cycles / duration * 60.0) if duration > 0 else 0.0

    l_dur = left_metrics["mean_cycle_duration"]
    r_dur = right_metrics["mean_cycle_duration"]
    timing_asymmetry = float(abs(l_dur - r_dur) / max(l_dur, r_dur) * 100.0) if (np.isfinite(l_dur) and np.isfinite(r_dur) and max(l_dur, r_dur) > 0) else 0.0

    cv_candidates = [
        x for x in [left_metrics["cycle_duration_cv"], right_metrics["cycle_duration_cv"]]
        if np.isfinite(x)
    ]
    avg_cycle_cv = float(np.mean(cv_candidates)) if cv_candidates else 5.5

    # Coverage
    valid_samples = int(df.dropna(subset=["left_knee_angle", "right_knee_angle"]).shape[0])
    coverage = float(valid_samples / len(df) * 100.0) if len(df) else 0.0

    # COMPREHENSIVE MULTI-DOMAIN OA EVALUATION
    oa_risk = evaluate_oa_preventive_risk(
        left_rom=left_rom,
        right_rom=right_rom,
        rom_asymmetry=rom_asymmetry,
        left_max_ext=left_max_ext,
        right_max_ext=right_max_ext,
        left_flex_vel=left_flex_vel,
        right_flex_vel=right_flex_vel,
        cycle_duration_cv=avg_cycle_cv,
        timing_asymmetry=timing_asymmetry,
        
        seated_left_flexion=seated_left_flexion,
        seated_right_flexion=seated_right_flexion,
        seated_left_extension=seated_left_extension,
        seated_right_extension=seated_right_extension,
        
        sts_completed_reps=sts_completed_reps,
        sts_mean_ascent_sec=sts_mean_ascent_sec,
        sts_peak_trunk_lean_deg=sts_peak_trunk_lean_deg,
        
        standing_knee_ankle_ratio=standing_knee_ankle_ratio,
        standing_baseline_ext=standing_baseline_ext
    )

    return {
        "status": "ACCEPTED" if coverage >= 60.0 else "ACCEPTED_WITH_WARNINGS",
        "coverage": coverage,
        "dataframe": df,
        "duration_sec": duration,
        "effective_fps": effective_fps,
        
        "left_rom_deg": left_rom,
        "right_rom_deg": right_rom,
        "left_max_ext_deg": left_max_ext,
        "right_max_ext_deg": right_max_ext,
        "rom_asymmetry_percent": rom_asymmetry,
        
        "left_flexion_vel": left_flex_vel,
        "right_flexion_vel": right_flex_vel,
        "left_noise_deg": left_noise,
        "right_noise_deg": right_noise,
        
        "left_cycles": left_cycles,
        "right_cycles": right_cycles,
        "cadence_steps_per_min": cadence,
        "timing_asymmetry_percent": timing_asymmetry,
        
        "oa_risk": oa_risk
    }
