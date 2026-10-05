"""Extract session-specific spikes from an ADC-synchronized joint Kilosort sort.

Run this script as before; edit the path variables and SESSION_NAMES below.
ROOT points directly to OneBox-ADC; its parent contains the probe folders.
Requires adc_spike_times*.npy from the EXACT Kilosort/Phy arrays being read,
with identical spike-row order. Equal array lengths alone cannot prove this.

Outputs, for each probe (and shank, if sorted separately):
    ROOT.parent/sessions/session_04/neural/<probe>/[shank_N]/
        adc_spike_times.npy         session-local ADC seconds
        spike_clusters.npy         unchanged joint-sort cluster IDs
        spike_times_global.npy     original concatenated PROBE sample indices
        original_spike_indices.npy rows in the original Kilosort arrays
        cluster*.tsv               unchanged cluster metadata, when available

Each session also has session_info.npz and an empty stimulus/ folder. Put its
square-presentation log and photodiode-derived on_list_times.npy there later.
Spikes and photodiode times must use the SAME session-local ADC clock:
    local_adc_seconds = global_adc_seconds - adc_global_start_s
Apply this offset to times on the concatenated ADC clock, not Psychtoolbox
GetSecs timestamps. Original-recording photodiode timestamps need their clock
relationship established before use; do not blindly subtract the offset twice.

This extracts spike arrays, not raw continuous.dat or a complete Phy project.
It trusts the existing synchronization and leaves original inputs untouched.
Tiny floating-point differences at timestamp joins are reconciled to the next
session's stored start; original boundaries are retained in the metadata.
Larger timestamp gaps are kept, with gap spikes unassigned. Overlaps fail.
The previous assignment NPZ is retained, including all unassigned spike rows.
Reruns replace generated files. Dependencies: numpy and the standard library.
"""

from pathlib import Path
import re
import shutil

import numpy as np

# Build the path directly to the folder containing periods_*.npy.
data_PATH = r"R:\Basic_Sciences\Phys\SenzaiLab\Aparna\Data_by_mouse"
mouseID = r"m002"
pipeline_output_PATH = r"small-spread-neuropixel-config"
date_PATH = r"2026-09-25"
pipeline_session_output_PATH = r"ad_session_2026_09_25"
onebox_PATH = r"OneBox-ADC"

ROOT = (
    Path(data_PATH)
    / mouseID
    / pipeline_output_PATH
    / date_PATH
    / pipeline_session_output_PATH
    / onebox_PATH
)

# Optional: actual session names in the EXACT concatenation order.
# These are labels; output folders remain session_01, session_02, ... .
SESSION_NAMES = [
    "free-360-landmarks",
    "free-360-stars",
    "free-openfield",
    "head-fixed-360-RF",
    "head-fixed-360-stars",
]
# SESSION_NAMES = None  # Use this for another dataset with unnamed periods.

# These optional arrays are known to have one row per exported Kilosort spike.
# Add other verified spike-row arrays here if needed (e.g. "pc_features.npy").
# Do not add template-level/channel-level arrays or pre-filter kept_spikes.npy.
EXTRA_SPIKE_ARRAYS = (
    "amplitudes.npy",
    "spike_templates.npy",
    "spike_positions.npy",
    "spike_detection_templates.npy",
)


def load_vector(path: Path) -> np.ndarray:
    """Read a vector, accepting (N,), (N, 1), or (1, N)."""
    x = np.load(path, mmap_mode="r", allow_pickle=False)
    if x.ndim not in (1, 2) or (x.ndim == 2 and 1 not in x.shape):
        raise ValueError(f"Expected a vector in {path}; got {x.shape}")
    return x.reshape(-1)


def assign_sessions(times: np.ndarray, periods: np.ndarray):
    """Use [start, end); a spike at a join belongs to the following session."""
    candidate = np.searchsorted(periods[:, 0], times, side="right") - 1
    safe = np.clip(candidate, 0, len(periods) - 1)
    valid = np.isfinite(times) & (candidate >= 0) & (times < periods[safe, 1])
    session_id = np.where(valid, candidate, -1).astype(np.int32)
    local_time_s = np.full(times.shape, np.nan, dtype=np.float64)
    local_time_s[valid] = times[valid] - periods[candidate[valid], 0]
    return session_id, local_time_s


def prepare_timestamp_periods(stored: np.ndarray):
    """Reconcile roundoff at joins without hiding actual gaps or overlaps.

    The upstream pipeline calculates a period end and the following start with
    different addition orders. Allow eight float64 epsilon steps at the scale
    of each join; never use a broad relative tolerance on elapsed seconds.
    Starts (the session-local time origins) remain exactly as stored.
    """
    if stored.ndim != 2 or stored.shape[1] != 2 or len(stored) == 0:
        raise ValueError("periods_timestamps.npy must have shape (N, 2).")
    if not np.issubdtype(stored.dtype, np.floating):
        raise ValueError("periods_timestamps.npy must contain ADC seconds.")
    periods = stored.astype(np.float64, copy=True)
    if not np.isfinite(periods).all() or np.any(periods[:, 1] <= periods[:, 0]):
        raise ValueError("Invalid timestamp boundaries.")
    if np.any(np.diff(periods[:, 0]) <= 0):
        raise ValueError("Timestamp period starts must be strictly increasing.")

    previous_ends = periods[:-1, 1]
    next_starts = periods[1:, 0]
    join_delta_s = next_starts - previous_ends
    scale = np.maximum(1.0, np.maximum(np.abs(previous_ends), np.abs(next_starts)))
    tolerance_s = 8 * np.finfo(np.float64).eps * scale
    overlaps = np.flatnonzero(join_delta_s < -tolerance_s)
    if overlaps.size:
        details = "\n".join(
            f"  Sessions {i + 1}/{i + 2}: previous end={previous_ends[i]:.17g}, "
            f"next start={next_starts[i]:.17g}, "
            f"overlap={-join_delta_s[i]:.17g} s "
            f"(roundoff tolerance={tolerance_s[i]:.3g} s)"
            for i in overlaps
        )
        raise ValueError(
            "Timestamp periods overlap beyond floating-point roundoff; "
            "session assignment would be ambiguous.\n" + details
        )

    roundoff_joins = np.abs(join_delta_s) <= tolerance_s
    periods[:-1, 1] = np.where(roundoff_joins, next_starts, previous_ends)
    return periods, join_delta_s, tolerance_s


def save_npy(path: Path, array: np.ndarray) -> None:
    """Replace a generated file only after its new contents have been written."""
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_npz(path: Path, **arrays) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_job(time_file: Path, ks_dir: Path, shank):
    """Validate matching spike arrays before creating session exports."""
    times = load_vector(time_file)
    spike_samples = load_vector(ks_dir / "spike_times.npy")
    clusters = load_vector(ks_dir / "spike_clusters.npy")
    if not np.issubdtype(times.dtype, np.floating):
        raise ValueError(f"Expected synchronized seconds in {time_file}.")
    if not np.issubdtype(spike_samples.dtype, np.integer):
        raise ValueError(f"Expected integer Kilosort samples in {ks_dir}.")
    if not np.issubdtype(clusters.dtype, np.integer):
        raise ValueError(f"Expected integer cluster IDs in {ks_dir}.")
    if not (times.size == spike_samples.size == clusters.size):
        raise ValueError(
            f"Spike count mismatch for {time_file}: ADC={times.size}, "
            f"Kilosort={spike_samples.size}, clusters={clusters.size}. "
            "Use matching files from the same sorting/curation output."
        )

    extras = {}
    for filename in EXTRA_SPIKE_ARRAYS:
        path = ks_dir / filename
        if not path.is_file():
            continue
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.ndim == 0 or array.shape[0] != times.size:
            raise ValueError(
                f"Spike-row mismatch for {path}: shape={array.shape}, "
                f"expected first dimension {times.size}."
            )
        extras[filename] = array

    sort_key = time_file.parent.name
    if shank is not None:
        sort_key += f"/shank_{shank}"
    return {
        "time_file": time_file,
        "ks_dir": ks_dir,
        "shank": shank,
        "sort_key": sort_key,
        "times": times,
        "spike_samples": spike_samples,
        "clusters": clusters,
        "extras": extras,
        "label_files": sorted(ks_dir.glob("cluster*.tsv")),
    }


def main(root: Path = ROOT, session_names=None) -> None:
    """Read periods from root (the ADC folder) and export beside that folder."""
    adc_dir = Path(root).resolve()
    pipeline_dir = adc_dir.parent
    samples = np.load(adc_dir / "periods_samples.npy", allow_pickle=False)
    stored_periods = np.load(adc_dir / "periods_timestamps.npy", allow_pickle=False)
    periods, join_delta_s, join_tolerance_s = prepare_timestamp_periods(stored_periods)

    if samples.shape != periods.shape or not np.issubdtype(samples.dtype, np.integer):
        raise ValueError("periods_samples.npy must be an integer array of the same shape.")
    if np.any(samples[:, 1] <= samples[:, 0]):
        raise ValueError("Invalid ADC sample boundaries.")
    if not np.array_equal(samples[1:, 0], samples[:-1, 1]):
        raise ValueError("Expected contiguous ADC sample periods; integer boundaries must match exactly.")
    if samples[0, 0] < 0:
        raise ValueError("ADC sample boundaries must be nonnegative.")

    adjusted = np.flatnonzero(periods[:-1, 1] != stored_periods[:-1, 1])
    for i in adjusted:
        print(
            f"Reconciled timestamp roundoff at sessions {i + 1}/{i + 2}: "
            f"next start - stored end = {join_delta_s[i]:.17g} s."
        )
    for i in np.flatnonzero(join_delta_s > join_tolerance_s):
        print(
            f"NOTE: ADC timestamp gap at sessions {i + 1}/{i + 2}: "
            f"{join_delta_s[i]:.17g} s; spikes in this gap stay unassigned."
        )

    n_sessions = len(periods)
    if session_names is None:
        session_names = [f"session_{i + 1:02d}" for i in range(n_sessions)]
    names = np.asarray(session_names, dtype=str)
    if names.ndim != 1 or names.size != n_sessions:
        raise ValueError(f"Supply a one-dimensional list of exactly {n_sessions} names.")
    if len(set(names)) != n_sessions or any(not name.strip() for name in names):
        raise ValueError("Session names must be unique and nonempty.")

    # Search only direct probe folders, not the large raw-data directory tree.
    # Preflight every sorting output before writing any session files.
    jobs = []
    for probe_dir in sorted(pipeline_dir.iterdir()):
        if not probe_dir.is_dir() or probe_dir == adc_dir or probe_dir.name == "sessions":
            continue
        for time_file in sorted(probe_dir.glob("adc_spike_times*.npy")):
            match = re.fullmatch(r"adc_spike_times(?:_(\d+))?\.npy", time_file.name)
            if match is None:
                continue
            shank = match.group(1)
            ks_dir = probe_dir / "kilosort"
            if shank is not None:
                ks_dir /= f"shank_{shank}"
            jobs.append(load_job(time_file, ks_dir, shank))

    if not jobs:
        raise FileNotFoundError(
            f"No adc_spike_times.npy or adc_spike_times_<shank>.npy found "
            f"inside direct probe folders under {pipeline_dir}. "
            "Do not substitute raw Kilosort spike_times.npy: it uses probe samples, "
            "not ADC seconds. Check where the synchronized outputs were saved."
        )

    print("Session IDs are zero-based; -1 means unassigned.")
    print("\nID  Name                     Start (s)      End (s)  Duration (min)")
    for i, (start, end) in enumerate(periods):
        print(f"{i:2d}  {names[i]:24s} {start:11.3f} {end:12.3f} {(end-start)/60:15.3f}")

    output_root = pipeline_dir / "sessions"
    session_dirs = [output_root / f"session_{i + 1:02d}" for i in range(n_sessions)]
    counts_by_sort = np.zeros((n_sessions, len(jobs)), dtype=np.int64)

    for job_index, job in enumerate(jobs):
        time_file, ks_dir = job["time_file"], job["ks_dir"]
        times, clusters = job["times"], job["clusters"]
        session_id, local_time_s = assign_sessions(times, periods)
        valid = session_id >= 0
        counts = np.bincount(session_id[valid], minlength=n_sessions)
        counts_by_sort[:, job_index] = counts

        print(f"\n{job['sort_key']}: {times.size:,} spikes")
        for i, session_dir in enumerate(session_dirs):
            # Use these identical original row indices for EVERY spike array.
            rows = np.flatnonzero(session_id == i)
            local_times = local_time_s[rows]
            duration = periods[i, 1] - periods[i, 0]
            if not np.all((local_times >= 0) & (local_times < duration)):
                raise ValueError(f"Invalid local times for {job['sort_key']}, session {i + 1}.")

            neural_dir = session_dir / "neural" / time_file.parent.name
            if job["shank"] is not None:
                neural_dir /= f"shank_{job['shank']}"
            neural_dir.mkdir(parents=True, exist_ok=True)
            (session_dir / "stimulus").mkdir(exist_ok=True)

            save_npy(neural_dir / "adc_spike_times.npy", local_times)
            save_npy(neural_dir / "spike_clusters.npy", clusters[rows])
            # These remain GLOBAL probe samples; never label them spike_times.npy.
            save_npy(neural_dir / "spike_times_global.npy", job["spike_samples"][rows])
            save_npy(neural_dir / "original_spike_indices.npy", rows.astype(np.int64))
            for filename, array in job["extras"].items():
                save_npy(neural_dir / filename, array[rows])
            for label_file in job["label_files"]:
                shutil.copy2(label_file, neural_dir / label_file.name)
            print(f"  {names[i]}: {rows.size:,} spikes -> {neural_dir}")

        # Keep the earlier output, including spike rows not assigned to a session.
        suffix = "" if job["shank"] is None else f"_shank_{job['shank']}"
        assignment_dir = time_file.parent / "session_assignments"
        assignment_dir.mkdir(exist_ok=True)
        save_npz(
            assignment_dir / f"spike_session_assignments{suffix}.npz",
            spike_session_id=session_id,
            spike_times_session_s=local_time_s,
            spike_clusters=clusters,
            unassigned_original_spike_indices=np.flatnonzero(~valid),
            session_names=names,
            periods_timestamps=periods,
            periods_timestamps_stored=stored_periods,
            timestamp_join_delta_s=join_delta_s,
            timestamp_join_tolerance_s=join_tolerance_s,
            periods_samples_adc=samples,
            source_adc_spike_times=str(time_file),
            source_kilosort_dir=str(ks_dir),
        )
        print(f"  Unassigned: {np.count_nonzero(~valid):,}")
        if not valid.all():
            print("  WARNING: invalid/out-of-range times retained in the assignment NPZ as ID=-1.")
        if not (ks_dir / "cluster_KSLabel.tsv").is_file():
            print("  NOTE: cluster_KSLabel.tsv is absent; supply labels if RF analysis needs them.")

    # One metadata file per session; counts remain separate for each probe/shank.
    for i, session_dir in enumerate(session_dirs):
        save_npz(
            session_dir / "session_info.npz",
            session_id=np.int32(i),
            session_number=np.int32(i + 1),
            session_name=names[i],
            adc_global_start_s=periods[i, 0],
            adc_global_end_s=periods[i, 1],
            adc_global_end_s_stored=stored_periods[i, 1],
            timestamp_end_adjustment_s=periods[i, 1] - stored_periods[i, 1],
            adc_global_start_sample=samples[i, 0],
            adc_global_end_sample=samples[i, 1],
            duration_s=periods[i, 1] - periods[i, 0],
            time_reference="session_local_adc_seconds",
            boundary_convention="[start, end)",
            sort_names=np.asarray([job["sort_key"] for job in jobs], dtype=str),
            n_spikes_per_sort=counts_by_sort[i],
            source_adc_spike_times=np.asarray([str(job["time_file"]) for job in jobs]),
            source_kilosort_dirs=np.asarray([str(job["ks_dir"]) for job in jobs]),
            source_adc_dir=str(adc_dir),
        )

    print(f"\nExtracted sessions: {output_root}")
    print("For RF mapping, use the chosen session's ADC-local spikes and cluster IDs.")
    print("Put its trial log and ADC-local photodiode onset times in its stimulus folder.")
    print("Use session_info.npz for the ADC global-to-local offset; no probe sample conversion is needed.")


if __name__ == "__main__":
    main(ROOT, SESSION_NAMES)
