"""Prepare session-local Kilosort inputs for Kai's unchanged waveform workflow.

Only file layout and probe-sample coordinates are adapted here. Waveforms and
positions are computed by the original Utils.si_utils.generate_unit_artifacts.
"""
from __future__ import annotations

import ast
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from Utils.si_utils import read_open_ephys_probe


def _session_info(raw_file: Path, probe_name: str) -> dict:
    if raw_file.parent.parent.name != "continuous":
        raise ValueError(f"Expected an Open Ephys continuous.dat path: {raw_file}")
    return {
        "base_path": str(raw_file.parents[5]),
        "record_nodes": raw_file.parents[4].name,
        "experiment_id": raw_file.parents[3].name,
        "recording_name": raw_file.parents[2].name,
        f"continuous_probe_{probe_name}_folder": raw_file.parent.name,
    }


def _oe_leaves(node) -> list[dict]:
    if isinstance(node, list):
        return [leaf for child in node for leaf in _oe_leaves(child)]
    if not isinstance(node, dict):
        return []
    if str(node.get("class", "")).endswith("OpenEphysBinaryRecordingExtractor"):
        return [node]
    return [leaf for child in node.get("kwargs", {}).values() for leaf in _oe_leaves(child)]


def _raw_files_from_provenance(concat_dir: Path, probe_name: str) -> list[Path]:
    provenance_file = concat_dir / "provenance.json"
    provenance = json.loads(provenance_file.read_text())
    if not str(provenance.get("class", "")).endswith("ConcatenateSegmentRecording"):
        raise ValueError(
            "Expected concatenate_recordings provenance. Supply raw_probe_files explicitly "
            "in the exact original concatenation order."
        )
    parents = provenance["kwargs"]["recording_list"]
    raw_files = []
    for parent in parents:
        leaves = _oe_leaves(parent)
        if len(leaves) != 1:
            raise ValueError("Each concatenated input must resolve to one Open Ephys recording.")
        folder = Path(leaves[0]["kwargs"]["folder_path"])
        if not folder.is_absolute():
            folder = provenance_file.parent / folder
        folder = folder.resolve()
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Original recording folder is unavailable: {folder}. "
                "Set raw_probe_files to the original ordered probe continuous.dat paths."
            )
        matches = sorted(folder.glob(f"**/continuous/OneBox-*.Probe{probe_name}/continuous.dat"))
        if len(matches) != 1:
            raise ValueError(
                f"Expected one Probe{probe_name} raw file under {folder}, found {len(matches)}. "
                "Set raw_probe_files explicitly instead of guessing."
            )
        raw_files.append(matches[0].resolve())
    return raw_files


def _literal_params(path: Path) -> dict:
    values = {}
    for statement in ast.parse(path.read_text()).body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    try:
                        values[target.id] = ast.literal_eval(statement.value)
                    except (ValueError, TypeError, SyntaxError):
                        pass
    return values


def _fingerprint(paths: list[Path]) -> list[dict]:
    return [
        {"path": str(p.resolve()), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
        for p in paths
    ]


def _read_concat_metadata(concat_dir: Path) -> dict:
    """Read the sample layout without reconstructing a version-specific clock.

    This inspection needs no concatenated traces or timestamps. In particular,
    SI 0.105's file_timestamps_paths is not an argument to the 0.104 reader.
    All metadata is read in place; timestamp files and binary.json are untouched.
    """
    metadata_file = concat_dir / "binary.json"
    metadata = json.loads(metadata_file.read_text())
    if not str(metadata.get("class", "")).endswith(".BinaryRecordingExtractor"):
        raise ValueError(f"Expected binary recording metadata: {metadata_file}")
    if metadata.get("relative_paths") is not True:
        raise ValueError("Expected a saved binary folder with relative data paths.")
    kwargs = metadata["kwargs"]
    file_paths = kwargs["file_paths"]
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    if not isinstance(file_paths, list) or len(file_paths) != 1:
        raise ValueError("Expected a single concatenated recording segment.")
    declared_file = Path(file_paths[0])
    if declared_file.is_absolute():
        raise ValueError("Expected a relative binary path in the saved concat metadata.")
    cached_file = concat_dir / "traces_cached_seg0.raw"
    if not (concat_dir / declared_file).samefile(cached_file):
        raise ValueError("Concat metadata does not describe traces_cached_seg0.raw.")
    if kwargs.get("time_axis", 0) != 0:
        raise ValueError("Expected sample-major interleaved concat data (time_axis=0).")

    fs = float(kwargs["sampling_frequency"])
    channels = kwargs.get("num_channels")
    if channels is None:
        channels = kwargs["num_chan"]
    n_channels = int(channels)
    dtype = np.dtype(kwargs["dtype"])
    file_offset = int(kwargs.get("file_offset", 0))
    if (
        not np.isfinite(fs) or fs <= 0 or n_channels <= 0 or n_channels != channels
        or dtype.hasobject or dtype.itemsize <= 0 or file_offset < 0
    ):
        raise ValueError("Invalid rate/channel/dtype/offset in concat metadata.")
    data_bytes = cached_file.stat().st_size - file_offset
    frame_bytes = n_channels * dtype.itemsize
    if data_bytes <= 0 or data_bytes % frame_bytes:
        raise ValueError("Concat binary size is inconsistent with its declared sample layout.")
    return {
        "fs": fs, "n_channels": n_channels, "dtype": dtype,
        "num_frames": data_bytes // frame_bytes,
        "metadata_file": metadata_file, "cached_file": cached_file,
        "saved_version": metadata.get("version", "unspecified"),
    }


def inspect_session_inputs(
    *,
    concat_dir: str | Path,
    joint_kilosort_dir: str | Path,
    session_dir: str | Path,
    session_number: int,
    expected_raw_file: str | Path,
    probe_name: str = "A",
    raw_probe_files: list[Path] | None = None,
    expected_recordings: int = 5,
) -> dict:
    """Read and validate sources; return a plan without writing input files."""
    concat_dir = Path(concat_dir)
    joint_dir = Path(joint_kilosort_dir)
    session_dir = Path(session_dir)
    expected_raw_file = Path(expected_raw_file).resolve()
    source_names = (
        "params.py", "spike_times.npy", "spike_clusters.npy",
        "channel_map.npy", "channel_positions.npy", "ops.npy", "cluster_KSLabel.tsv",
    )
    for path in [expected_raw_file, *[joint_dir / name for name in source_names]]:
        if not path.is_file():
            raise FileNotFoundError(path)
    raw_files = (
        [Path(p).resolve() for p in raw_probe_files]
        if raw_probe_files is not None
        else _raw_files_from_provenance(concat_dir, probe_name)
    )
    if len(raw_files) != expected_recordings or not 1 <= session_number <= len(raw_files):
        raise ValueError("Recording count/session number does not match the original concatenation.")
    if not raw_files[session_number - 1].samefile(expected_raw_file):
        raise ValueError("The selected concatenation entry is not your expected RF recording.")

    concat_info = _read_concat_metadata(concat_dir)
    fs = concat_info["fs"]
    n_channels = concat_info["n_channels"]
    dtype = concat_info["dtype"]
    counts = []
    for index, raw_file in enumerate(raw_files, start=1):
        recording = read_open_ephys_probe(
            _session_info(raw_file, probe_name), probe_name, load_sync_timestamps=False
        )
        if (
            recording.get_num_segments() != 1
            or recording.get_sampling_frequency() != fs
            or recording.get_num_channels() != n_channels
            or np.dtype(recording.get_dtype()) != dtype
        ):
            raise ValueError(f"Recording {index} dimensions/rate/dtype do not match the concat.")
        counts.append(int(recording.get_num_frames()))
    if sum(counts) != concat_info["num_frames"]:
        raise ValueError("Sum of raw probe recording lengths does not equal the concat length.")

    params = _literal_params(joint_dir / "params.py")
    if params.get("sample_rate") != fs:
        raise ValueError("Joint Kilosort sampling rate does not match the probe recordings.")

    samples = np.load(joint_dir / "spike_times.npy", mmap_mode="r").reshape(-1)
    clusters = np.load(joint_dir / "spike_clusters.npy", mmap_mode="r").reshape(-1)
    if samples.size != clusters.size or not np.issubdtype(samples.dtype, np.integer):
        raise ValueError("Expected matching integer probe-sample times and cluster assignments.")
    previous = None
    for first in range(0, samples.size, 1_000_000):
        chunk = samples[first:first + 1_000_000]
        if (previous is not None and chunk.size and chunk[0] < previous) or np.any(chunk[1:] < chunk[:-1]):
            raise ValueError("Joint Kilosort spike times must be sorted.")
        if chunk.size:
            previous = chunk[-1]

    sample_start = int(sum(counts[:session_number - 1]))
    sample_stop = sample_start + counts[session_number - 1]
    row_start = int(np.searchsorted(samples, sample_start, side="left"))
    row_stop = int(np.searchsorted(samples, sample_stop, side="left"))
    if row_stop <= row_start:
        raise ValueError("No joint-sorting spikes fall inside the selected probe recording.")

    metadata = [joint_dir / name for name in ("channel_map.npy", "channel_positions.npy", "ops.npy")]
    metadata += sorted(joint_dir.glob("cluster*.tsv")) + sorted(joint_dir.glob("cluster*.csv"))
    for path in metadata:
        if path.suffix in (".csv", ".tsv"):
            with path.open(newline="") as handle:
                rows = csv.DictReader(handle, delimiter="\t" if path.suffix == ".tsv" else ",")
                for row in rows:
                    if "si_unit_id" in row:
                        alias = (row.get("si_unit_id") or "").strip()
                        cluster_id = row.get("cluster_id", row.get("id"))
                        if (
                            not alias or not np.isfinite(float(alias))
                            or cluster_id is None or float(alias) != float(cluster_id)
                        ):
                            raise ValueError(
                                "Cluster metadata would remap IDs; resolve si_unit_id before linking RF units."
                            )

    concat_files = [concat_info["metadata_file"], concat_info["cached_file"]]
    if raw_probe_files is None:
        concat_files.append(concat_dir / "provenance.json")
    fingerprint_paths = list(dict.fromkeys(
        raw_files + concat_files + [joint_dir / "params.py",
                     joint_dir / "spike_times.npy", joint_dir / "spike_clusters.npy"] + metadata
    ))
    identity = {
        "session_number": session_number, "probe": probe_name,
        "raw_file": str(expected_raw_file), "sample_start": sample_start,
        "sample_stop": sample_stop, "row_start": row_start, "row_stop": row_stop,
        "source_files": _fingerprint(fingerprint_paths),
    }
    return {
        "session_dir": session_dir, "joint_dir": joint_dir, "metadata_files": metadata,
        "raw_files": raw_files, "recording_lengths": counts, "fs": fs,
        "concat_saved_version": concat_info["saved_version"],
        "n_channels": n_channels, "dtype": str(dtype), "params": params,
        "session_info": _session_info(expected_raw_file, probe_name),
        "target_kilosort_dir": session_dir / "kilosort" / f"Probe{probe_name}" / f"kilosort_{session_number}",
        "identity": identity, "fingerprint_paths": fingerprint_paths,
    }


def prepare_session_inputs(plan: dict) -> Path:
    """Write only derived session-local sorting inputs and the required session metadata."""
    identity = plan["identity"]
    if _fingerprint(plan["fingerprint_paths"]) != identity["source_files"]:
        raise ValueError("Source files changed after inspection; inspect them again before preparing.")
    target = plan["target_kilosort_dir"]
    marker = target / "waveform_input_source.json"
    info_path = plan["session_dir"] / "data" / "session_info.json"
    info = json.loads(info_path.read_text()) if info_path.is_file() else {"session_info": {}}
    existing_info = info.get("session_info", {})
    for key, value in plan["session_info"].items():
        if key in existing_info and existing_info[key] != value:
            raise ValueError(f"Existing session metadata differs at {key}; it was left untouched.")

    if target.exists():
        if not marker.is_file() or json.loads(marker.read_text()) != identity:
            raise FileExistsError(
                f"{target} already exists with different or unmanaged inputs. "
                "Use the correct existing project or prepare into a new session output folder."
            )
        print("Reusing matching prepared session-local Kilosort inputs:", target)
    else:
        target.mkdir(parents=True)
        joint_dir = plan["joint_dir"]
        samples = np.load(joint_dir / "spike_times.npy", mmap_mode="r").reshape(-1)
        clusters = np.load(joint_dir / "spike_clusters.npy", mmap_mode="r").reshape(-1)
        start, stop = identity["row_start"], identity["row_stop"]
        local_samples = np.asarray(samples[start:stop], dtype=np.int64) - identity["sample_start"]
        if int(local_samples.min()) < 0 or int(local_samples.max()) >= identity["sample_stop"] - identity["sample_start"]:
            raise ValueError("Prepared samples fall outside the selected raw probe recording.")
        np.save(target / "spike_times.npy", local_samples)
        np.save(target / "spike_clusters.npy", clusters[start:stop])
        for path in plan["metadata_files"]:
            shutil.copy2(path, target / path.name)
        params = (
            f"dat_path = {identity['raw_file']!r}\n"
            f"n_channels_dat = {plan['n_channels']!r}\n"
            f"dtype = {plan['dtype']!r}\n"
            f"offset = 0\nsample_rate = {plan['fs']!r}\n"
            f"hp_filtered = {plan['params'].get('hp_filtered', False)!r}\n"
        )
        (target / "params.py").write_text(params)
        marker.write_text(json.dumps(identity, indent=2) + "\n")

    if not info_path.is_file():
        info_path.parent.mkdir(parents=True, exist_ok=True)
        info["session_info"] = {**existing_info, **plan["session_info"]}
        info_path.write_text(json.dumps(info, indent=2) + "\n")
    elif any(key not in existing_info for key in plan["session_info"]):
        info["session_info"] = {**existing_info, **plan["session_info"]}
        info_path.write_text(json.dumps(info, indent=2) + "\n")
    print("Session-local spike count:", identity["row_stop"] - identity["row_start"])
    print("Probe sample offset:", identity["sample_start"])
    print("Kai's session metadata:", info_path)
    return target

