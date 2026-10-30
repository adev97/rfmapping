# Standalone session splitter

`identify_and_separate_sessions.py` splits an already synchronized joint
Kilosort sort into session inputs. It runs independently of RF, HD, waveform,
and sorting workflows. It imports no repository helpers; its only third-party
dependency is NumPy. Run commands from the repository root with a Python
environment containing NumPy. Repository validation follows `AGENTS.md`.

## Configure and run

Open `preprocessing/identify_and_separate_sessions.py` and edit the
**USER SETTINGS** section near the top:

- `PIPELINE_DIR`: the joint sorting run folder for your mouse/date.
- `ADC_DIR`: defaults to `PIPELINE_DIR / "OneBox-ADC"`; change only if needed.
- `SESSION_NAMES`: labels in concatenation order, or `None` for generic names.
- `DRY_RUN`: `True` to check inputs; `False` to export the split session files.

The settings initially select your m002 September 25, 2026 dataset with
`DRY_RUN = True`. Run the file with your editor's Run command, or from the
repository root in PowerShell:

```powershell
& .\.venv\Scripts\python.exe .\preprocessing\identify_and_separate_sessions.py
```

After reviewing the dry-run output, change `DRY_RUN = False` and run it again
to export. Reruns replace generated files at the same paths.

### Optional command-line mode

If any command-line arguments are supplied, they select command-line mode
instead of the USER SETTINGS block. This mode requires `--adc-dir` and uses
`--dry-run` to control whether files are written, as before.

PowerShell example for the existing five-session dataset:

```powershell
$pipelineDir = 'R:\Basic_Sciences\Phys\SenzaiLab\Aparna\Data_by_mouse\m002\small-spread-neuropixel-config\2026-09-25\ad_session_2026_09_25'
$sessionNames = @('free-360-landmarks', 'free-360-stars', 'free-openfield', 'head-fixed-360-RF', 'head-fixed-360-stars')

python preprocessing/identify_and_separate_sessions.py --adc-dir "$pipelineDir\OneBox-ADC" --session-names $sessionNames --dry-run

# After inspecting the reported counts, export the sessions:
python preprocessing/identify_and_separate_sessions.py --adc-dir "$pipelineDir\OneBox-ADC" --session-names $sessionNames
```

Use the Python interpreter for your prepared environment in place of `python`
when needed. `--help` lists the command-line options.

| Option | Meaning |
| --- | --- |
| `--adc-dir PATH` | Required folder containing `periods_samples.npy` and `periods_timestamps.npy` |
| `--pipeline-dir PATH` | Folder containing direct probe folders; defaults to the ADC folder's parent. Session exports go under this folder's `sessions/` |
| `--session-names NAME ...` | Optional unique labels in exact concatenation order; omit to use `session_01`, `session_02`, etc. Labels never change folder names |
| `--dry-run` | Validate boundaries, labels, and spike-array dimensions; report assignment counts without writing files |

For timing files stored separately from sorting outputs:

```powershell
python preprocessing/identify_and_separate_sessions.py --adc-dir 'D:\timing\OneBox-ADC' --pipeline-dir $pipelineDir --dry-run
```

The splitter automatically finds all direct probe folders containing
`adc_spike_times.npy` or `adc_spike_times_<shank>.npy`. Each must have matching
`kilosort/` or `kilosort/shank_<N>/` arrays. Paths containing spaces are supported.
`--dry-run` checks structural compatibility; it cannot prove that synchronized
times and Kilosort arrays originated from the same sort or curation.

## Preserved output contract

```text
<pipeline-dir>/
  sessions/
    session_01/
      session_info.npz
      stimulus/
      neural/ProbeA/[shank_N]/
        adc_spike_times.npy
        spike_clusters.npy
        spike_times_global.npy
        original_spike_indices.npy
        cluster*.tsv
        <available supported spike-row arrays>
    session_02/
    ...
  ProbeA/session_assignments/
    spike_session_assignments*.npz
```

Assignment files are saved beside their source synchronized spike-time files,
as before. Actual filenames have no spaces: `spike_session_assignments.npz`
or, for example, `spike_session_assignments_shank_2.npz`.

Exported spike times are session-local ADC seconds:
`local = synchronized_global_ADC_seconds - session_ADC_start`.
Session intervals are `[start, end)`. Cluster IDs remain joint-sort IDs, and
all spike-row arrays use the same selected original indices. Global probe
sample indices remain explicitly named `spike_times_global.npy`.

The existing roundoff handling, gap/unassigned-spike records, label copying,
and metadata fields are preserved. Real timestamp overlaps are rejected.
Reruns replace generated files; original input arrays are unchanged. The
script creates the `stimulus/` folder but does not generate trial logs or
photodiode boundaries. Existing RF and HD consumers keep their current paths.

## Synthetic validation

On the validation server, from `~/Developer/rfmapping` after making the same
source and test files available:

```bash
~/.virtualenvs/rfmapping/bin/python tests/test_session_splitter.py
```

Checks cover configurable paths, session-local timing and boundary assignment,
joint-sort identity and extra arrays, metadata, shank folders, dry-run behavior,
and rejection of invalid labels or mismatched arrays. They use temporary
synthetic files and do not read recordings.
