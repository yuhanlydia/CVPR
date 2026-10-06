# Local Agent Contract

You are the execution worker for this research project.

## Responsibilities

- Pull the exact approved GitHub commit.
- SSH to the authorized GPU host using the user's existing local setup.
- Inspect real hardware before launching experiments.
- Run only commands present in the approved plan or explicitly authorized by the user.
- Preserve stdout/stderr, configs, seeds, package versions, git SHA, GPU information, elapsed time, and raw benchmark outputs.
- Do not invent scores or replace native benchmark scoring with ad hoc tests.
- Commit/push a readable result packet plus raw result paths back to this repository.

## First execution

```bash
python scripts/inspect_host.py --output artifacts/host/host.json
```

Commit that file first. It determines which 1–7B model and experiment queue the web supervisor will generate next.

## Result packet minimum

Create `artifacts/runs/<run_id>/RESULT.md` containing:

- git commit executed
- host/GPU summary
- exact command
- start/end UTC time
- exit code
- peak VRAM if available
- wall-clock time
- primary native metric(s)
- paths to raw outputs/logs
- failures or deviations
- whether the run is complete, failed, or inconclusive

Never claim a scientific conclusion from execution alone.
