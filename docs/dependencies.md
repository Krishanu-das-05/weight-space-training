# Dependency Review

> Curated copy of a completed historical report. Original flat-layout paths refer to the local workbench reconstructed by `scripts/lab.py`; full raw runs are not distributed with the code. The original report is preserved locally.

Reviewed on 2026-11-04 for local research on Windows x64 / Python 3.12.

## Verdict

Aggregate: **PASS** for this local setup. No known OSV advisories were found
for any of the 130 resolved packages before installation, or by `pip-audit`
against the installed environment afterward (130 checked, zero skipped).
An advisory lookup is not a guarantee that a package has no vulnerabilities.

| Direct dependency | Exact version | License | Verdict |
| --- | --- | --- | --- |
| torch | 2.14.1+cu130 | Upstream permissive SPDX expression; see note below | PASS |
| transformers | 5.20.2 | Apache-2.0 | PASS |
| huggingface-hub | 2.3.0 | Apache-2.0 | PASS |
| safetensors | 0.8.0 | Apache-2.0 | PASS |
| ipykernel | 7.4.0 | BSD-3-Clause | PASS |
| jupyterlab | 4.6.4 | BSD | PASS |
| pip-audit | 2.10.1 | Apache-2.0 | PASS |
| truststore | 0.10.5 | MIT | PASS |

The PyTorch package declares Apache-2.0, Apache-2.0 WITH LLVM-exception,
BSD-2-Clause, BSD-3-Clause, BSL-1.0, and MIT components. Bundled native/CUDA
components retain their upstream notices and terms. This is a local
experiment, not a redistribution or product license approval.

## Checks

- Vulnerabilities: OSV batch lookup over the complete lock, followed by
  installed-environment `pip-audit` using OSV. The CUDA local-version suffix
  was also stripped for an explicit lookup of upstream `torch==2.14.1`.
- Release age: every direct version is older than seven days. Resolution
  excludes artifacts newer than `2026-10-28T00:00:00Z` for transitive
  dependencies as well.
- Maintenance: all direct upstream GitHub repositories are active and not
  archived; their identities match the registry project links.
- Pinning: exact direct and transitive versions, SHA-256 distribution hashes,
  hash-required installation, and binary wheels only. The installed
  environment was checked against every locked pin.
- Provenance: PyTorch CUDA wheels come from its explicit official
  `https://download.pytorch.org/whl/cu130` index. Other packages come from
  PyPI. Package names match the established upstream projects.
- Necessity: PyTorch/Transformers provide inference, Hub provides downloads,
  Safetensors supplies non-pickle weights, and JupyterLab/ipykernel provide
  notebooks. `pip-audit` is development tooling; `truststore` lets it use
  Windows' native trusted certificates on this network.

No security check, TLS validation, or authentication requirement was disabled.
The first direct `pip-audit` invocation could not validate this network's CA
chain; `audit_dependencies.py` resolves that with the native certificate store.

## Existing Global PyTorch

The pre-existing global `torch==2.12.1` is affected by
[GHSA-rrmf-rvhw-rf47 / CVE-2025-3000](https://osv.dev/vulnerability/GHSA-rrmf-rvhw-rf47),
fixed in 2.13.0. The detailed advisory rates it **Low**; its summary uses the
word "critical." This project uses patched 2.14.1 in `.venv`. The global
installation was not modified; upgrade it separately before using it for
other experiments.

## Repeat The Audit

```powershell
.\.venv\Scripts\python.exe audit_dependencies.py --local --vulnerability-service osv --cache-dir .cache\pip-audit-cache
```

The machine-readable result from setup is in `.cache/pip-audit.json`
(intentionally ignored by Git).
