# Third-Party Notices

This repository does not redistribute model weights, the downloaded GSM8K TRAIN
payload, or the published p50c NPZ edit arrays.

## Published p50c Edit

The lab studies the unmodified edit by Rikiya Takehi et al. from
[lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs), commit
`4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`,
`exp8_math/runs/p50c/edit/`. The upstream logs, data and networks are identified as
CC BY 4.0. Attribution and the distinction between local measurements and the
authors' evaluation are retained in the reports.

Related article: [Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/).

## Model

[Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), revision
`2fc06364715b967f1860aea9cf38778875588b17`, is identified as Apache-2.0.
Obtain the model separately and retain its upstream notices.

## Dataset

[openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k), `main` TRAIN only,
revision `740312add88f781978c0658806c59bc2815b9866`, is identified as MIT-licensed.
The 30 sampled questions and full source payload remain local and are not part
of this publication. Reported aggregates do not establish an uncontaminated benchmark.

## Python Environment

The exact dependency review, source indexes, versions and upstream license notes
are recorded in [docs/dependencies.md](docs/dependencies.md) and the hash-pinned
lock. No third-party environment or binary package is bundled in Git.

No license for the project's original code is implied by these notices.
