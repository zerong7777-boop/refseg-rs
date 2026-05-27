# RefSeg-RS Runtime

A lightweight PyTorch runtime for referring image segmentation on remote-sensing imagery. The repository provides model code, dataset checks, evaluation scripts, visualization utilities, continuation-training entrypoints, and reproducible metric reports for two dataset lines.

## Features

- Query-conditioned mask decoding for language-aware local feature modulation.
- Multi-scale visual-language fusion with sparse attention-style feature interaction.
- Metric-specific checkpoint selection for mIoU/oIoU evaluation.
- Lightweight runtime for evaluation, visualization, checkpoint inspection, and continuation training.
- Dataset checking and preprocessing utilities for referring segmentation datasets.

## Repository Layout

    refseg_runtime/      Runtime package for inference, eval, visualization, and training
    scripts/             Shell entrypoints for env checks, data checks, eval, and resume training
    docs/                Environment, data preparation, running, and troubleshooting notes
    requirements/        Runtime dependency list
    results/             Reproducible evaluation summaries and JSON reports
    tools/               Helper utilities
    examples/            Local path configuration templates
    checkpoints/         Checkpoint download/placement notes

Large .pth checkpoint files are intentionally not committed to the repository. Place downloaded .state_dict.pth files under checkpoints/ following checkpoints/README.md.

## Quick Start

    python3.10 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -r requirements/runtime.txt
    cp examples/paths.env.example examples/paths.env
    ${EDITOR:-vi} examples/paths.env
    export REFSEG_PATHS_FILE="$PWD/examples/paths.env"
    source examples/env.sh
    scripts/check_env.sh

Then validate data and run evaluation:

    scripts/check_data_refer.sh
    scripts/check_data_rsrefsegrs.sh
    scripts/eval_refer_miou.sh
    scripts/eval_refer_oiou.sh
    scripts/eval_rsrefsegrs_test_miou.sh
    scripts/eval_rsrefsegrs_test_oiou.sh

See docs/DEPLOY_AND_RUN.md for full commands and smoke-test variants.

## Results

This release keeps separate checkpoint/report choices for metric-specific evaluation rather than presenting a single universal best checkpoint. JSON reports are available under results/eval_reports, with summary files in results/.

## License

See LICENSE.
