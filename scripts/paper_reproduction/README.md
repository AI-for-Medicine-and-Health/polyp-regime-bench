# Paper reproduction scripts

This directory contains the manuscript analysis scripts supplied in `analysis/paper_analysis_scripts.zip` and an executable WBF cache-fusion script.

## WBF

`results/wbf_validation_selection.json` records the six validation trials for each condition, seed, and dataset, along with the selected Top-6 models, validation cache hashes, and the frozen parameter choice. The paper uses equal model weights, average confidence, IoU grid `{0.20, 0.50, 0.70}`, input score threshold grid `{0.001, 0.020}`, and the top 50 candidates per model/image. Selection is by validation mAP50:95. `wbf.py --tune` reruns this grid from raw validation caches and materialized YOLO labels; `--show-grid` prints the archived trials for comparison. The six precomputed choices remain the source of the paper's reported values.

`wbf.py --output` reruns WBF over the exported per-model prediction caches using the frozen validation choice. It writes per-image fused predictions; it does not evaluate those predictions. The test split always uses the parameters selected on validation.

Install the dependency with `python -m pip install ensemble-boxes==1.0.9`. Download the prediction exports from the [Hugging Face model repository](https://huggingface.co/AI-for-Medicine-and-Health/polyp-regime-bench-models) into the paths recorded in the JSON, keeping the project-relative `runs/inference/...` tree.

Example from the project root:

```bash
python scripts/paper_reproduction/wbf.py \
  --selection scripts/paper_reproduction/results/wbf_validation_selection.json \
  --condition pretrained_aug10x --seed 88 --dataset kvasir_seg \
  --show-grid

python scripts/paper_reproduction/wbf.py \
  --selection scripts/paper_reproduction/results/wbf_validation_selection.json \
  --condition pretrained_aug10x --seed 88 --dataset kvasir_seg --tune

python scripts/paper_reproduction/wbf.py \
  --selection scripts/paper_reproduction/results/wbf_validation_selection.json \
  --condition pretrained_aug10x --seed 88 --dataset kvasir_seg --split test \
  --output runs/reproduced_wbf/pretrained_aug10x/seed88/kvasir_seg/test.json
```

The fixed-parameter Tran baseline selection record is also included as `results/wbf_fixed_selection.json`.

## Manuscript analysis scripts

`paper_analysis/analyze.py`, `figs.py`, and `tables.py` are the scripts from the supplied archive. They require the paper's `generated/` input directory described in `paper_analysis/README.md`. Those generated JSON and TeX inputs were not present in the supplied archive or project tree, so the scripts are included unchanged but cannot run from this repository alone. The figure script calls the ensemble plot Fig.  3 although the manuscript labels it Fig.  5.
