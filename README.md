# PolypRegimeBench

**Training regime shapes detector rankings and ensemble gains in white-light polyp detection.**

This project benchmarks 12 detection architectures across Kvasir-SEG, PolypGen-WLI, and PolypDB-WLI. Each architecture was evaluated with three training seeds under six conditions: scratch, pretrained baseline, 3×, 5×, and 10× offline augmentation, and a 5× repetition control. It includes code for the split protocol, augmentation, single-model training, cached prediction export, and ensemble analysis.

The study's central result is that training regime changes detector rankings. The project also compares hard voting and weighted boxes fusion against selected single models and reports inference cost. These are retrospective research benchmarks, not clinical validation.

## Repositories

- Code and project documentation: `AI-for-Medicine-and-Health/polyp-regime-bench` (this repository)
- Data manifests and source instructions: `AI-for-Medicine-and-Health/polyp-regime-bench-data` on Hugging Face
- Model checkpoints and cached predictions: `AI-for-Medicine-and-Health/polyp-regime-bench-models` on Hugging Face

The dataset repository contains manifests, not copies of third-party endoscopy images. Obtain the images from the original publishers and follow their terms. The model repository contains the 648 completed canonical `best.pt` checkpoints and per-image prediction exports. Check the Hub repository cards for the exact published revision and files.

## Quick start

Python 3.10 and a PyTorch installation appropriate for your CPU or CUDA version are recommended. The original runs used PyTorch 2.5.1, Torchvision 0.20.1, and Ultralytics 8.4.130. Install those first using the [official PyTorch selector](https://pytorch.org/get-started/locally/), then:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tools/download.py manifests
python tools/download.py model --condition pretrained_aug10x --seed 88 --dataset kvasir_seg --model rtdetr_x
python tools/verify.py --condition pretrained_aug10x --seed 88 --dataset kvasir_seg --model rtdetr_x
python tools/infer.py --checkpoint models/checkpoints/pretrained_aug10x/seed88/kvasir_seg/rtdetr_x/best.pt --source /path/to/image.jpg
```

For the dataset preparation protocol, place the officially downloaded Kvasir-SEG, PolypGen and PolypDB files in the expected `datasets/raw/` subdirectories, then run `python tools/prepare_data.py --check` followed by `python tools/prepare_data.py --base`. Augmentation is an explicit additional step: `python tools/prepare_data.py --augment`. It takes substantial disk space. The source and manifest paths are documented in the dataset card.

Single-model training uses the original experiment entry point. Check prerequisites without launching a job:

```bash
python scripts/experiments/kvasir_seg__polypgen_wli__polypdb_wli__seed42_training/pretrain/common/train_one.py \
  --dataset kvasir_seg --model rtdetr_x --condition pretrained_aug10x --seed 88 --check-only
```

Remove `--check-only` only after providing the dataset and the corresponding original pretrained weights in `assets/pretrained_weights/`. The training script writes to `runs/training/` and skips completed runs. For scratch training, use the corresponding `scratch/common/train_one.py` entry point.

## Reproducibility and attribution

The split seed is 42 and model seeds are 88, 123, and 666. All 648 canonical training runs completed. The release index associates each checkpoint with its condition, dataset, model, seed, size and SHA-256. Source dataset licenses and citation instructions are controlled by their publishers; do not assign this repository's future software license to those data. The study manuscript and author citation will be linked after publication review.
