# Kvasir-SEG + PolypGen-WLI + PolypDB-WLI seed42 training

This directory contains the existing public-pretrained training entry points
and configuration for the three independently trained datasets:

- Kvasir-SEG
- PolypGen-WLI
- PolypDB-WLI

CVC-ClinicDB is excluded. The dataset split seed is 42; training seeds are
88, 123, and 666.

## Training conditions

- `pretrained_base`: public pretrained weights on the base split.
- `pretrained_aug3x`: public pretrained weights on the 3x training view.
- `pretrained_aug5x`: public pretrained weights on the 5x training view.
- `pretrained_aug10x`: public pretrained weights on the 10x training view.

Each condition is run independently for every dataset, training seed, and
model. The 12-model catalog is stored in `configs/models/`. Scratch training
is kept separately under the sibling `../scratch/` directory.

Training outputs belong under:

`runs/training/kvasir_seg__polypgen_wli__polypdb_wli__seed42/`

No training is launched by creating this directory.
