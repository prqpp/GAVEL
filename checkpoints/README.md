# Checkpoints

Fusion-head checkpoints are local/private by default and are not committed to
GitHub.

Expected filenames:

- `refclip_fusion.pt`: default 7B fusion head.
- `refclip_fusion_32B.pt`: optional 32B ablation fusion head.

You can either place private checkpoints here locally or train a new head with
`scripts/train_fusion.py`.
