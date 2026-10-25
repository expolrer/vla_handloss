# Workspaces

`architectures/pi05/` tracks the exact trained model source overlay and its license.
`Bench2Dex/`, `IsaacLab/`, `envs/`, `runtime/` and dataset assets are local checkouts or links and are ignored by Git.

All owned entrypoints resolve paths relative to this repository. Host-specific overrides go in ignored `local.yaml`, not committed source files. Model checkpoints and environment packages are never committed.
