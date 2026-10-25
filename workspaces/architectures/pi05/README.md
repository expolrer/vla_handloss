# Pi0.5 Hand Modeling Overlay

Captured from the Bench2Dex policy source used by the committed training checkpoints, rather than a newly invented inference architecture. `overlay/` maps to `Bench2Dex/policy/pi05/`; `bench2dex_overlay/` maps to the Bench2Dex repository root.

The hand residual is `Linear -> GELU -> zero-initialized Linear`. It is added either to the audited 40 hand coordinates or to all action coordinates for the parameter-budget control. Weighted flow-matching loss normalizes by the sum of per-coordinate weights. Inference reconstructs action_dim, hand_indices, branch scope/width and max_token_len from each checkpoint's captured configuration.

Files retain their upstream notices. The included Apache-2.0 license applies to the OpenPI-derived source; see repository NOTICE for attribution. This overlay is source code, not weights or an environment archive.
