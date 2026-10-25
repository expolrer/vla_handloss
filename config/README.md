# Configuration

`project.yaml` controls workspace paths, fixed evaluation protocol and training-budget provenance.
`tasks.yaml` defines robot dimensions and the exact fixed instructions.
`experiments.yaml` defines the eight architecture/loss conditions.

Do not put credentials in YAML. Keep local paths and host-specific overrides in ignored `workspaces/local.yaml`. Changing evaluation parameters changes the protocol fingerprint and requires a new result namespace rather than silently appending old results.

`transfer.example.yaml` describes relay settings; it is not a credential store and is not executed by the evaluation queue. Existing transfer jobs keep their own manifests and verified receipts.
