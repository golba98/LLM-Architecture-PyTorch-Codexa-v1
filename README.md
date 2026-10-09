# LLM-Architecture-PyTorch-Codexa-v1

Decoder-only Transformer, neutral runtime policies and native checkpoint readers.

Owns embeddings, learned/RoPE positions, causal SDPA, RMSNorm, SwiGLU, KV caches and native model-state readers. State-dictionary names and model equations are unchanged. Labels must already be shifted by Data/SFT. Generic training counters live here solely to deserialize native metadata; this package does not import the trainer.

## Development

In the sibling workspace, use `../LLM-From-Scratch/run.py --repo LLM-Architecture test`.
This selects the existing environment and sibling package sources without installing dependencies.
For a separately installed checkout, run `python -m pytest` after provisioning the documented dependencies and exact sibling version 0.1.0. These packages are local and not published to PyPI.

## Entry points

Import the package modules directly.

## Integration and assets

`../LLM-From-Scratch/compatibility.json` records the complete tested version set.
Checkpoint weights, tokenizers, datasets and generated logs are referenced by path; none are distributed in this package. Preserve tokenizer fingerprints and architecture lineage. Source provenance is in PROVENANCE.md.

## Validation and limitations

See the central VALIDATION.md for commands, results and unverified large-model checks.
The original project is preserved unchanged. No model promotion, training pipeline or remote publishing occurs as part of extraction.

## Canonical workspace integration

This repository remains independently versioned at its existing remote and is pinned as a sibling in LLM-From-Scratch/compatibility.json. Integration decisions live in ../LLM-From-Scratch/documentation/training/SESSION_DECISIONS.md. Historical assets are external inputs; never commit weights, datasets or recovery snapshots.
