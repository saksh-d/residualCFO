2026-05-25 17:59:30 EDT

This update was limited to repository state management before moving from the Stage 1.2 line of work to Stage 2.2. The main concern was preventing local orchestration artifacts and generated result folders from being misinterpreted as experiment deltas while preserving the exact tracked Stage 1.2 code state.

The checked Stage 1.2 tip was commit `e2b9052`, which corresponds to the PAPR sidecar tooling addition. The local branch was synchronized with `origin/stage1.2`, so there was no unpublished model or experiment logic at risk in the tracked history. The only pending workspace items were local-control artifacts (`AGENTS.md`, `.codex/`) and Stage 2 result directories (`stage2_usrnet_outputs/`, `stage2_grrnet_outputs/`).

From a communication-systems workflow perspective, these paths are not part of the Stage 1 reproducible artifact set. They represent either operator metadata or generated outputs from later-stage receiver experiments. Folding them into ignore rules keeps the Stage 1 branch semantically clean: tracked history continues to represent waveform/receiver logic and canonical reporting code, while transient outputs remain excluded from version control.

An additional sanity check showed that Stage 2.2 already ignores these same local-output paths, which reduces branch-switch risk and keeps branch behavior consistent. The Stage 1.2 ignore update therefore mainly serves as local branch hygiene and as an explicit record that these files should not be treated as pending experimental code changes.
