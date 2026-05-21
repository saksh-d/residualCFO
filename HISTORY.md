# HISTORY

- 2026-05-20 00:00 EDT
  - Initialized project history log for the Neural-USR refactor workstream.
  - Target change: move Stage 2 USR-Net from anchor-replacement updates to anchor-initialized neural unfolded updates.
  - Scope includes receiver forward path, phased training/freeze policy, guard-vs-core ablation, and report wording.

- 2026-05-20 00:25 EDT
  - Replaced the Neural-USR unfolded update with anchor-initialized residual neural refinement: `s0` from shared diagonal anchor, then three learned `eta_t` residual steps.
  - Split Stage 2 trainable parameters into shared anchor, per-layer diagonal reference, and neural Conv1D/`eta_t` groups to match the requested Phase 0/1/2 freeze schedule.
  - Reworked guard validation to compare the final neural output against the separately trained `r_core` baseline, and updated public reporting language to Neural-USR / Neural USR-Net.

- 2026-05-20 14:21 EDT
  - Smoke-tested the full `run_stage2_usrnet.py --smoke` workflow after the Neural-USR refactor; imports, training phases, evaluation, CSV outputs, and report generation completed successfully.
  - Verified required ablations and per-layer artifacts: `usrnet_summary.csv`, `core_reference_summary.csv`, `per_layer_evm.csv`, and `per_layer_ber.csv`.
  - Learned + Neural-USR passed the requested checks at CFO `0.10` and zero CFO, and improved over `r_core` in the smoke run (`3.61e-4` vs `4.45e-4` BER at `0.10`).
  - Fixed report appendix duplication by appending supplemental sections only to the exact report path returned by the report writer.

- 2026-05-20 19:17 EDT
  - Added a fully separate publication-only USR-Net figure script under the ignored Stage 2 results tree, using only saved CSV artifacts and not the main reporting pipeline.
  - Produced publication exports for BER-vs-CFO, stacked BER-vs-SNR, 4x2 constellation panels, and a transmit-only PAPR CCDF in both PNG and PDF formats.
  - Standardized publication labels to `OFDM`, `OFDM + USR-Net`, `Learned Basis`, and `Learned + USR-Net`, removed titles, and tuned sizing/layout for IEEE two-column readability.
  - Validated the script end-to-end and refined the SNR legend placement plus constellation row-label layout after visual inspection of the generated figures.

- 2026-05-20 19:24 EDT
  - Reverted the publication-only USR-Net figure styling toward standard Matplotlib defaults after the first pass proved visually overworked and poorly balanced.
  - Simplified typography, colors, legend placement, axis presentation, and panel labeling while keeping the separate script/results-tree workflow intact.
  - Regenerated all four publication figures in PNG/PDF and visually re-checked the BER, SNR, constellation, and PAPR layouts for basic paper readability.

- 2026-05-20 19:25 EDT
  - Reduced the publication script further to a near-stock Matplotlib presentation so figure labeling follows normal subplot/title placement rather than custom annotations.
  - Kept one fixed method-color mapping across all figures and converted the constellation summary to a vertical `4x2` layout for cleaner comparison.
  - Re-ran the standalone publication script and visually confirmed the regenerated BER, SNR, constellation, and PAPR figures from saved CSV artifacts only.

- 2026-05-20 19:32 EDT
  - Tightened the stock publication figures with figure-specific legend placement, slightly smaller typography, and a fixed `1e-4` BER floor for the stacked SNR plot.
  - Reworked the constellation rendering to emphasize symbol clouds and ideal locations more clearly, using `I/Q` axis notation and the vertical `4x2` organization.
  - Reduced the PAPR panel footprint and moved its legend away from the dominant tail region, then visually re-checked all four exported figures.

- 2026-05-20 19:36 EDT
  - Unified legend sizing at `10`, renamed the BER y-axis in the CFO sweep plot, and stacked the SNR legend entries vertically in the lower subplot.
  - Adjusted the constellation view again so ideal symbols appear as dark circular anchors with the estimated clouds rendered beneath them at lower density.
  - Resized the PAPR panel to match the Fig. 2 footprint and re-exported the full publication figure set after direct visual review.

- 2026-05-20 19:45 EDT
  - Restored the Fig. 2 y-axis label to `BER` and kept the publication figure typography/layout otherwise stable.
  - Added four separate Fig. 4 constellation rendering variants so anchor style, cloud density, and density visualization could be compared directly without altering the underlying data/layout.
  - Re-exported the publication set and reviewed the four constellation alternatives side by side to identify a more credible paper-ready rendering direction.

- 2026-05-20 19:45 EDT
  - Finalized Fig. 4 back to a single rendering style using black hollow 16QAM reference rings with method-colored symbol clouds and removed the extra variant exports.
  - Tightened the constellation axis window to trim excess empty space and make the cloud structure occupy more of each panel.
  - Re-exported the cleaned publication figure set and visually verified the updated BER label plus the simplified single-design constellation panel.

- 2026-05-20 19:45 EDT
  - Reduced the visual weight of the black reference rings and increased cloud opacity so the estimated symbols are now visually dominant in Fig. 4.
  - Re-exported the publication figure set and confirmed the constellation hierarchy now favors the colored symbol clouds over the nominal reference markers.

- 2026-05-20 19:45 EDT
  - Nudged the Fig. 4 crop slightly wider, separated the `Q` axis label from the waveform row labels, and increased symbol-cloud opacity again for clearer cluster visibility.
  - Re-exported and visually checked the updated single-design constellation panel after the final label/opacity adjustment.
