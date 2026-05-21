# TECHNICAL

## 2026-05-20 00:00 EDT

Planning/implementation kickoff for the Neural-USR refactor. The current Stage 2 USR-Net uses a diagonal model-guided reference inside each unfolded layer through a convex replacement-style update, with per-layer `rho_t`, `gamma_t`, and `alpha_t`. The requested change is to preserve the model-guided diagonal reference only as initialization and conditioning while forcing the final detector output to be produced by three learned residual neural updates.

The retained communication-system interpretation is:
- `z0 = V y` remains the Stage 1 post-`V` symbol-domain observation.
- `r_core` remains the diagonal model-guided baseline and must stay available for guard loss and explicit ablation.
- The practical receiver should no longer output the anchor directly; instead it should start from an anchor-derived initialization and then refine through learned residual updates conditioned on the same diagonal information.

Implementation intent:
- keep `W_tx` and `V` frozen,
- keep the same residual dilated Conv1D block and three unfolded layers,
- replace `rho_t`/`alpha_t` with bounded `eta_t`,
- preserve zero-CFO identity protection,
- and judge success relative to the explicit `r_core` baseline rather than the old hybrid anchor-plus-residual update.

## 2026-05-20 00:25 EDT

The receiver refactor now separates three roles that were previously entangled in the old USR-Net update:

1. External baseline:
- `DiagonalCoreReference` is retained unchanged as the standalone trained `r_core` model-guided receiver.
- This object remains the guard-loss comparator and the explicit ablation reported alongside the neural receiver.

2. Internal initialization anchor:
- `UnfoldedSymbolRefiner` now builds a shared diagonal initialization anchor from `z0` and the diagonal of `A_hat(c_b)`.
- The retained final-gain parameters are applied only inside this initialization path so that the practical detector output is not post-multiplied after the neural iterations.

3. Neural unfolded refinement:
- The unfolded state now starts from `s0 = r_core_internal`.
- Each layer still uses the same 10-channel conditioning tensor based on `(s_t, z0, r_t, s_t-r_t, d_hat)`.
- The Conv1D residual block remains unchanged structurally, but the update law is now additive: `s_{t+1} = s_t + eta_t * delta_t`.
- `eta_t` is bounded to `[0.05, 0.20]`, implemented as `0.05 + 0.15 * sigmoid(raw_eta_t)`.

Training semantics were also changed:
- Phase 0 tunes only the shared anchor gamma/final-gain parameters while neural updates are disabled.
- Phase 1 freezes anchor and layer-reference parameters, then trains only Conv1D and `eta_t`.
- Phase 2 unfreezes all Stage 2 parameters for low-rate joint fine-tuning.

Validation semantics were corrected so the guard term compares the final Neural-USR output against the separately trained external `r_core`, not against the neural model’s own internal shared anchor.

## 2026-05-20 14:21 EDT

Execution validation was completed with the project smoke configuration by running the Stage 2 workflow end-to-end. This exercised:
- standalone diagonal-core reference training for OFDM and Learned bases,
- Phase 0 anchor-matching with neural updates disabled,
- Phase 1 neural residual training with anchor/reference parameters frozen,
- Phase 2 joint fine-tuning,
- full BER-vs-CFO evaluation,
- explicit `r_core` ablation export,
- per-layer neural iterate diagnostics,
- and final report generation.

Observed smoke behavior:
- Learned + Neural-USR remained very close to the trained diagonal-core baseline near zero CFO and did not introduce the zero-CFO regression that the guard and identity mechanisms are meant to prevent.
- At CFO `0.10`, Learned + Neural-USR achieved BER `3.6079e-4` while the explicit `r_core` ablation was `4.4488e-4`, so the smoke run already falls into the requested “neural refinement gain” case.
- Zero-CFO BER degradation constraints were satisfied:
  - OFDM gap: `+1.3563e-05`
  - Learned gap: `-1.0851e-05`
- The per-layer outputs were exported as three-step unfolded trajectories, confirming that the workflow now treats the neural output sequence as `s1, s2, s3` rather than as anchor-replacement states.

Implementation correction from validation:
- The report appendix helper was still appending to both `report.md` and `report.MD`, which duplicated the Neural-USR appendix sections when both files existed in the output directory.
- This was fixed by targeting the exact report path returned by `write_markdown_report`, so supplemental Neural-USR acceptance and `r_core` ablation sections are appended only once to the canonical report artifact.

## 2026-05-20 19:17 EDT

A publication-only figure path was added for the USR-Net results without touching the main experiment or report-generation code. The new script lives under the ignored results tree and consumes only the already generated CSV artifacts from the existing Stage 2 run directory. This keeps publication formatting experiments isolated from the reproducibility-critical pipeline.

The script uses:
- `ber_vs_cfo.csv` for the absolute residual-CFO BER curve,
- `ber_vs_snr_by_cfo.csv` for the stacked BER-vs-SNR panels at `delta = 0.05` and `delta = 0.10`,
- `constellation_snapshots.csv` for the `2 x 4` constellation grid,
- and `papr_samples.csv` for the transmit-only PAPR CCDF.

Publication formatting choices implemented:
- titles removed everywhere,
- axis labels set to `14 pt`,
- legends set to `12 pt`,
- restrained serif typography and Type-42 vector text export for paper-friendly PDF output,
- moderate line width and light grids for readability without visual clutter,
- consistent method styling across figures,
- and both PNG and PDF export from the same script.

Figure-specific behavior:
- Fig. 2 uses the absolute residual CFO view, matching the Stage 2 USR-Net convention rather than the signed-CFO diagnostic.
- Fig. 3 uses a vertical `2 x 1` layout with CFO annotations inside the axes instead of subplot titles, which preserves the no-title requirement while keeping each panel self-contained.
- Fig. 4 uses a `2 x 4` layout with method labels as column headers and CFO annotations embedded in the first-column panels. The estimated constellations are plotted with deterministic subsampling to keep density legible in print while preserving the qualitative structure. Ideal reference points are shown as subtle hollow markers beneath the estimated clouds.
- Fig. 5 includes only `OFDM` and `Learned Basis`, consistent with the fact that the transmit waveform statistics are unaffected by the USR-Net post-receiver refinement.

Validation outcome:
- the standalone script executed successfully against the saved USR-Net run directory,
- generated `fig2_ber_vs_residual_cfo`, `fig3_ber_vs_snr_stacked`, `fig4_constellations_grid`, and `fig5_papr_ccdf` in both `.png` and `.pdf`,
- and visual inspection identified and corrected two presentation issues: the stacked-SNR legend placement and the constellation row-label collision with the shared `Quadrature` axis label.

## 2026-05-20 19:24 EDT
- Publication-only USR-Net figure generation was revised after the initial styling pass introduced excessive layout manipulation and weak visual balance relative to standard research-figure expectations.
- The revised objective was to preserve the separate post-processing workflow while reverting presentation choices toward conventional Matplotlib output: simpler sans-serif typography, restrained axes/spines, standard color assignments, and less aggressive legend/layout customization.
- Fig. 2 and Fig. 3 were adjusted away from oversized external or floating layout treatments and back toward normal in-axes composition, with the intent of restoring immediate readability in a two-column paper context.
- Fig. 4 retained the requested 2x4 constellation structure but replaced the earlier more ornamental global labeling strategy with standard per-panel axis labeling and lighter row annotation for CFO.
- Fig. 5 remained transmit-only and visually stable; the main concern in this iteration was overall consistency and removal of the earlier over-designed styling choices.
- The regenerated figure set was exported again in both PNG and PDF from the standalone publication script using only the saved USR-Net CSV artifacts; no experiment rerun or main reporting-pipeline change was required.

## 2026-05-20 19:25 EDT
- The publication-only figure workflow was pushed fully back toward stock Matplotlib behavior after the previous revision still carried too much manual layout intervention for labels and panel structure.
- The line plots now rely on standard axis titles, standard legends, default spine behavior, and only the requested font-size constraints plus fixed per-method color/style assignments for consistency across figures.
- The constellation summary was restructured from a horizontal 2x4 arrangement into a vertical 4x2 grid so method identity can be communicated through normal row labeling while CFO severity is handled by standard column titles.
- This change removes the earlier custom column-header and row-annotation scheme, which was the main source of awkward placement and nonstandard visual balance.
- The regenerated outputs were reviewed directly after export: the BER-vs-CFO and PAPR plots now read as conventional single-figure Matplotlib outputs, and the constellation summary is materially clearer in the vertical arrangement.
- The figure generation remained isolated to the ignored results tree and reused only the previously generated USR-Net CSV artifacts, with no experiment rerun and no modifications to the main reporting pipeline.

## 2026-05-20 19:32 EDT
- The publication-only figure refinement focused on practical readability corrections rather than further stylistic experimentation: legend collisions, axis truncation, overly large typography, and weak constellation interpretation were treated as the primary defects.
- Fig. 2 was adjusted by pinning the legend to the upper-left region and reducing legend size slightly so the curve family remains visible across the full CFO range.
- Fig. 3 was made more publication-consistent by forcing a common BER floor of 1e-4 across both SNR subplots and relocating the legend to the lower-left corner of the bottom panel, avoiding duplication while preserving the requested stacked structure.
- Fig. 4 was changed to behave more like a true detector-cloud visualization: fewer plotted points, lower scatter opacity, explicit ideal-symbol markers, no grid clutter, and compact I/Q axis notation. The row-wise method labeling on the left remains the cleanest way to encode waveform/receiver configuration in the vertical 4x2 arrangement without introducing another custom label layer.
- Fig. 5 was resized downward and its legend was moved into a quieter lower-left region so the right-hand CCDF tail remains unobstructed, which is the part of the curve most likely to matter in the paper discussion.
- The full figure set was regenerated from the saved USR-Net CSV artifacts only and then visually reviewed again to verify that the new placements and rendering choices actually improved legibility rather than just changing appearance.

## 2026-05-20 19:36 EDT
- The latest publication-only revision standardized legend sizing at 10 pt across the figure set and cleaned up the line-plot annotations so the CFO-sweep panel now uses the fuller Bit Error Rate naming while the stacked SNR figure keeps a single vertically ordered legend block in the lower subplot.
- The constellation rendering was adjusted away from cross-style reference marks and toward dark circular ideal-symbol anchors, with the cloud samples drawn first and at lower point density so the nominal constellation centers remain visually obvious.
- For the constellation panel specifically, the remaining tradeoff is between density information and crisp cluster boundaries: plotting fewer samples with slightly higher opacity improves cluster readability, while plotting more points better conveys dispersion but muddies the cloud edges. This pass biased toward readability.
- The PAPR figure was resized to the same overall footprint as the BER-vs-CFO figure so it no longer looks visually undersized relative to the rest of the publication set.
- All figures were regenerated again from the saved Stage 2 USR-Net CSV artifacts only, preserving the separate post-processing workflow and avoiding any change to the main experiment/report scripts.

## 2026-05-20 19:45 EDT
- The publication figure workflow was extended to support multiple constellation-rendering alternatives because the central problem was no longer data availability or layout but the visual encoding of cloud dispersion versus ideal decision points.
- Four separate Fig. 4 variants were generated from the same sampled data and identical 4x2 structure: a soft-dot anchor view, a hollow-anchor view, a dual-layer scatter view, and a hexbin density view. This isolates stylistic choices cleanly enough for side-by-side judgment.
- The soft-dot version uses restrained filled anchor dots and light cloud opacity; the hollow-anchor version makes the reference points less visually heavy; the dual-layer version tries to recover cloud shape by mixing a faint full scatter with a darker subsample; the density-hex version emphasizes occupancy structure over individual point appearance.
- From a communication-figure standpoint, the main tradeoff is between preserving the intuitive “cloud around a symbol” appearance and making the dispersion legible at small subplot size. Filled scatter preserves intuition best, while density bins make high-level spread easier to read but look less like conventional constellation panels.
- The BER-vs-CFO figure was also corrected to use the shorter y-axis label BER, as requested, and the full publication set was re-exported after the variant generation pass.

## 2026-05-20 19:45 EDT
- The constellation panel was simplified again after the variant comparison: the chosen direction is a conventional scatter-only rendering with method-colored estimated symbols and black hollow rings for the nominal 16QAM reference points.
- The key visual adjustment beyond marker style was axis cropping. Instead of using the raw extremal points, the panel bounds now come from a high-percentile envelope so a small number of outliers do not force large margins and visually shrink the useful cloud structure.
- This produces a more publication-typical constellation look: visible symbol centers, colored dispersion around them, and less dead white space around the outer clusters.
- The extra experimental Fig. 4 option files were removed from the publication output folder so the directory now reflects only the current intended figure set.
- Fig. 2 was also confirmed with the short y-axis label BER after the rerender pass.

## 2026-05-20 19:45 EDT
- The final constellation refinement changed only visual hierarchy: the nominal 16QAM rings were made more transparent while the estimated symbol clouds were made more opaque, so the detected-symbol distribution now carries the panel and the reference constellation acts as a secondary guide.
- No layout or data-selection changes were introduced in this pass; the intent was only to rebalance foreground/background emphasis in the existing Fig. 4 design.
- The updated export was visually checked and now reads more naturally as a communication-system constellation plot, especially for the learned-receiver rows where the cloud tightness is the main result being communicated.

## 2026-05-20 19:45 EDT
- The last Fig. 4 adjustment was intentionally small: a slight expansion of the crop window to restore a bit of breathing room, stronger cloud opacity to improve cluster salience, and a left-label spacing correction so the waveform row labels no longer visually interfere with the Q-axis label.
- This pass preserved the previously chosen single-design approach of method-colored scatter clouds over black hollow 16QAM reference rings, with only the emphasis and spacing rebalanced.
- The regenerated panel was checked visually and now reads more cleanly: clouds are easier to see, the anchors remain secondary, and the left-side labeling is no longer cramped.
