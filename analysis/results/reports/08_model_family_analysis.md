# MambaVision Model Family Analysis

This report is generated from measured JSON outputs. Explanations are phrased as interpretations of the probes, not as proof of training-time causality.

## Model Summary

| Model | Params | Channels | SSM | Attention | Throughput |
|---|---:|---|---:|---:|---:|
| tiny | 31,794,248 | [80, 160, 320, 640] | 6 | 6 | 602.80 img/s |
| tiny2 | 35,104,008 | [80, 160, 320, 640] | 8 | 7 | 342.22 img/s |
| small | 50,140,584 | [96, 192, 384, 768] | 7 | 5 | 673.05 img/s |
| base | 97,685,288 | [128, 256, 512, 1024] | 8 | 7 | 164.76 img/s |
| large | 227,961,048 | [196, 392, 784, 1568] | 8 | 7 | 76.82 img/s |

## What The Results Mean

### tiny

- **Capacity and geometry.** The measured stage channels are `[80, 160, 320, 640]` at resolutions `[[56, 56], [28, 28], [14, 14], [7, 7]]`. Parameter count is `31,794,248`. Wider channels increase representational capacity but also increase compute, which is reflected in throughput.
- **Effective receptive field.** The measured r90 radii by stage are `[10.1, 29.8, 98.5, 100.4]` pixels. The increase from early to late stages is expected because downsampling and token mixing let a center feature aggregate information from a larger input region. This is a gradient-based reach measurement, not a semantic importance proof.
- **Frequency response.** Dominant raw activation stages by frequency are `[4, 4, 4, 4, 4, 4, 4]` for frequencies `[0.009999999776482582, 0.05000000074505806, 0.10000000149011612, 0.20000000298023224, 0.30000001192092896, 0.4000000059604645, 0.5]`. Raw activation magnitude is affected by normalization, channel scale, and depth, so this is best read as a stage-energy probe rather than a calibrated transfer function.
- **Edge selectivity.** The top layer by ESI is `model.levels.0.downsample.reduction.0` with ESI `0.0773`. ESI compares edge stimuli against a flat image, so positive values indicate stronger response to controlled edges than to uniform input.
- **SSM stability.** `6/6` SSM blocks have sampled discrete eigenvalues inside the unit circle. Mean |eigenvalue| is `0.5919`, and mean impulse half-life is `2.00` samples. This indicates stable decay in the diagonal SSM approximation used by the script.
- **Selective scan verification.** Two different inputs produced mean absolute differences of Δ `0.1493`, B `0.5477`, and C `0.5679` in the first mixer. That numerically supports the claim that these scan parameters are input-dependent for the inspected block.
- **Branch complementarity.** First-mixer branch correlation is `-0.0166`. A value near zero means the SSM and symmetric branches are not linearly redundant under this synthetic input, though complementarity should be validated on task data before making stronger claims.

### tiny2

- **Capacity and geometry.** The measured stage channels are `[80, 160, 320, 640]` at resolutions `[[56, 56], [28, 28], [14, 14], [7, 7]]`. Parameter count is `35,104,008`. Wider channels increase representational capacity but also increase compute, which is reflected in throughput.
- **Effective receptive field.** The measured r90 radii by stage are `[10.6, 28.0, 103.5, 104.3]` pixels. The increase from early to late stages is expected because downsampling and token mixing let a center feature aggregate information from a larger input region. This is a gradient-based reach measurement, not a semantic importance proof.
- **Frequency response.** Dominant raw activation stages by frequency are `[4, 4, 4, 4, 4, 4, 4]` for frequencies `[0.009999999776482582, 0.05000000074505806, 0.10000000149011612, 0.20000000298023224, 0.30000001192092896, 0.4000000059604645, 0.5]`. Raw activation magnitude is affected by normalization, channel scale, and depth, so this is best read as a stage-energy probe rather than a calibrated transfer function.
- **Edge selectivity.** The top layer by ESI is `model.levels.0.downsample.reduction.0` with ESI `0.0789`. ESI compares edge stimuli against a flat image, so positive values indicate stronger response to controlled edges than to uniform input.
- **SSM stability.** `8/8` SSM blocks have sampled discrete eigenvalues inside the unit circle. Mean |eigenvalue| is `0.5851`, and mean impulse half-life is `2.00` samples. This indicates stable decay in the diagonal SSM approximation used by the script.
- **Selective scan verification.** Two different inputs produced mean absolute differences of Δ `0.1491`, B `0.5235`, and C `0.4776` in the first mixer. That numerically supports the claim that these scan parameters are input-dependent for the inspected block.
- **Branch complementarity.** First-mixer branch correlation is `0.0228`. A value near zero means the SSM and symmetric branches are not linearly redundant under this synthetic input, though complementarity should be validated on task data before making stronger claims.

### small

- **Capacity and geometry.** The measured stage channels are `[96, 192, 384, 768]` at resolutions `[[56, 56], [28, 28], [14, 14], [7, 7]]`. Parameter count is `50,140,584`. Wider channels increase representational capacity but also increase compute, which is reflected in throughput.
- **Effective receptive field.** The measured r90 radii by stage are `[14.7, 31.6, 97.1, 110.0]` pixels. The increase from early to late stages is expected because downsampling and token mixing let a center feature aggregate information from a larger input region. This is a gradient-based reach measurement, not a semantic importance proof.
- **Frequency response.** Dominant raw activation stages by frequency are `[4, 4, 4, 4, 4, 4, 4]` for frequencies `[0.009999999776482582, 0.05000000074505806, 0.10000000149011612, 0.20000000298023224, 0.30000001192092896, 0.4000000059604645, 0.5]`. Raw activation magnitude is affected by normalization, channel scale, and depth, so this is best read as a stage-energy probe rather than a calibrated transfer function.
- **Edge selectivity.** The top layer by ESI is `model.levels.0.downsample.reduction.0` with ESI `0.0675`. ESI compares edge stimuli against a flat image, so positive values indicate stronger response to controlled edges than to uniform input.
- **SSM stability.** `7/7` SSM blocks have sampled discrete eigenvalues inside the unit circle. Mean |eigenvalue| is `0.6093`, and mean impulse half-life is `2.00` samples. This indicates stable decay in the diagonal SSM approximation used by the script.
- **Selective scan verification.** Two different inputs produced mean absolute differences of Δ `0.0852`, B `0.7013`, and C `0.7451` in the first mixer. That numerically supports the claim that these scan parameters are input-dependent for the inspected block.
- **Branch complementarity.** First-mixer branch correlation is `-0.0067`. A value near zero means the SSM and symmetric branches are not linearly redundant under this synthetic input, though complementarity should be validated on task data before making stronger claims.

### base

- **Capacity and geometry.** The measured stage channels are `[128, 256, 512, 1024]` at resolutions `[[56, 56], [28, 28], [14, 14], [7, 7]]`. Parameter count is `97,685,288`. Wider channels increase representational capacity but also increase compute, which is reflected in throughput.
- **Effective receptive field.** The measured r90 radii by stage are `[14.6, 29.3, 98.2, 123.3]` pixels. The increase from early to late stages is expected because downsampling and token mixing let a center feature aggregate information from a larger input region. This is a gradient-based reach measurement, not a semantic importance proof.
- **Frequency response.** Dominant raw activation stages by frequency are `[4, 4, 4, 4, 4, 4, 4]` for frequencies `[0.009999999776482582, 0.05000000074505806, 0.10000000149011612, 0.20000000298023224, 0.30000001192092896, 0.4000000059604645, 0.5]`. Raw activation magnitude is affected by normalization, channel scale, and depth, so this is best read as a stage-energy probe rather than a calibrated transfer function.
- **Edge selectivity.** The top layer by ESI is `model.levels.0.downsample.reduction.0` with ESI `0.0629`. ESI compares edge stimuli against a flat image, so positive values indicate stronger response to controlled edges than to uniform input.
- **SSM stability.** `8/8` SSM blocks have sampled discrete eigenvalues inside the unit circle. Mean |eigenvalue| is `0.6583`, and mean impulse half-life is `2.38` samples. This indicates stable decay in the diagonal SSM approximation used by the script.
- **Selective scan verification.** Two different inputs produced mean absolute differences of Δ `0.0871`, B `0.1638`, and C `0.3130` in the first mixer. That numerically supports the claim that these scan parameters are input-dependent for the inspected block.
- **Branch complementarity.** First-mixer branch correlation is `-0.0053`. A value near zero means the SSM and symmetric branches are not linearly redundant under this synthetic input, though complementarity should be validated on task data before making stronger claims.

### large

- **Capacity and geometry.** The measured stage channels are `[196, 392, 784, 1568]` at resolutions `[[56, 56], [28, 28], [14, 14], [7, 7]]`. Parameter count is `227,961,048`. Wider channels increase representational capacity but also increase compute, which is reflected in throughput.
- **Effective receptive field.** The measured r90 radii by stage are `[15.0, 30.2, 105.5, 133.8]` pixels. The increase from early to late stages is expected because downsampling and token mixing let a center feature aggregate information from a larger input region. This is a gradient-based reach measurement, not a semantic importance proof.
- **Frequency response.** Dominant raw activation stages by frequency are `[4, 4, 4, 4, 4, 4, 4]` for frequencies `[0.009999999776482582, 0.05000000074505806, 0.10000000149011612, 0.20000000298023224, 0.30000001192092896, 0.4000000059604645, 0.5]`. Raw activation magnitude is affected by normalization, channel scale, and depth, so this is best read as a stage-energy probe rather than a calibrated transfer function.
- **Edge selectivity.** The top layer by ESI is `model.levels.0.downsample.reduction.0` with ESI `0.0909`. ESI compares edge stimuli against a flat image, so positive values indicate stronger response to controlled edges than to uniform input.
- **SSM stability.** `8/8` SSM blocks have sampled discrete eigenvalues inside the unit circle. Mean |eigenvalue| is `0.6634`, and mean impulse half-life is `2.62` samples. This indicates stable decay in the diagonal SSM approximation used by the script.
- **Selective scan verification.** Two different inputs produced mean absolute differences of Δ `0.1059`, B `0.3511`, and C `0.4446` in the first mixer. That numerically supports the claim that these scan parameters are input-dependent for the inspected block.
- **Branch complementarity.** First-mixer branch correlation is `-0.0069`. A value near zero means the SSM and symmetric branches are not linearly redundant under this synthetic input, though complementarity should be validated on task data before making stronger claims.

## Generated Comparison Figures

- `analysis/results/figures/family_erf_r90.png`
- `analysis/results/figures/family_throughput.png`
- `analysis/results/figures/family_edge_top_esi.png`
- `analysis/results/figures/family_frequency_dominance.png`
- `analysis/results/figures/family_ssm_summary.png`
