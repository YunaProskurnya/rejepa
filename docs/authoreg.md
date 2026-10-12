# Gaussian noise and I-JEPA prediction

## Question and result so far

Can independent Gaussian noise on context representations, followed by an MSE prediction objective, prevent collapse in joint I-JEPA training without SIGReg or VISReg?

The answer is **no as a general guarantee**. A constant encoder and constant predictor achieve zero MSE despite context noise. Noise may still alter optimization enough to improve representations empirically. That narrower claim needs training runs; none have been completed for this branch.

The [I-JEPA paper](https://arxiv.org/abs/2301.08243) defines masked context to target prediction. The [noise-injected Deep InfoMax paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/c8ff6807d1f362bb22b4f0be7b66e9ca-Paper-Conference.pdf) adds independent noise to normalized encoder outputs **while maximizing a mutual information objective**. Its Gaussian theorem assumes zero mean, unit population variance in every feature, independent isotropic Gaussian noise, and conditions on paired views; equality also requires weak invariance and a Gaussian representation distribution. [AWGN capacity](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/24fb44b6a101e847c32ffe38bbeeea6e_MIT6_441S16_chapter_17.pdf) maximizes mutual information between a channel input and its noisy output under an average power constraint. Neither result says that minimizing a learned target's prediction error maximizes that information.

## Objective and counterexample

Let `C = f_theta(image, context_mask)`, `T = select(LN(f_theta(image)), target_mask)`, and let `N ~ Normal(0, sigma^2 I)` be independent of the image, mask, and parameters. The experiment minimizes

`E[||g_phi(C + N, target_positions) - T||^2]`.

If the encoder outputs the same vector for every image and patch, `T` is also constant. The predictor can ignore `C + N` and output that constant, giving zero MSE. This remains true for any finite noise level and with either joint or EMA targets. In the current ViT, learned normalization offsets and a predictor output bias can represent this solution. Adding independent noise to the target instead would add an irreducible noise floor but would still leave a constant clean representation optimal.

For a direct Gaussian example, take `S ~ Normal(0,1)`, `C = T = aS`, and `N ~ Normal(0,sigma^2)`. The best linear predictor of `T` from `C + N` has gain `a^2/(a^2+sigma^2)` and MSE `a^2 sigma^2/(a^2+sigma^2)`. The minimum over `a` is attained at `a=0`, while `I(C; C+N) = 0.5 log(1+a^2/sigma^2)` increases with `|a|`. At `sigma=0.5`, the MSE is `0` for `a=0`, `0.125` for `a=0.5`, and `0.2` for `a=1`. This is an exact toy model, not an I-JEPA training result.

Extending the AWGN result would require at least an objective that actually maximizes `I(C;C+N)`, independent fixed variance channel noise, and a specified representation power budget. An upper power bound alone permits zero power under MSE minimization. A positive power equality or variance lower bound rules out collapse by assumption and is itself an explicit constraint. For image prediction, one must also establish that high information in `C` is information useful for predicting `T`; the channel theorem does not supply this. Tokenwise LayerNorm fixes within-token scale approximately, not variation across images, and does not prevent constant token vectors. A finite deterministic image dataset also cannot produce an exact continuous Gaussian representation distribution.

MSE can lower bound predictive mutual information through `I(T;Y) >= h(T) - (d/2) log(2 pi e MSE/d)` for a suitable continuous target `T` and estimate from `Y`. In ordinary reconstruction, the target entropy `h(T)` is fixed. In joint I-JEPA it is learned and can shrink along with MSE; at exact collapse the continuous-entropy expression is inapplicable. The InfoMax theorem also treats two encoded views of a shared source, whereas masked I-JEPA uses asymmetric context and target representations. Establishing an equivalence would require proving both the needed view relation and a fixed nonzero target entropy or scale.

## Training experiment

[imagenet100_vit_base_400ep_sk_container_authoreg.json](../configs/imagenet100_vit_base_400ep_sk_container_authoreg.json) matches the existing Skoltech container configs: full ImageNet100, 400 epochs, ViT Base, and output under `/home/container/shared/rejepa/runs/`. It uses joint encoder training, MSE, no regularizer, and context noise `sigma=0.1`. Noise is sampled independently for each context token after the context encoder and before the predictor. Targets stay clean. Evaluation uses clean encoder features. The original smooth L1 and zero noise settings remain the defaults for existing runs.

The launcher requires `COMET_API_KEY`; it saves `config.json`, `metrics.csv`, checkpoints, and representation spectra in the output directory. Check GPU availability, run training in tmux, and save terminal output:

```sh
cd /home/container/shared/rejepa
mkdir -p logs
nvidia-smi
tmux new -s authoreg
python /home/container/rejepa/scripts/train.py \
  --config /home/container/rejepa/configs/imagenet100_vit_base_400ep_sk_container_authoreg.json \
  2>&1 | tee logs/authoreg-400ep.log
```

The working directory must contain `train/` and `val/`. For matched full length ablations, override `--noise-std` with `0` or `0.5` and give each run a distinct `--output` and log name. Use `--regularizer sigreg --noise-std 0` for a positive control. Repeat with matched seeds if resources permit. Do not compare raw loss across SIGReg and unregularized runs as a collapse measure.

Compare clean feature mean variance and effective rank from `representation_analysis/representations.csv` at matched epochs, then assess downstream utility with the repository's kNN or linear probe evaluation on held-out data. Loss alone can fall while the representation collapses. Diagnostics are recorded at the first, every 25th, and final completed epoch. [VISReg](https://arxiv.org/abs/2606.02572) and [VICReg](https://arxiv.org/abs/2105.04906) motivate the positive regularization comparison.

## Current limits

No ImageNet100 training was run. The default Python lacks PyTorch; an alternate local environment has PyTorch but lacks torchvision and Comet. This workspace also has no prepared dataset, usable GPU, or tmux, and the configured remote server is unavailable. The analytical counterexample settles the universal anti-collapse claim, while the empirical effect of noise on this implementation remains open.
