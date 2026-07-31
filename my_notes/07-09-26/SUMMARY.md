

## What the fine-tuning did, exactly

**The setup.** We're using pretrained GemmaScope transcoders — sparse modules that approximate each MLP layer of `gemma-2-2b` with an interpretable feature basis. They were trained to imitate the **base model's MLP** on the **base model's** hidden states. But our use case feeds them the **instruction-tuned model's** hidden states (to circuit-trace the base↔instruct gap).

**The problem it solved.** Experiment 1 showed those transcoders reconstruct worse on instruct-distribution inputs than on base-distribution inputs — the reconstruction error (FVU) was higher at every layer, worst at the endpoints (layer 0 +33%, layer 24 +32%, layer 25 **+104%**). The features fire, but the transcoder's approximation of the MLP output degrades because the input distribution shifted. In circuit-tracer terms, that lost fidelity shows up as larger "error nodes" — the part of the computation the graph *can't* attribute to interpretable features.

**What the fine-tune actually changed.** For the three worst layers (0, 24, 25), we took the pretrained transcoders and did a short (~14 min, 2M instruct tokens) supervised fine-tune minimizing:

  `MSE( transcoder(x) , MLP_base(x) )`, where `x` = the instruct model's hidden states.

I.e. we nudged each transcoder to keep imitating the **base MLP**, but now accurately on the **instruct input distribution**. The JumpReLU threshold was frozen, and — this is the key detail from the encoder-drift check — the **encoder directions barely moved** (cosine ≈ 0.9997, zero features below 0.99). So it adjusted the **decoder and biases** (what each feature writes / when it crosses threshold), **not what each feature detects**. The feature identities are essentially unchanged.

**The result.** On held-out instruct tokens, FVU dropped back toward the base level: layer 0 `0.108 → 0.075` (−31%, fully recovered), layer 24 `0.281 → 0.221` (−22%, recovered), layer 25 `0.317 → 0.212` (−33%, most of the way). And in the actual attribution graphs, the error-node fraction fell ~5% on every prompt tested — i.e. slightly more of the circuit is carried by interpretable features and less by unexplained error.

**One-sentence version.** The fine-tune re-adapted three GemmaScope transcoders so they still reconstruct the base MLP sparsely, but now on instruction-tuned hidden states instead of base ones — closing most of the reconstruction gap the distribution shift had opened, while leaving the features themselves (what they detect) intact.
