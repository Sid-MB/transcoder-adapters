# Annotator Ideas

This file collects candidate annotators for `analysis/features/annotate/`.
The goal is to create dashboard filters that make feature inspection faster and
more informative, not just to add broad labels.

Annotators should generally subclass `FeatureAnnotator`, write their results
under their own `annotation_name`, and preserve update-in-place semantics: a
rerun should replace only that annotator's prior tags and scores.

## Best Ideas

These are the strongest annotator ideas from the discussion. Some can run over
existing metadata or `features/*.json`; others are interesting enough to justify
recollecting data.

1. Contrastive Run Annotator (`contrastive_run`)
   - Compare the same feature across two runs: base vs instruct, pretrain vs
     finetune, transcoder checkpoint A vs checkpoint B, or two datasets.
   - Tags: `it_amplified`, `base_amplified`, `checkpoint_emergent`,
     `checkpoint_suppressed`, `stable_feature`.
   - Needs: matching feature IDs across comparable runs. Recollection is
     worthwhile when the existing runs do not share the same feature space,
     dataset slice, or thresholds.
   - Why it is useful: probably the highest-value recollecting idea. It tells
     you which features are actually different between settings, not just what a
     single run contains.

2. Reasoning-Move Annotator (`reasoning_move`)
   - Collect lightweight lexical/region stats around activations and tag
     features that fire near recognizable reasoning moves.
   - Tags: `self_correction`, `uncertainty`, `planning`, `verification`,
     `answer_commitment`, `backtracking`.
   - Needs: either example-text parsing from `features/*.json` or richer
     collection-time counters.
   - Why it is useful: much more useful than broad `thinking` tags because it
     points at concrete reasoning behaviors.

3. Structural Token Annotator (`structural_token`)
   - Generalize `assistant_token` to all chat/control structure.
   - Tags: `user_token`, `assistant_token`, `think_start_token`,
     `think_end_token`, `bos_token`, `template_feature`.
   - Needs: existing `region_fraction`, `region_density`, and
     `tokens_per_region` metadata.
   - Why it is useful: less flashy than behavior tags, but very useful for
     filtering out boring template/control features before deeper analysis.

4. Logit-Effect Annotator (`logit_effect`)
   - Use `top_logits` and `bottom_logits` to tag what the feature seems to
     promote or suppress.
   - Tags: `promotes_numbers`, `promotes_newline`, `promotes_code`,
     `promotes_uncertainty`, `promotes_refusal`, `promotes_special_tokens`,
     `promotes_answer_tokens`, `suppresses_answer_tokens`.
   - Needs: existing `features/{cantor_id}.json` files with logit lens data.
   - Why it is useful: can run without recollecting if feature JSONs already
     have logit lens data. It pairs nicely with activation-side tags: "fires in
     thinking, promotes final-answer tokens" is interesting.

5. Activation-Shape Annotator (`activation_shape`)
   - Use `tokens_acts_list` around top examples to distinguish spike-like
     features from sustained/context features.
   - Tags: `single_token_spike`, `sustained_context`, `ramp_up`, `ramp_down`,
     `multi_token_phrase`.
   - Needs: existing `features/{cantor_id}.json` files with top examples and
     activation traces.
   - Why it is useful: good for separating token detectors from broader
     state/context features.

6. Feature Specificity Annotator (`feature_specificity`)
   - Score how concentrated a feature is across domains, regions, thinking
     bins, and activating token surfaces.
   - Tags: `high_precision`, `broad_context`, `domain_specialist`,
     `region_specialist`, `token_specialist`.
   - Needs: metadata only for domain/region/thinking-bin specificity, plus
     `features/*.json` if activating token surfaces are included.
   - Why it is useful: a good triage annotator. It helps decide which features
     are worth inspecting manually.

7. Token-Surface Annotator (`token_surface`)
   - Parse top activating examples and tag surface forms of the activated token.
   - Tags: `digit_feature`, `operator_feature`, `punctuation_feature`,
     `newline_feature`, `capitalized_token`, `whitespace_token`, `quote_token`.
   - Needs: existing `features/{cantor_id}.json` files with top examples.
   - Why it is useful: not conceptually deep, but many features are token/string
     artifacts, and this makes them easy to filter.

8. Thinking-Timeline Annotator (`thinking_timeline`)
   - Use thinking-position metadata to tag where features fire inside reasoning
     traces.
   - Tags: `thinking_early`, `thinking_mid`, `thinking_late`,
     `thinking_boundary`, `finalization_feature`.
   - Needs: existing `thinking_position_fraction` and
     `thinking_position_density` metadata.
   - Why it is useful: metadata-only and cheap. It becomes much more
     interesting when combined with logit effects.

9. Cross-Example Consistency Annotator (`cross_example_consistency`)
   - Measure whether top examples share the same activated token, same nearby
     phrase, same region, or same output logits.
   - Tags: `consistent_token`, `consistent_phrase`, `consistent_context`,
     `incoherent_feature`.
   - Needs: existing `features/{cantor_id}.json` files with top examples and,
     for output-logit consistency, `top_logits` / `bottom_logits`.
   - Why it is useful: helps prioritize clean features for interpretation and
     de-prioritize noisy features with no stable story.

10. Auto-Interp Confidence Annotator (`auto_interp_confidence`)
    - Wrap existing LLM-based auto-interp/classification outputs into dashboard
      annotations, but tag confidence and disagreement rather than only storing
      descriptions.
    - Tags: `auto_interp_confident`, `auto_interp_unclear`, `input_simple`,
      `input_abstract`, `output_feature`, `reasoning_feature`,
      `auto_interp_disagreement`.
    - Needs: auto-interp outputs, classifier outputs, or repeated/evaluated
      interpretation calls.
    - Why it is useful: makes automatic interpretations easier to trust, audit,
      and filter by behavioral category.

## High-Value Composite Filters

Some of the best dashboard slices come from combining independent annotators
rather than adding one more broad tag.

- `thinking_late` + `promotes_answer_tokens`: features that may help transition
  from hidden reasoning to final answer.
- `assistant_token` + `promotes_special_tokens`: template/control features that
  are probably useful to filter out before semantic inspection.
- `self_correction` + `it_amplified`: reasoning-move features strengthened by
  instruction tuning.
- `answer_commitment` + `checkpoint_emergent`: features that appear late in
  training and may track answer finalization.
- `domain_specialist` + `consistent_context`: clean dataset-specific features.
- `high_precision` + `auto_interp_confident`: strong candidates for manual
  writeups or feature cards.
- `incoherent_feature` + `broad_context`: likely low-priority features unless
  they have an interesting logit effect.
- `thinking_boundary` + `ramp_up`: features that activate as reasoning starts
  or ends, not just somewhere inside a thinking span.

## Metadata-Only Annotators

### `role_contrast`

- Compare activation density across `question`, `thinking`, `answer`, and
  marker regions.
- Tags: `question_feature`, `thinking_feature`, `answer_feature`,
  `role_boundary_feature`, `role_general_feature`.
- Scores: region fractions, region densities, best-region lift over the rest,
  and best-region activation count.
- Why it is useful: gives a simple first-pass map of where a feature lives in
  the chat transcript.

### `domain_specialist`

- Detect features concentrated in one validation source or domain.
- Tags: `domain_specialist`, plus optional domain-specific tags such as
  `domain_math`, `domain_code`, `domain_chat` when domain names are stable.
- Scores: max domain fraction, max domain density, density lift over other
  domains, and activation count in the top domain.
- Why it is useful: separates general features from dataset-specific artifacts
  and can highlight benchmark or source-specific behavior.

### `rare_dense`

- Find sparse features that are highly concentrated in a domain, role, region,
  or timeline bin.
- Tags: `rare_feature`, `localized_feature`, `high_precision_feature`.
- Scores: activation frequency, best-region lift, best-domain lift, and minimum
  activation-count checks.
- Why it is useful: triages features that may be worth manual inspection even
  if they do not fire often.

### `feature_specificity`

- Summarize how concentrated a feature is across regions, domains, thinking
  bins, and token surfaces when available.
- Tags: `high_precision`, `broad_context`, `region_specialist`,
  `domain_specialist`, `token_specialist`, `diffuse_feature`.
- Scores: entropy or Herfindahl concentration over each distribution.
- Why it is useful: provides a generic quality/triage filter before deeper
  interpretability work.

### `position_shape`

- Use thinking-position distributions to detect whether activations are tightly
  localized or spread out.
- Tags: `position_localized`, `position_broad`,
  `thinking_progression_feature`.
- Scores: peak bin, peak-bin fraction, entropy over thinking bins, and adjacent
  mass around the peak.
- Why it is useful: distinguishes features tied to a specific reasoning phase
  from broad "inside thinking" features.

## Per-Feature JSON Annotators

These require reading `features/{cantor_id}.json` in addition to
`feature_metadata.json`.

### `activation_shape`

- Inspect `tokens_acts_list` around top activating examples.
- Tags: `single_token_spike`, `sustained_context`, `ramp_up`, `ramp_down`,
  `multi_token_phrase`.
- Scores: peak width, activation mass near the highlighted token, left/right
  activation slope, and context activation entropy.
- Why it is useful: separates token detectors from context/state features.

### `token_surface`

- Inspect the highlighted token in top examples.
- Tags: `digit_feature`, `operator_feature`, `punctuation_feature`,
  `newline_feature`, `whitespace_token`, `quote_token`, `capitalized_token`,
  `special_token_surface`.
- Scores: fraction of top examples matching each surface class and diversity of
  highlighted tokens.
- Why it is useful: many features are simple token or formatting detectors, and
  this makes them easy to filter out or study directly.

### `cross_example_consistency`

- Measure whether the top examples share the same activated token, same nearby
  phrase, same region, or same local pattern.
- Tags: `consistent_token`, `consistent_phrase`, `consistent_context`,
  `incoherent_feature`.
- Scores: activated-token concentration, local n-gram concentration, region
  concentration, and top-example agreement.
- Why it is useful: prioritizes clean features for interpretation and flags
  noisy features where manual inspection may be low value.

### `auto_interp_import`

- Import existing auto-interp or classifier outputs into dashboard annotations.
- Tags: `input_simple`, `input_abstract`, `output_feature`,
  `reasoning_feature`, `language_feature`, `unclear_feature`.
- Scores: classifier confidence, number of examples used, and any existing
  evaluator score.
- Why it is useful: makes LLM-generated interpretations visible and filterable
  in the same dashboard as metadata annotations.

### `auto_interp_confidence`

- Focus on the reliability of auto-interp outputs rather than the description
  itself.
- Tags: `auto_interp_confident`, `auto_interp_unclear`,
  `auto_interp_disagreement`, `needs_manual_review`.
- Scores: agreement between classifier calls, detection accuracy on held-out
  examples, description length, and confidence score if available.
- Why it is useful: helps select which automatic descriptions are worth trusting
  or auditing.

## Recollection-Worthy Annotators

These ideas are interesting enough to justify a new collection pass if the
needed counters or paired runs do not already exist.

### `contrastive_run`

- Collect comparable runs over the same feature space, dataset slice, and
  thresholds, then compare each feature's behavior across runs.
- Useful comparisons:
  - base model vs instruction-tuned model
  - early checkpoint vs late checkpoint
  - clean data vs adversarial or style-shifted data
  - reasoning dataset vs non-reasoning dataset
- Tags: `emergent_in_target`, `suppressed_in_target`, `source_only`,
  `target_only`, `stable_across_runs`, `changed_region`, `changed_domain`.
- Scores: activation-frequency ratio, region-distribution shift,
  domain-distribution shift, and top-example overlap.

### `reasoning_move`

- Add collection-time counters for local text around activations or run a
  post-pass over top examples.
- Candidate move families:
  - self-correction: "wait", "actually", "hold on", "mistake"
  - uncertainty: "maybe", "probably", "not sure", "could be"
  - planning: "first", "then", "let's", "strategy"
  - verification: "check", "verify", "substitute", "test"
  - finalization: "therefore", "so the answer", "final"
- Tags: `self_correction`, `uncertainty`, `planning`, `verification`,
  `backtracking`, `answer_commitment`.
- Scores: phrase-family counts, density near phrase matches, and activation
  lift inside phrase windows.

### `instruction_following`

- Collect data with explicit instruction categories or postprocess examples for
  instruction-following markers.
- Tags: `constraint_following`, `format_following`, `refusal_related`,
  `tool_instruction`, `style_instruction`, `safety_instruction`.
- Scores: activation density by instruction category and lift over generic
  answer regions.
- Why it is useful: could reveal features tied to instruction-tuned behavior
  rather than plain next-token prediction.

### `error_recovery`

- Collect or label traces with wrong turns, corrections, and successful
  recoveries.
- Tags: `error_detection`, `correction_start`, `recovery_step`,
  `failed_recovery`.
- Scores: activation density before and after correction phrases, plus
  contrast against examples without corrections.
- Why it is useful: targets a behavior that is central to reasoning traces and
  may differ strongly between base and instruction-tuned models.

### `answer_transition`

- Collect longer contexts around the transition from reasoning to final answer.
- Tags: `pre_answer`, `answer_start`, `answer_commitment`,
  `post_reasoning_summary`.
- Scores: distance to answer boundary, density in boundary windows, and logit
  effects for final-answer markers.
- Why it is useful: highlights features involved in moving from internal
  reasoning to externally visible answer production.

## Implementation Todo: Non-Auto-Interp, Non-Thinking-Token Ideas

This todo list focuses on the most interesting annotators that are not about
auto-interp and are not specifically about thinking/control tokens. It excludes
`auto_interp_import`, `auto_interp_confidence`, `thinking_timeline`, and the
`think_start_token` / `think_end_token` parts of structural-token annotation.

### Phase 1: Shared Infrastructure

- [ ] Add a helper for loading `features/{cantor_id}.json` from an annotator.
  Keep it optional so metadata-only annotators do not need per-feature JSON
  reads.
- [ ] Add a small utility module for concentration scores: entropy, Herfindahl
  index, max-fraction, and density-lift-over-rest.
- [ ] Add reusable helpers for extracting top activating examples, highlighted
  tokens, local token windows, and `top_logits` / `bottom_logits`.
- [ ] Add tests or lightweight fixtures that cover metadata-only annotation,
  per-feature JSON annotation, and update-in-place ownership semantics.

### Phase 2: Cheap High-Signal Annotators

- [ ] Implement `logit_effect`.
  - Read `top_logits` and `bottom_logits` from feature JSON files.
  - Tag promoted/suppressed token families such as numbers, newlines, code-like
    tokens, uncertainty words, refusal words, special tokens, and answer tokens.
  - Persist scores for token-family hit counts and top-family fractions.

- [ ] Implement `activation_shape`.
  - Read `tokens_acts_list` around top activating examples.
  - Tag `single_token_spike`, `sustained_context`, `ramp_up`, `ramp_down`, and
    `multi_token_phrase`.
  - Persist scores for peak width, activation mass near the highlighted token,
    left/right slope, and context activation entropy.

- [ ] Implement `feature_specificity`.
  - Use existing metadata to score concentration over domains and regions.
  - Optionally include activating-token concentration when feature JSONs are
    present.
  - Tag `high_precision`, `broad_context`, `domain_specialist`,
    `region_specialist`, and `token_specialist`.

- [ ] Implement `token_surface`.
  - Parse highlighted tokens in top activating examples.
  - Tag `digit_feature`, `operator_feature`, `punctuation_feature`,
    `newline_feature`, `capitalized_token`, `whitespace_token`, and
    `quote_token`.
  - Persist scores for each surface-class fraction and token diversity.

- [ ] Implement `cross_example_consistency`.
  - Measure agreement across top examples on highlighted token, nearby phrase,
    region, and output-logit family.
  - Tag `consistent_token`, `consistent_phrase`, `consistent_context`, and
    `incoherent_feature`.
  - Use this as a triage layer for selecting features worth manual inspection.

### Phase 3: Behavior-Focused Annotators

- [ ] Implement a first `reasoning_move` pass using existing feature JSONs.
  - Start with lexical windows around the highlighted token rather than
    recollecting data.
  - Tag `self_correction`, `uncertainty`, `planning`, `verification`,
    `answer_commitment`, and `backtracking`.
  - Persist phrase-family counts and activation lift near matched phrase
    windows.

- [ ] Add optional collection-time counters for `reasoning_move` if the
  feature-JSON pass is too brittle.
  - Count phrase-family windows during feature collection.
  - Store per-feature move-family counts in `feature_metadata.json`.
  - Keep the annotator rerunnable without recollecting once those counters
    exist.

- [ ] Implement `instruction_following`.
  - Use explicit data labels when available, otherwise parse examples for
    instruction categories.
  - Tag `constraint_following`, `format_following`, `refusal_related`,
    `tool_instruction`, `style_instruction`, and `safety_instruction`.
  - Compare against generic answer-region activation to avoid tagging broad
    answer features.

- [ ] Implement `error_recovery`.
  - Detect activations around wrong turns, corrections, and recovery phrases.
  - Tag `error_detection`, `correction_start`, `recovery_step`, and
    `failed_recovery`.
  - Prefer paired examples or labeled traces if available; otherwise start with
    lexical correction windows.

- [ ] Implement `answer_transition`.
  - Score activations around the transition into final-answer production.
  - Tag `pre_answer`, `answer_start`, `answer_commitment`, and
    `post_reasoning_summary`.
  - Combine with `logit_effect` to surface features that both fire near answer
    transition and promote answer-like tokens.

### Phase 4: Contrastive Runs

- [ ] Define the comparable-run contract for `contrastive_run`.
  - Require matching `cantor_id`, compatible layers/features, comparable
    thresholds, and comparable dataset slices.
  - Record source and target run metadata in the annotator scores.

- [ ] Implement metadata-only `contrastive_run`.
  - Compare activation frequency, region distribution, and domain distribution
    for each shared feature.
  - Tag `it_amplified`, `base_amplified`, `checkpoint_emergent`,
    `checkpoint_suppressed`, and `stable_feature`.

- [ ] Extend `contrastive_run` with per-feature JSON comparisons.
  - Compare top-example token surfaces, consistency scores, and logit-effect
    families across runs.
  - Add tags such as `changed_context`, `changed_output_effect`, and
    `stable_interpretation` only if they prove useful on real runs.

- [ ] Run `contrastive_run` on at least one real paired comparison.
  - Candidate comparisons: base vs instruct, checkpoint A vs checkpoint B, or
    reasoning dataset vs non-reasoning dataset.
  - Inspect the dashboard output before hardening thresholds.

### Phase 5: Wrapper And Dashboard Workflow

- [ ] Add one CLI module per annotator under `analysis/features/annotate/`, or
  add a small multi-annotator CLI if that keeps repeated runs simpler.
- [ ] Add shell wrappers only for annotators that are likely to be rerun often.
- [ ] Make each annotator log its top hits with useful scores for quick terminal
  inspection.
- [ ] Verify that rerunning one annotator updates only its own `auto_tags` and
  `auto_scores`.
- [ ] Run the selected annotators on a real `feature_data` directory and inspect
  the dashboard tag filters before considering the first implementation pass
  done.

## Suggested Build Order

For the non-auto-interp, non-thinking-token subset, implement in this order:

1. Implement `logit_effect`, `activation_shape`, `feature_specificity`, and
   `token_surface`.
   These are concrete, mostly local to existing data, and give useful dashboard
   filters quickly.
2. Add `cross_example_consistency`.
   This turns the first batch of annotators into a better manual-inspection
   triage workflow.
3. Implement `reasoning_move` from existing `features/*.json`, then decide
   whether collection-time counters are needed.
   Start heuristic and only recollect if the feature-JSON pass is too brittle.
4. Implement metadata-only `contrastive_run`, then run it on a real paired
   comparison before adding richer per-feature JSON comparisons.
   This is the highest-value recollecting direction.
5. Add `instruction_following`, `error_recovery`, and `answer_transition` once
   the first behavior taxonomy is useful in the dashboard.
