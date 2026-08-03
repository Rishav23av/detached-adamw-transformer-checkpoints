# Detached AdamW Transformer Checkpoints

## Overview

This original dataset contains 320 decoder-only transformer checkpoints produced through actual from-scratch language-model training. Each model was trained on procedurally generated executable token streams using AdamW. No external text corpus, pretrained checkpoint, or downloaded benchmark is included.

One group of same-shaped optimizer-state tensors has been detached from its parameter names in every case. The visible parameter slots all come from the same attention projection role in different transformer layers. The anonymous optimizer bundles contain AdamW first and second moments, but their original layer ownership is stored only in `labels.csv`.

The data represents a checkpoint-recovery problem. It is intended for training and evaluating models that reason jointly over neural parameter tensors, optimizer moments, transformer depth, and healthy continuation telemetry.

## File Structure

- `manifest.json`: dataset version, deterministic generator seed, total case count, and provenance flags.
- `cases.jsonl`: one complete checkpoint-recovery record per line.
- `labels.csv`: the correct one-to-one slot-to-bundle ownership mapping for every raw case.
- `archives/`: compressed NumPy archives containing model parameters, attached optimizer state, anonymous candidate bundles, and canary token batches.
- `public_model.py`: PyTorch implementation of the transformer architecture and archive loader.
- `README.md`: short dataset guide.
- `SOURCE.md`: generation and provenance statement.
- `LICENSE.txt`: CC BY 4.0 license notice.

## Features

### `manifest.json`

- `dataset_version` (`string`): frozen schema version.
- `generator_seed` (`integer`): master seed used for deterministic raw generation.
- `case_count` (`integer`): number of records in `cases.jsonl`.
- `generator` (`string`): relative path of the dataset builder.
- `external_corpus` (`boolean`): always `false`; no external text was used.
- `external_checkpoint` (`boolean`): always `false`; every model was initialized and trained from scratch.

### `cases.jsonl`

- `case_id` (`string`): unique opaque checkpoint-recovery identifier.
- `archive_path` (`string`): relative path to the case's `.npz` archive.
- `split` (`string`): deterministic `train` or `test` assignment used by the challenge preparation pipeline.
- `model_spec` (`object`): transformer architecture with `vocab_size`, `max_length`, `dims`, `num_heads`, `num_layers`, and `mlp_dims`, all integers.
- `optimizer` (`object`): AdamW hyperparameters `learning_rate`, `beta1`, `beta2`, `eps`, and `weight_decay`, all floats.
- `optimizer_step` (`integer`): number of completed AdamW updates at the saved checkpoint.
- `training_steps` (`integer`): number of language-model training steps used to create the checkpoint.
- `language_family_count` (`integer`): number of procedural token-process families mixed during training.
- `projection_family` (`string`): attention projection role shared by every detached slot in the case: `query`, `key`, `value`, or `out`.
- `parameter_keys` (`object`): mapping from model parameter paths to array keys in the archive.
- `fixed_optimizer_state` (`list[object]`): attached optimizer moments for parameters outside the detached group. Each object contains `parameter_path`, `m_key`, and `v_key`.
- `slots` (`list[object]`): detached parameter slots. Each object contains opaque `slot_id` and its visible `parameter_path`.
- `bundles` (`list[object]`): anonymous AdamW candidate bundles. Each object contains `bundle_id`, `m_key`, and `v_key`.
- `canary_keys` (`list[string]`): archive keys for public canary token batches.
- `healthy_telemetry` (`list[object]`): healthy one-step continuation measurements for each canary batch: `optimizer_step`, `pre_loss`, `post_loss`, `gradient_l2`, `update_l2`, `post_logit_mean`, and `post_logit_std`, all numeric.
- `generation_audit` (`object`): direct-correlation leakage measurements recorded during raw generation.

### Case archives

Every archive contains only numeric arrays and can be opened with `numpy.load(..., allow_pickle=False)`.

- `parameter_####` (`float32`, variable shape): one named model parameter referenced by `parameter_keys`.
- `fixed_m_####` (`float32`, same shape as its parameter): attached AdamW first moment for a non-candidate parameter.
- `fixed_v_####` (`float32`, same shape as its parameter): attached AdamW second moment for a non-candidate parameter.
- `bundle_m_###` (`float32`, two-dimensional): anonymous first moment for one detached attention parameter.
- `bundle_v_###` (`float32`, same shape as `bundle_m_###`): anonymous second moment paired with that bundle.
- `canary_0`, `canary_1`, `canary_2` (`int32`, shape `[6, max_length]`): token batches used to produce healthy continuation telemetry.

Within a case, every detached parameter and every anonymous bundle has exactly the same shape and data type. Shapes therefore cannot reveal the answer.

### `labels.csv`

- `case_id` (`string`): case identifier from `cases.jsonl`.
- `slot_id` (`string`): detached parameter slot.
- `bundle_id` (`string`): anonymous bundle that originally belonged to the slot.

Each case's labels form a complete bijection: every slot occurs once and every candidate bundle is used once.

## Data Characteristics

- 320 independently initialized and trained checkpoints.
- The `split` field in `cases.jsonl` marks 243 records as `train` and 77 records as `test` for later challenge preparation.
- Five transformer architecture configurations with 12, 14, 16, 18, or 20 layers.
- Candidate groups contain 12 to 20 same-role, same-shaped attention parameters.
- AdamW learning rate, beta values, epsilon, weight decay, training horizon, token grammar, vocabulary permutation, and candidate order vary by case.
- Three canary continuation batches and seven aggregate telemetry values per batch.
- Splits are assigned by opaque hash and complete cases never cross splits.
- The raw package contains 5,120 slot assignments across approximately 1.4 GB of files.
- A direct parameter-to-moment correlation audit achieved 11.48% assignment accuracy.
- A trained handcrafted-statistics audit achieved 21.28% held-out assignment accuracy and 0.382 mean reciprocal rank.

## Source And License

This is an original procedurally generated dataset created for the Optimizer State Reattachment benchmark. The included generator trains every model and records genuine AdamW moment tensors. No external dataset or model weights are redistributed.

The dataset is released under the Creative Commons Attribution 4.0 International license.

## Intended Use

The dataset is intended for research and competition use in neural checkpoint recovery, optimizer-state forensics, tensor matching, set prediction, and training-dynamics modeling. It is synthetic and should not be treated as evidence about failures in any particular production training system.
