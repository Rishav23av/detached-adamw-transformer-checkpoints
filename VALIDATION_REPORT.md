# Validation Report

## Dataset Build

- Cases: 320
- Training cases: 243
- Hidden-label test cases: 77
- Hidden slot assignments: 5,120
- Unique archive hashes: 320
- Raw package size: approximately 1.4 GB
- Architectures: 12, 14, 16, 18, and 20 transformer layers
- Candidate group sizes: 12 to 20

## Integrity Checks

- Every `case_id` is unique.
- Every archive path exists and every archive has a unique SHA-256 digest.
- Every raw mapping is a complete slot-to-bundle bijection.
- Candidate first and second moments match the corresponding slot tensor shape.
- Test records contain no split field, raw mapping, generation audit, or target column.
- All public NumPy archives load with `allow_pickle=False`.

## Difficulty Audits

- Random-bijection slot accuracy: 6.25%
- Direct parameter-to-moment correlation accuracy: 11.48%
- Handcrafted logistic held-out slot accuracy: 21.28%
- Handcrafted logistic held-out mean reciprocal rank: 0.382

The audit gates reject a build if direct correlation reaches 30% or if the handcrafted learned baseline reaches 45% held-out slot accuracy. The completed build remains below both thresholds.

## Preparation Checks

- Prepared public files: 329
- Prepared private files: 1
- Prepared archives: 320
- Repeated preparation produced 330 byte-identical public/private files.
- `train.csv`: 243 rows
- `test.csv`: 77 rows
- `sample_submission.csv`: 77 rows
- `answers.csv`: 77 rows

## Grader Checks

- Sample submission score: 0.01921404541903157
- Full oracle score: 1.0
- Answer-subset oracle score: 1.0
- Missing case IDs: rejected
- Duplicate case IDs: rejected
- Invalid JSON: rejected
- Incomplete or non-bijective mappings: rejected by schema validation

## Cross-Framework Check

The public PyTorch runtime loaded a generated 18-layer case containing 509,664 parameters with an exact parameter-name schema match. All three canary losses were finite and agreed with the original MLX telemetry to floating-point precision.

## Scope

These checks establish package integrity, deterministic preparation, grader behavior, and resistance to the audited simple shortcuts. They do not guarantee a particular Shipd novelty score, agent score, or reviewer decision.
