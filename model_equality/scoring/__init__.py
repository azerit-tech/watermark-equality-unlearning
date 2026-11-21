"""Local scoring, validation, partitioning and submission tooling for Model Equality Testing.

Modules:
    metrics    pairwise MCC scoring (family / model, two "within family" variants)
    csvio      reading, validating and writing submission CSVs
    partition  pairwise similarity/probability matrices -> transitive partition
    submit     guarded submission to the scoring server
"""
