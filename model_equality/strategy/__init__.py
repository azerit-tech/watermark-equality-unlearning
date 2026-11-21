"""Submission strategy for Model Equality Testing: which partition to submit next, what a score teaches.

The public score is deterministic, the submission kept is the best public one, and it is judged on the
other pairs.  The module turns every 30-minute slot into a test of uncertain decisions (an endpoint joins
or leaves a group, two groups merge) and keeps a belief over the true partition:

    core      belief model (decoding of the scores, posterior over the true partition, proposal)
    cli       update / next / assess / final / status / simulate  (python3.12 -m model_equality.strategy ...)
    simulate  synthetic truths and comparison with the naive strategy

Usage, from the repository root (nothing here sends a submission or queries the API):

    python3.12 -m model_equality.strategy update     # after every score
    python3.12 -m model_equality.strategy next       # writes strategy/out/next/partition.csv + summary.txt
    python3.12 -m model_equality.strategy assess FILE.csv
    python3.12 -m model_equality.strategy final

Tests:  python3.12 -m unittest discover -s model_equality/strategy/tests -t .
"""
