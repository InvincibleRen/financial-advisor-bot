# `explanation/` — plain-language explanation layer

Turns a structured recommendation into readable, non-technical prose. Currently
template-based over the rule-based signal (`advisor.py`); will be extended to
explain the ranked-selection output (why each top-N stock was chosen) and to
vary wording by signal strength. Optionally phrased by a local LLM that only
rewords pre-computed evidence — it never invents recommendations.
