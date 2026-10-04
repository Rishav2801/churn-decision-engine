# Design review

## Business question

Can a held-out risk ranking prioritize a limited retention budget, and how sensitive is that decision to unobserved treatment effectiveness?

## Analytical contract

The sample has no prospective outcome horizon or trustworthy event timestamps, so a random split supports only retrospective ranking. Predictions are not calibrated by a separate calibration model. The dashboard exposes calibration quality. Risk is not treatment effect; hypothetical economics require a retention experiment.

## Review prompts

1. Why use average precision instead of accuracy?
2. How is preprocessing leakage prevented?
3. Why can high churn risk imply low retention responsiveness?
4. Which assumptions would fail if the model were used six months later?

## Extension

Collect as-of timestamps and label windows, evaluate out-of-time performance, and estimate treatment response with a randomized pilot.
