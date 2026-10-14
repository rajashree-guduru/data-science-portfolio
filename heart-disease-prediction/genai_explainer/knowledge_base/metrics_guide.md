# Metrics guide

## Accuracy
The share of patients the model classifies correctly.

## Recall (sensitivity)
Of the patients who truly have heart disease, the share the model correctly flags. In screening, high recall matters because a missed case is costly.

## Precision
Of the patients the model flags as at risk, the share who truly have heart disease.

## ROC-AUC
How well the model ranks patients by risk, from 0.5 (random) to 1.0 (perfect). An AUC of 0.91 means a randomly chosen patient with heart disease receives a higher score than a randomly chosen patient without it 91% of the time.
