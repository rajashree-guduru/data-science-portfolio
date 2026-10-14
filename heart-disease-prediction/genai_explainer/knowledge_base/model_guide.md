# Model guide: Heart Disease Prediction project

## What the model is
A logistic regression classifier trained on 70% of the 303 patients. Numeric features are standardised and categorical codes (chest pain type, resting ECG, ST slope, number of major vessels, thal) are one-hot encoded. It outputs a probability between 0 and 1 that the patient is labelled with heart disease.

## How accurate it is
On the 91-patient test set the model reaches 0.85 accuracy and 0.91 ROC-AUC, with recall of 0.89 (it identified 42 of 47 heart-disease patients). Ten-fold cross-validation gives the same 0.85 accuracy. A baseline that always predicts the most common class scores 0.52.

## How to read a risk score
A score above 0.5 means the model classifies the patient as likely having heart disease. Scores near 0.5 are uncertain; scores near 0 or 1 are confident. The score reflects patterns in 303 patients from one hospital, not an individual diagnosis.

## How factors are explained
Each factor's contribution is its model coefficient multiplied by how far the patient's value is from the average training patient. Positive contributions push the risk up, negative ones push it down. The factors with the largest absolute contributions are the main reasons for the prediction.

## Strongest factors overall
Across all patients, the features with the largest coefficients are the number of major vessels, chest pain type, thal result, sex, ST depression and maximum heart rate. Cholesterol has only a weak relationship with the label in this dataset (correlation -0.09).

## Limitations and safe use
The model was trained on a small sample from a single hospital and has not been validated elsewhere. It shows associations, not causes. It is a screening aid for discussion with a clinician, never a diagnosis or a basis for treatment decisions.
