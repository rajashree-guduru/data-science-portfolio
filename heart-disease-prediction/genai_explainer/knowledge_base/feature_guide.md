# Feature guide: Heart Disease Prediction project

Each section describes one input feature, how it is coded, and what this dataset shows. Dataset: UCI Cleveland Heart Disease, 303 patients (165 labelled with heart disease, 138 without).

## Age (age)
Patient age in years.
In this dataset the average age is 54.4 overall: 52.5 for patients labelled with heart disease vs. 56.6 for those without (correlation with the label -0.23).

## Sex (sex)
Biological sex, coded 1 = male, 0 = female.
In this dataset: value 0: 96 patients, 75% labelled with heart disease; value 1: 207 patients, 45% labelled with heart disease.

## Chest pain type (cp)
Type of chest pain reported, coded 0-3. In this dataset type 0 is the most common among patients without heart disease, while types 1-3 are more common among patients with it.
In this dataset: value 0: 143 patients, 27% labelled with heart disease; value 1: 50 patients, 82% labelled with heart disease; value 2: 87 patients, 79% labelled with heart disease; value 3: 23 patients, 70% labelled with heart disease.

## Resting blood pressure (trestbps)
Resting systolic blood pressure in mm Hg, measured on admission. Readings of 130 mm Hg or higher are generally considered elevated.
In this dataset the average resting blood pressure is 131.6 overall: 129.3 for patients labelled with heart disease vs. 134.4 for those without (correlation with the label -0.14).

## Serum cholesterol (chol)
Total serum cholesterol in mg/dl. Levels above 240 mg/dl are generally considered high.
In this dataset the average serum cholesterol is 246.3 overall: 242.2 for patients labelled with heart disease vs. 251.1 for those without (correlation with the label -0.09).

## Fasting blood sugar (fbs)
Whether fasting blood sugar is above 120 mg/dl (1 = yes, 0 = no), a marker associated with diabetes risk.
In this dataset: value 0: 258 patients, 55% labelled with heart disease; value 1: 45 patients, 51% labelled with heart disease.

## Resting ECG result (restecg)
Resting electrocardiogram result, coded 0-2 (0 = normal, 1 = ST-T wave abnormality, 2 = probable or definite left ventricular hypertrophy).
In this dataset: value 0: 147 patients, 46% labelled with heart disease; value 1: 152 patients, 63% labelled with heart disease; value 2: 4 patients, 25% labelled with heart disease.

## Maximum heart rate (thalach)
Maximum heart rate (beats per minute) reached during an exercise stress test. Maximum achievable heart rate naturally declines with age.
In this dataset the average maximum heart rate is 149.6 overall: 158.5 for patients labelled with heart disease vs. 139.1 for those without (correlation with the label +0.42).

## Exercise-induced angina (exang)
Whether exercise brought on chest pain (angina): 1 = yes, 0 = no.
In this dataset: value 0: 204 patients, 70% labelled with heart disease; value 1: 99 patients, 23% labelled with heart disease.

## ST depression (oldpeak)
ST-segment depression on the ECG induced by exercise relative to rest, in millimetres. ST depression on an exercise ECG can indicate reduced blood flow to the heart muscle.
In this dataset the average st depression is 1.0 overall: 0.6 for patients labelled with heart disease vs. 1.6 for those without (correlation with the label -0.43).

## ST slope (slope)
Slope of the peak exercise ST segment, coded 0-2.
In this dataset: value 0: 21 patients, 43% labelled with heart disease; value 1: 140 patients, 35% labelled with heart disease; value 2: 142 patients, 75% labelled with heart disease.

## Number of major vessels (ca)
Number of major blood vessels (0-4) coloured by fluoroscopy.
In this dataset: value 0: 175 patients, 74% labelled with heart disease; value 1: 65 patients, 32% labelled with heart disease; value 2: 38 patients, 18% labelled with heart disease; value 3: 20 patients, 15% labelled with heart disease; value 4: 5 patients, 80% labelled with heart disease.

## Thalassemia test result (thal)
Result of a thallium stress test / thalassemia indicator, coded 0-3 in this dataset.
In this dataset: value 0: 2 patients, 50% labelled with heart disease; value 1: 18 patients, 33% labelled with heart disease; value 2: 166 patients, 78% labelled with heart disease; value 3: 117 patients, 24% labelled with heart disease.
