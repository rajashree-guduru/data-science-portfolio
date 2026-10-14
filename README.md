# Data Science Portfolio: Rajashree Guduru

Projects from my M.S. at Sacred Heart University, covering classification, regression, clustering, driver analysis, Generative AI, a Flask web app and interactive Tableau dashboards.

| Project | Question | Methods | Key result |
|---|---|---|---|
| [Heart Disease Prediction](heart-disease-prediction/) | Can clinical measurements predict heart disease? | Logistic regression, decision tree, random forest, one-hot encoding, 10-fold CV | **0.85 accuracy, 0.91 ROC-AUC** vs. 0.52 baseline |
| [Online News Popularity](online-news-popularity/) | What drives article shares, and can popularity be predicted? | OLS with controls, logistic regression, gradient boosting, TF-IDF headline features, permutation importance | **0.73 ROC-AUC** vs. 0.50 baseline; headline text added no lift |
| [USA Housing Rental Analysis](usa-housing-rental-analysis/) | What drives rental prices across 385K U.S. listings? | Data cleaning, random forest, gradient boosting, state-level prediction-error analysis | **R² 0.00 → 0.76** after cleaning; typical error \$176/month |
| [Tableau Dashboards](tableau-dashboards/) | How did LaGuardia weather change year over year? How do the healthiest foods compare? | Tableau: time series, YoY table calculations, dual-axis charts, filled map, calculated fields, interactive filters | 2 dashboards, 7 sheets |
| [University Clustering App](university-clustering-app/) | What groups of universities exist, and can they fill in missing data? | K-means, standardisation, silhouette analysis, cluster-based imputation, Flask web app with login, pytest | 4 segments; 93% match sector; imputation error 13-34% lower than mean-fill |

**Generative AI extensions:** a [RAG explainer and LLM chat app](heart-disease-prediction/genai_explainer/) for the heart-disease model (92.7% retrieval hit rate, 5/5 chat answers with no invented numbers) and [LLM headline embeddings](online-news-popularity/llm_headline_embeddings.ipynb) with topic clustering for the news project. Run instructions: [GENAI_SETUP.md](GENAI_SETUP.md).

Each project folder contains a **PDF report**, a **PowerPoint presentation**, an executed **Jupyter notebook** and its charts.

## Run the notebooks
```bash
pip install -r requirements.txt
jupyter notebook
```
Each project folder has its own README with the question, data, method and results.
