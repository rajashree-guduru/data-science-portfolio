# Running the Generative AI parts (Windows + Anaconda)

Two GenAI extensions run on your own laptop:

| Project | Notebook | What it does |
|---|---|---|
| Online News Popularity | `online-news-popularity/llm_headline_embeddings.ipynb` | Sentence-transformer embeddings of 39,644 headlines; tests whether they improve popularity prediction; topic clusters |
| Heart Disease Prediction | `heart-disease-prediction/genai_explainer/rag_explainer.ipynb` + `app.py` | RAG pipeline + local LLM that explains each prediction; Gradio chat app |

Both notebooks have already been run; their outputs, charts and `results/*.json` files are in the repo. Re-run them only if you want to reproduce or change something.

## 1. Install the Python packages (once, about 5-10 minutes)
Open **Anaconda Prompt** (Start menu → type "Anaconda Prompt") and run:
```
pip install sentence-transformers transformers torch gradio requests
```
The default LLM (Qwen2.5-1.5B-Instruct, about 3 GB) downloads automatically from Hugging Face the first time and runs on your CPU.

## 2. Optional: other LLMs
- **Ollama:** install from https://ollama.com/download, run `ollama pull llama3.2:3b`, then `set LLM_BACKEND=ollama` before starting Jupyter.
- **OpenAI:** `set LLM_BACKEND=openai` and `set OPENAI_API_KEY=sk-...`

## 3. Run the News embeddings notebook
1. `OnlineNewsPopularity.csv` is already in `online-news-popularity/data/` (GitHub Desktop skips uploading it; the source is https://archive.ics.uci.edu/dataset/332/online+news+popularity).
2. In Anaconda Prompt: `cd %USERPROFILE%\Downloads\data-science-portfolio` then `jupyter notebook`.
3. Open `online-news-popularity/llm_headline_embeddings.ipynb` and choose **Kernel → Restart & Run All**. The first run downloads the embedding model (about 90 MB) and takes 5-10 minutes.
4. It saves charts to `images/` and numbers to `results/embedding_results.json`.

## 4. Run the Heart RAG explainer
1. Open `heart-disease-prediction/genai_explainer/rag_explainer.ipynb` and choose **Kernel → Restart & Run All** (a few minutes).
2. It saves `results/rag_results.json`.
3. Chat app: in Anaconda Prompt, `cd` into `heart-disease-prediction\genai_explainer` and run `python app.py`, then open the local link (usually http://127.0.0.1:7860). Take a screenshot for your README.

## 5. Results files
- `online-news-popularity/results/embedding_results.json`
- `heart-disease-prediction/genai_explainer/results/rag_results.json`
