"""RAG explainer for the Heart Disease Prediction model.

Pipeline: predict risk -> find the patient's top contributing factors -> retrieve matching
knowledge-base passages with sentence embeddings -> explain. In the default hybrid design the
facts (risk, values, directions) come from a template so they are always correct, and the LLM
only adds plain-language context from the retrieved passages, with no numbers allowed.

LLM backends (set LLM_BACKEND):
  hf      (default) free, local: Hugging Face transformers runs Qwen2.5-1.5B-Instruct on CPU
  ollama  free, local: install Ollama, then `ollama pull llama3.2:3b`
  openai  needs OPENAI_API_KEY (model set by OPENAI_MODEL, default gpt-4o-mini)
  none    no LLM: returns the retrieved context only (useful for testing retrieval)
"""
import os, re, json, glob
import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data', 'heart.csv')
KB_DIR = os.path.join(HERE, 'knowledge_base')
CATEGORICAL = ['cp', 'restecg', 'slope', 'ca', 'thal']
NAMES = {'age': 'Age', 'sex': 'Sex', 'cp': 'Chest pain type', 'trestbps': 'Resting blood pressure',
         'chol': 'Serum cholesterol', 'fbs': 'Fasting blood sugar', 'restecg': 'Resting ECG result',
         'thalach': 'Maximum heart rate', 'exang': 'Exercise-induced angina', 'oldpeak': 'ST depression',
         'slope': 'ST slope', 'ca': 'Number of major vessels', 'thal': 'Thalassemia test result'}


# ---------- 1. Model: identical to the final model in heart_disease_prediction.ipynb ----------
def train_model():
    df = pd.read_csv(DATA)
    X, y = df.drop(columns='target'), df['target']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
    numeric = [c for c in X.columns if c not in CATEGORICAL]
    pre = ColumnTransformer([('num', StandardScaler(), numeric),
                             ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL)])
    model = make_pipeline(pre, LogisticRegression(max_iter=5000, C=0.1)).fit(X_tr, y_tr)
    return model, X_tr, X_te, y_te


def explain_patient(model, X_train, patient, top_k=3):
    """Risk score plus each feature's contribution (coefficient x deviation from the average
    training patient), grouped back to the original 13 features."""
    pre, lr = model[0], model[-1]
    names = pre.get_feature_names_out()
    x = pre.transform(patient.to_frame().T)
    x = x.toarray()[0] if hasattr(x, 'toarray') else np.asarray(x)[0]
    m = pre.transform(X_train)
    m = np.asarray(m.mean(axis=0)).ravel()
    contrib_cols = lr.coef_[0] * (x - m)
    grouped = {}
    for n, c in zip(names, contrib_cols):
        base = n.split('__', 1)[1]
        feat = base if base in NAMES else base.rsplit('_', 1)[0]
        grouped[feat] = grouped.get(feat, 0.0) + c
    factors = sorted(grouped.items(), key=lambda kv: -abs(kv[1]))
    risk = float(model.predict_proba(patient.to_frame().T)[0, 1])
    fmt = lambda v: int(v) if float(v).is_integer() else round(float(v), 1)
    top = [{'feature': f, 'name': NAMES[f], 'value': fmt(patient[f]), 'display': readable(f, patient[f]), 'contribution': round(float(v), 3),
            'direction': 'raises risk' if v > 0 else 'lowers risk'} for f, v in factors[:top_k]]
    return risk, top


UNITS = {'age': 'years', 'trestbps': 'mm Hg', 'chol': 'mg/dl', 'thalach': 'bpm', 'oldpeak': 'mm'}
CODES = {'sex': {0: 'female', 1: 'male'}, 'exang': {0: 'no', 1: 'yes'}, 'fbs': {0: 'no (120 mg/dl or below)', 1: 'yes (above 120 mg/dl)'}}


def readable(feature, value):
    v = int(value) if float(value).is_integer() else round(float(value), 1)
    if feature in CODES:
        return f"{CODES[feature].get(v, v)} (coded {v})"
    if feature in UNITS:
        return f"{v} {UNITS[feature]}"
    return f"code {v}"


# ---------- 2. Knowledge base + retrieval ----------
def load_chunks():
    chunks = []
    for path in sorted(glob.glob(os.path.join(KB_DIR, '*.md'))):
        text = open(path, encoding='utf-8').read()
        for sec in re.split(r'\n(?=## )', text):
            if sec.startswith('## '):
                title = sec.splitlines()[0][3:].strip()
                chunks.append({'source': os.path.basename(path), 'title': title, 'text': sec.strip()})
    return chunks


class Embedder:
    """Sentence-transformer embeddings (all-MiniLM-L6-v2). EMBED_BACKEND=tfidf is a test-only stand-in."""
    def __init__(self, corpus):
        self.backend = os.environ.get('EMBED_BACKEND', 'sentence-transformers')
        if self.backend == 'sentence-transformers':
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.model = TfidfVectorizer().fit(corpus)

    def encode(self, texts):
        if self.backend == 'sentence-transformers':
            return self.model.encode(texts, normalize_embeddings=True)
        v = self.model.transform(texts).toarray()
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


class Retriever:
    def __init__(self):
        self.chunks = load_chunks()
        self.embedder = Embedder([c['text'] for c in self.chunks])
        self.vectors = self.embedder.encode([c['text'] for c in self.chunks])

    def search(self, query, k=4):
        q = self.embedder.encode([query])[0]
        scores = self.vectors @ q
        return [dict(self.chunks[i], score=round(float(scores[i]), 3)) for i in np.argsort(-scores)[:k]]


# ---------- 3. Prompt + LLM ----------
SYSTEM = ("You explain a heart-disease risk model's prediction to a non-expert. Rules: "
          "1) Use ONLY the facts in the prompt and context. "
          "2) For each top factor, state its value exactly as written and say whether it pushed the risk score UP or DOWN, "
          "exactly as given; never reverse a direction. "
          "3) Directions are patterns the model learned from this dataset and may differ from general medical expectations; "
          "do not justify them with outside medical knowledge, and do not call a value normal or high unless the context gives the threshold. "
          "4) Be concise (under 170 words). "
          "5) End with one sentence saying this is a screening aid, not a diagnosis.")


def build_prompt(risk, factors, passages, question=None):
    fac = '\n'.join(f"- {f['name']}: {f.get('display', f['value'])} -> pushes the risk score {'UP' if f['contribution'] > 0 else 'DOWN'} "
                    f"(contribution {f['contribution']:+.2f})" for f in factors)
    ctx = '\n\n'.join(f"[{i + 1}] {p['text']}" for i, p in enumerate(passages))
    q = question or 'Explain why the model gave this risk score.'
    return f"Model risk score: {risk:.2f} (probability of heart disease)\nTop factors:\n{fac}\n\nContext:\n{ctx}\n\nQuestion: {q}"


_HF = {}


def _hf_generate(prompt):
    if 'pipe' not in _HF:
        import torch
        from transformers import pipeline
        _HF['pipe'] = pipeline('text-generation', model=os.environ.get('HF_MODEL', 'Qwen/Qwen2.5-1.5B-Instruct'),
                               torch_dtype=torch.bfloat16)
    msgs = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}]
    out = _HF['pipe'](msgs, max_new_tokens=260, do_sample=False)
    return out[0]['generated_text'][-1]['content'].strip()


def call_llm(prompt, backend=None):
    backend = backend or os.environ.get('LLM_BACKEND', 'hf')
    if backend == 'hf':
        return _hf_generate(prompt)
    if backend == 'ollama':
        r = requests.post('http://localhost:11434/api/chat', timeout=180, json={
            'model': os.environ.get('OLLAMA_MODEL', 'llama3.2:3b'), 'stream': False,
            'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}],
            'options': {'temperature': 0.2}})
        r.raise_for_status()
        return r.json()['message']['content'].strip()
    if backend == 'openai':
        from openai import OpenAI
        resp = OpenAI().chat.completions.create(model=os.environ.get('OPENAI_MODEL', 'gpt-4o-mini'), temperature=0.2,
            messages=[{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}])
        return resp.choices[0].message.content.strip()
    return '[LLM disabled: LLM_BACKEND=none]\n' + prompt


CONTEXT_SYSTEM = ("You rewrite medical definitions in plain language for a patient. For each definition given, write exactly "
                  "one short sentence that says what it measures. Use only the information in that definition: add no new facts, "
                  "no causes, no advice, no numbers, and do not say whether it raised or lowered risk. "
                  "Output one line per factor in the form '- Name: sentence'.")
SAFETY = "This is a screening aid built on 303 patients from one hospital, not a diagnosis."


def facts_block(risk, factors):
    """Deterministic, always-correct summary of the prediction (no LLM involved)."""
    level = 'high' if risk >= 0.7 else 'low' if risk <= 0.3 else 'uncertain (close to 50/50)'
    lines = [f"**Risk score: {risk:.2f}**, which the model rates as {level}. The main reasons:"]
    for f in factors:
        lines.append(f"- **{f['name']}: {f.get('display', f['value'])}**, which pushed the risk score "
                     f"{'up' if f['contribution'] > 0 else 'down'} (contribution {f['contribution']:+.2f}).")
    return '\n'.join(lines)


def call_llm_system(system, prompt, backend=None):
    global SYSTEM
    saved, SYSTEM = SYSTEM, system
    try:
        return call_llm(prompt, backend)
    finally:
        SYSTEM = saved


def explain(model, X_train, retriever, patient, question=None, k=4, backend=None, mode='grounded'):
    """mode='grounded' (default, final design): facts from a template, factor meanings quoted from the
    retrieved knowledge-base definitions, and the LLM used only to answer follow-up questions.
    mode='paraphrase': the LLM rewrites the definitions (v4).  mode='free': the LLM writes everything (v1/v2)."""
    risk, factors = explain_patient(model, X_train, patient)
    query = (question + ' ' if question else '') + ' '.join(f"{f['name']} ({f['feature']})" for f in factors) + ' risk score'
    passages = retriever.search(query, k=k)
    if mode == 'free':
        answer = call_llm(build_prompt(risk, factors, passages, question), backend)
        return {'risk': risk, 'factors': factors, 'passages': passages, 'answer': answer, 'llm_text': answer}
    defs = definitions(retriever, factors)
    llm_text = ''
    if question:
        ctx = facts_block(risk, factors) + '\n\n' + '\n\n'.join(f"[{i + 1}] {p['text']}" for i, p in enumerate(passages))
        llm_text = call_llm_system(QA_SYSTEM, f"Context:\n{ctx}\n\nQuestion: {question}", backend)
        meaning = f"**AI answer** _(generated by a small local LLM from the facts and passages above; may contain errors)_: {llm_text}"
    elif mode == 'paraphrase':
        llm_text = call_llm_system(CONTEXT_SYSTEM, '\n'.join(f"{n}: {d}" for n, d in defs), backend)
        meaning = f"**What these factors mean:** {llm_text}"
    else:
        meaning = '**What these factors mean** _(from the project knowledge base)_:\n' + '\n'.join(f"- **{n}:** {d}" for n, d in defs)
    answer = f"{facts_block(risk, factors)}\n\n{meaning}\n\n_{SAFETY}_"
    return {'risk': risk, 'factors': factors, 'passages': passages, 'answer': answer, 'llm_text': llm_text}


QA_SYSTEM = ("You answer a reader's question about one patient's heart-disease risk prediction. Use ONLY the facts and "
             "passages in the context. Copy any number exactly as it appears in the context and never introduce new numbers, "
             "percentages or statistics. Do not reverse any factor's direction. No medical advice. At most three sentences.")


def definitions(retriever, factors):
    """The retrieved definition sentence(s) for each top factor: the only facts the LLM may paraphrase."""
    out = []
    for f in factors:
        sec = next((c for c in retriever.chunks if c['title'].endswith(f"({f['feature']})")), None)
        out.append((f['name'], sec['text'].splitlines()[1] if sec else f['name']))
    return out


def unsupported_numbers(llm_text, source_text):
    """Numbers in the LLM's text that do not appear in the source it was given (list markers ignored)."""
    body = re.sub(r"(?m)^\s*\d+[.)]\s", "", llm_text)
    return [n for n in re.findall(r"\d+(?:\.\d+)?", body) if n not in source_text]


# ---------- 4. Evaluation ----------
UP_WORDS = ('positive', 'positively', 'up', 'raise', 'raises', 'raised', 'raising', 'increase', 'increases', 'increased', 'increasing', 'adds', 'higher risk')
DOWN_WORDS = ('negative', 'negatively', 'down', 'lower', 'lowers', 'lowered', 'lowering', 'reduce', 'reduces', 'reduced', 'reducing', 'decrease', 'decreases', 'decreased', 'lower risk')


def check_faithfulness(answer, factors):
    """Heuristic check: for each top factor, does the passage of the answer that discusses it state the
    same direction (UP/DOWN) as the model? Also checks coded values such as sex are stated correctly."""
    text = answer.lower()
    segs = [s for s in re.split(r'(?:\n+|(?<=[.!?])\s+)', text) if s.strip()]
    results = []
    for f in factors:
        key = f['name'].lower().split()[0]
        seg = ' '.join(s for s in segs if key in s)
        words = re.findall(r"[a-z]+", seg)
        up = sum(w in UP_WORDS for w in words) + seg.count('higher risk')
        down = sum(w in DOWN_WORDS for w in words) + seg.count('lower risk')
        ok = (up > down) if f['contribution'] > 0 else (down > up)
        if f['feature'] == 'sex':
            right, wrong = ('female', 'male') if f['value'] == 0 else ('male', 'female')
            if re.search(rf"\b{wrong}\b", seg) and not re.search(rf"\b{right}\b", seg):
                ok = False
        results.append(bool(ok))
    return sum(results) / len(results), results
