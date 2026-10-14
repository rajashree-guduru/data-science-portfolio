"""Chat app for the heart-disease RAG explainer.  Run:  python app.py  then open the printed local URL."""
import os
os.environ.setdefault('LLM_BACKEND', 'hf')
import gradio as gr
import pandas as pd
import rag_core as R

model, X_train, X_test, y_test = R.train_model()
retriever = R.Retriever()
choices = [f"Test patient {i} (age {int(X_test.iloc[i]['age'])}, actual label {int(y_test.iloc[i])})" for i in range(len(X_test))]
DISCLAIMER = 'Portfolio demo: a screening aid trained on 303 patients, not medical advice.'


def patient_from(choice):
    return X_test.iloc[int(choice.split()[2])]


def run_explain(choice):
    out = R.explain(model, X_train, retriever, patient_from(choice))
    factors = pd.DataFrame(out['factors'])[['name', 'display', 'direction', 'contribution']].rename(
        columns={'name': 'Factor', 'display': 'Value', 'direction': 'Effect', 'contribution': 'Contribution'})
    sources = '**Retrieved passages**\n' + '\n'.join(f"- {p['title']} ({p['source']}, similarity {p['score']:.2f})" for p in out['passages'])
    return f"### Risk score: {out['risk']:.2f}", factors, out['answer'], sources, []


def chat(message, history, choice):
    out = R.explain(model, X_train, retriever, patient_from(choice), question=message)
    reply = f"{out['llm_text']}\n\n_AI answer from a small local LLM; may contain errors. Check it against the facts above._"
    return history + [{'role': 'user', 'content': message}, {'role': 'assistant', 'content': reply}], ''


with gr.Blocks(title='Heart Disease RAG Explainer') as demo:
    gr.Markdown('# Heart Disease Prediction: RAG Explainer\n' + DISCLAIMER)
    with gr.Row():
        pick = gr.Dropdown(choices, value=choices[0], label='Patient from the test set')
        btn = gr.Button('Explain', variant='primary')
    risk = gr.Markdown()
    with gr.Row():
        table = gr.Dataframe(label='Top factors behind the prediction')
        srcs = gr.Markdown()
    answer = gr.Markdown()
    gr.Markdown('### Ask a follow-up question')
    try:
        bot = gr.Chatbot(type='messages', height=300)
    except TypeError:          # Gradio 6+ uses the messages format by default
        bot = gr.Chatbot(height=300)
    msg = gr.Textbox(placeholder='e.g. What does ST depression mean for this patient?', label='Question')
    btn.click(run_explain, pick, [risk, table, answer, srcs, bot])
    msg.submit(chat, [msg, bot, pick], [bot, msg])

if __name__ == '__main__':
    demo.launch()
