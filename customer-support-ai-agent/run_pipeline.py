"""End-to-end pipeline: train -> evaluate agent -> simulate 10 weeks in production with drift -> retrain.

Usage:  python run_pipeline.py          (about 1-2 minutes on a laptop CPU)
All outputs are written to results/.
"""
from __future__ import annotations

import json
import os

import matplotlib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

import torch

from src.data import fill_placeholders, load_dataset, make_splits
from src.model import IntentModel
from src.monitoring import DriftMonitor
from src.simulate import run_episodes, summarize

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEED = 42
OUT = os.path.join(os.path.dirname(__file__), "results")
WEEKS = 10
PER_WEEK = 500
TYPO_SHARE = [0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.6, 0.6, 0.6, 0.6]
SPIKE_FROM_WEEK = 6                      # 1-indexed: refund / cancellation surge
SPIKE_INTENTS = ["get_refund", "track_refund", "cancel_order", "track_order"]
SPIKE_SHARE = 0.30


def filled(texts, rng):
    return [fill_placeholders(t, rng) for t in texts]


def main():
    torch.set_num_threads(1)
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    df = load_dataset()
    sp = make_splits(df, SEED)
    print({k: len(v) for k, v in sp.items()})

    X_tr, y_tr = filled(sp["train"].instruction, rng), list(sp["train"].intent)
    X_va, y_va = filled(sp["val"].instruction, rng), list(sp["val"].intent)
    X_ty, y_ty = filled(sp["typo_holdout"].instruction, rng), list(sp["typo_holdout"].intent)

    # ---------------- 1. Models -------------------------------------------------
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    lr = LogisticRegression(max_iter=2000, C=10).fit(tfidf.fit_transform(X_tr), y_tr)
    v1 = IntentModel(seed=SEED, version="v1")
    history = v1.fit(X_tr, y_tr, X_va, y_va, epochs=10)

    def scores(name, pred_va, pred_ty):
        return {"model": name,
                "clean_val_accuracy": round(accuracy_score(y_va, pred_va), 4),
                "clean_val_macro_f1": round(f1_score(y_va, pred_va, average="macro"), 4),
                "typo_holdout_accuracy": round(accuracy_score(y_ty, pred_ty), 4),
                "typo_holdout_macro_f1": round(f1_score(y_ty, pred_ty, average="macro"), 4)}

    model_rows = [
        scores("TF-IDF + Logistic Regression (scikit-learn baseline)",
               lr.predict(tfidf.transform(X_va)), lr.predict(tfidf.transform(X_ty))),
        scores("PyTorch fastText-style classifier v1", v1.predict(X_va), v1.predict(X_ty)),
    ]

    # ---------------- 2. Agent evaluation: threshold sweep ----------------------
    # pre-launch stress test: clean validation messages + 25% held-out typo messages
    sweep_pool = pd.concat([sp["val"], sp["typo_val"]])
    sweep_rows = pd.concat([sp["val"].sample(1125, random_state=SEED), sp["typo_val"].sample(375, random_state=SEED)])
    sweep = []
    for th in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        res, _ = run_episodes(v1, sweep_rows, sweep_pool, th, seed=SEED)
        sweep.append({"threshold": th, **summarize(res)})
    sweep = pd.DataFrame(sweep)
    # pick the threshold with best completion after penalising wrong actions 5x
    best_th = float(sweep.loc[(sweep.task_completion_rate - 5 * sweep.wrong_action_rate).idxmax(), "threshold"])

    prod_clean = sp["prod_clean"].sample(frac=1, random_state=SEED).reset_index(drop=True)
    offline_test, prod_clean = prod_clean.iloc[:2000], prod_clean.iloc[2000:].reset_index(drop=True)
    off_res, traces = run_episodes(v1, offline_test, sp["val"], best_th, seed=SEED + 1, keep_traces=400)
    offline_agent = summarize(off_res)

    # ---------------- 3. Ten weeks in production --------------------------------
    pools = {"clean": prod_clean, "typo": sp["typo_prod"].sample(frac=1, random_state=SEED).reset_index(drop=True)}
    used = {"clean": set(), "typo": set()}

    def draw(kind, n, intents=None):
        p = pools[kind]
        idx = [i for i in p.index if i not in used[kind] and (intents is None or p.at[i, "intent"] in intents)]
        take = idx[:n]
        used[kind].update(take)
        return p.loc[take]

    ref_texts = X_va
    ref_conf = v1.predict_proba(ref_texts).max(1)
    reference = {"pred": v1.predict(ref_texts), "conf": ref_conf, "mean_conf": float(ref_conf.mean()),
                 "oov": v1.oov_rate(ref_texts)}
    monitor = DriftMonitor(reference, v1.labels)

    model, weekly, alert_streak, retrain_week = v1, [], 0, None
    drift_log = []                         # human-reviewed production messages (gold labels)
    for w in range(1, WEEKS + 1):
        n_typo = int(PER_WEEK * TYPO_SHARE[w - 1])
        n_spike = int(PER_WEEK * SPIKE_SHARE) if w >= SPIKE_FROM_WEEK else 0
        n_spike_typo = int(n_spike * TYPO_SHARE[w - 1])
        parts = [draw("typo", n_spike_typo, SPIKE_INTENTS), draw("clean", n_spike - n_spike_typo, SPIKE_INTENTS),
                 draw("typo", n_typo - n_spike_typo), draw("clean", PER_WEEK - n_typo - (n_spike - n_spike_typo))]
        week = pd.concat(parts).sample(frac=1, random_state=w)
        wrng = np.random.default_rng(SEED + w)
        texts = filled(week.instruction, wrng)
        proba = model.predict_proba(texts)
        stats = monitor.check(np.array(model.labels)[proba.argmax(1)], proba.max(1), model.oov_rate(texts), best_th)
        # clarifying replies come from the same mix of clean + typo phrasing customers use
        res, _ = run_episodes(model, week, sweep_pool, best_th, seed=SEED + 100 + w)
        kpi = summarize(res)
        weekly.append({"week": w, "model": model.version, "typo_share": TYPO_SHARE[w - 1],
                       "refund_cancel_spike": w >= SPIKE_FROM_WEEK, **{k: v for k, v in stats.items() if k != "alerts"},
                       "alerts": "; ".join(stats["alerts"]), **kpi})
        print(f"week {w:2d} {model.version} completion={kpi['task_completion_rate']:.3f} "
              f"escal={kpi['escalation_rate']:.3f} wrong={kpi['wrong_action_rate']:.3f} psi={stats['psi_intent']:.3f} oov={stats['oov_rate']:.3f} {stats['alerts']}")

        if stats["alerts"]:
            drift_log.append(pd.DataFrame({"text": texts, "intent": list(week.intent)}))
        input_alert = any(a in stats["alerts"] for a in ["input drift (OOV)", "confidence drop"])
        alert_streak = alert_streak + 1 if input_alert else 0
        if alert_streak >= 2 and retrain_week is None:
            # Retrain on original training data + human-labelled messages from the drifted weeks.
            # 20% of the reviewed messages are held back to form the new monitoring reference window.
            reviewed = pd.concat(drift_log).sample(frac=1, random_state=SEED)
            n_ref = len(reviewed) // 5
            ref_win, extra = reviewed.iloc[:n_ref], reviewed.iloc[n_ref:]
            v2 = IntentModel(seed=SEED, version="v2")
            v2.fit(X_tr + list(extra.text), y_tr + list(extra.intent), epochs=10)
            model, retrain_week = v2, w
            # the post-retrain week becomes the new reference window ("new normal")
            rt = list(ref_win.text)
            p2 = v2.predict_proba(rt)
            monitor = DriftMonitor({"pred": np.array(v2.labels)[p2.argmax(1)], "conf": p2.max(1),
                                    "mean_conf": float(p2.max(1).mean()), "oov": v2.oov_rate(rt)}, v2.labels)
            print(f"  -> retrained v2 on {len(extra)} reviewed production messages; deployed for week {w + 1}")
            retrain_info = {"retrain_after_week": w, "reviewed_messages_added": int(len(extra)),
                            "reference_window_messages": int(n_ref)}

    weekly = pd.DataFrame(weekly)
    v2_typo_rows = scores("PyTorch classifier v2 (retrained after drift)", model.predict(X_va), model.predict(X_ty)) \
        if retrain_week else None
    if v2_typo_rows:
        model_rows.append(v2_typo_rows)

    # ---------------- 4. Save outputs ------------------------------------------
    pd.DataFrame(model_rows).to_csv(f"{OUT}/model_comparison.csv", index=False)
    pd.DataFrame(history).to_csv(f"{OUT}/training_history_v1.csv", index=False)
    sweep.to_csv(f"{OUT}/threshold_sweep.csv", index=False)
    weekly.to_csv(f"{OUT}/weekly_monitoring.csv", index=False)
    per_intent = off_res.groupby("gold").agg(conversations=("task_success", "size"),
                                             task_completion=("task_success", "mean"),
                                             wrong_action=("wrong_action", "mean")).round(3)
    per_intent.sort_values("task_completion").to_csv(f"{OUT}/offline_agent_by_intent.csv")
    summary = {"dataset_rows": int(len(df)), "splits": {k: int(len(v)) for k, v in sp.items()},
               "chosen_confidence_threshold": best_th, "offline_agent": offline_agent,
               "retrain": retrain_info if retrain_week else None, "models": model_rows}
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # example traces: one success, one clarification, one escalation, one wrong action if any
    off_res = off_res.reset_index(drop=True)
    picks = []
    for cond in [off_res.task_success & ~off_res.clarified, off_res.clarified & off_res.task_success,
                 off_res.escalated, off_res.wrong_action]:
        hits = [i for i in off_res.index[cond] if i < len(traces)]
        if hits:
            picks.append(hits[0])
    with open(f"{OUT}/example_traces.md", "w") as f:
        f.write("# Example agent traces (offline test set)\n\n")
        for i in picks:
            t, r = traces[i], off_res.loc[i]
            f.write(f"## Customer: \"{t['message']}\"\n\nTrue intent: `{t['gold']}`  |  predicted: `{t['pred']}` "
                    f"({t['confidence']})  |  task success: {bool(r.task_success)}  |  wrong action: {bool(r.wrong_action)}\n\n")
            f.write("| # | Thought | Action | Observation |\n|---|---|---|---|\n")
            for k, s in enumerate(t["trace"], 1):
                f.write(f"| {k} | {s['thought']} | {s['action']} | {s['observation']} |\n")
            f.write("\n")

    # dashboard
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    wk = weekly.week
    def mark(a):
        a.axvspan(3.5, WEEKS + 0.5, color="#f4a261", alpha=0.08)
        if retrain_week:
            a.axvline(retrain_week + 0.5, color="#2a9d8f", ls="--", lw=1.5, label="v2 deployed")
        a.set_xticks(wk); a.set_xlabel("week"); a.grid(alpha=0.3)
    ax[0, 0].plot(wk, weekly.task_completion_rate, "o-", label="task completion")
    ax[0, 0].plot(wk, weekly.intent_accuracy, "s--", label="intent accuracy")
    ax[0, 0].set_ylim(0.95, 1.005)
    ax[0, 0].set_title("Agent outcome KPIs (weekly labelled sample)"); mark(ax[0, 0]); ax[0, 0].legend(loc="lower left")
    ax[0, 1].plot(wk, weekly.clarification_rate, "o-", label="clarifying-question rate")
    ax[0, 1].plot(wk, weekly.low_conf_rate, "s--", label="low-confidence share")
    ax[0, 1].set_title("Customer friction & confidence (no labels needed)"); mark(ax[0, 1]); ax[0, 1].legend(loc="upper left")
    ax[1, 0].plot(wk, weekly.oov_rate, "o-", color="#e76f51")
    ax[1, 0].set_title("Input drift: out-of-vocabulary token rate"); mark(ax[1, 0])
    ax[1, 1].bar(wk, weekly.psi_intent, color="#264653")
    ax[1, 1].axhline(0.10, color="red", ls=":", label="PSI alert (0.10)")
    ax[1, 1].set_title("Intent-mix drift: PSI vs reference"); mark(ax[1, 1]); ax[1, 1].legend()
    fig.suptitle("Customer-support agent: 10 simulated production weeks (typo drift from week 4, refund surge from week 6)")
    fig.savefig(f"{OUT}/monitoring_dashboard.png", dpi=130)

    fig, a = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    a.plot(sweep.threshold, sweep.task_completion_rate, "o-", label="task completion")
    a.plot(sweep.threshold, sweep.escalation_rate, "s-", label="escalation")
    a.plot(sweep.threshold, sweep.wrong_action_rate * 10, "^-", label="wrong-action rate x10")
    a.axvline(best_th, color="grey", ls="--", label=f"chosen = {best_th}")
    a.set_xlabel("confidence threshold"); a.set_title("Agent threshold trade-off (validation)"); a.grid(alpha=0.3); a.legend()
    fig.savefig(f"{OUT}/threshold_tradeoff.png", dpi=130)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
