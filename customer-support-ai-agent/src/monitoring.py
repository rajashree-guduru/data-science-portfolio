"""Production monitoring: drift statistics and alert rules.

Signals that need NO labels (available in real time):
  * PSI of the predicted-intent distribution vs. the reference window (prior / concept drift)
  * mean confidence, share of low-confidence predictions, KS test on confidence
  * out-of-vocabulary token rate (input / covariate drift)
  * escalation and clarification rates (agent KPIs)
Signal that needs labels (weekly human-reviewed sample):
  * task completion rate
"""
from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp

PSI_ALERT = 0.10
CONF_DROP_ALERT = 0.05
OOV_RISE_ALERT = 0.03
KS_P_ALERT = 0.01


def psi(ref_labels, cur_labels, categories, eps: float = 1e-4) -> float:
    ref = np.array([np.mean(np.asarray(ref_labels) == c) for c in categories]) + eps
    cur = np.array([np.mean(np.asarray(cur_labels) == c) for c in categories]) + eps
    ref, cur = ref / ref.sum(), cur / cur.sum()
    return float(np.sum((cur - ref) * np.log(cur / ref)))


class DriftMonitor:
    def __init__(self, reference: dict, categories):
        """reference: {'pred': [...], 'conf': [...], 'oov': float}"""
        self.ref = reference
        self.categories = categories

    def check(self, pred, conf, oov, low_conf_threshold) -> dict:
        conf = np.asarray(conf)
        stats = {
            "psi_intent": psi(self.ref["pred"], pred, self.categories),
            "mean_conf": float(conf.mean()),
            "low_conf_rate": float((conf < low_conf_threshold).mean()),
            "oov_rate": oov,
            "ks_conf_p": float(ks_2samp(self.ref["conf"], conf).pvalue),
        }
        alerts = []
        if stats["psi_intent"] > PSI_ALERT:
            alerts.append("intent-mix drift (PSI)")
        if self.ref["mean_conf"] - stats["mean_conf"] > CONF_DROP_ALERT:
            alerts.append("confidence drop")
        if stats["oov_rate"] - self.ref["oov"] > OOV_RISE_ALERT:
            alerts.append("input drift (OOV)")
        if stats["ks_conf_p"] < KS_P_ALERT:
            alerts.append("confidence distribution shift (KS)")
        stats["alerts"] = alerts
        return stats
