"""Simulated customers + agent evaluation against an oracle.

For every conversation we run the agent, and separately run the playbook for the
TRUE intent on an identical copy of the CRM (the oracle). Comparing the two gives:

* task_success     - same end state in the CRM as the oracle AND the same customer-facing
                     tool calls (names + arguments, ignoring order lookups), with no escalation
* trajectory_match - same sequence of tool names (reasoning-path quality)
* wrong_action     - agent changed order state differently from the oracle
                     (e.g. cancelled when the customer only wanted tracking) - critical error
* escalated / clarified / turns
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .agent import SupportAgent, plan
from .backend import MockCRM
from .data import fill_placeholders, random_order_id


class SimulatedCustomer:
    def __init__(self, row, pool_by_intent, rng):
        self.rng = rng
        self.intent = row["intent"]
        self.raw = row["instruction"]
        self.order_id = random_order_id(rng)
        self.address = "500 Pine Ave, Denver"
        self.pool = pool_by_intent[self.intent]

    def opening(self):
        self.first_message = fill_placeholders(self.raw, self.rng, self.order_id)
        return self.first_message

    def clarify(self):
        alt = self.pool[int(self.rng.integers(len(self.pool)))]
        return fill_placeholders(alt, self.rng, self.order_id)

    def provide(self, slot):
        return {"order_id": f"It's {self.order_id}", "address": self.address}[slot]


def run_episodes(model, rows: pd.DataFrame, pool: pd.DataFrame, threshold: float, seed: int = 0,
                 keep_traces: int = 0):
    rng = np.random.default_rng(seed)
    pool_by_intent = pool.groupby("intent")["instruction"].apply(list).to_dict()
    agent = SupportAgent(model, conf_threshold=threshold)
    recs, traces = [], []
    for _, row in rows.iterrows():
        cust = SimulatedCustomer(row, pool_by_intent, rng)
        crm = MockCRM.with_order(cust.order_id, rng)
        oracle = crm.clone()
        ep = agent.run(cust, crm)
        plan(cust.intent, oracle, cust.order_id, lambda s: cust.provide(s))

        names = [n for n, _ in crm.log]
        oracle_names = [n for n, _ in oracle.log]
        low_conf_escalation = ep.escalated and ep.intent_pred is not None and ep.confidence < threshold
        effective = [c for c in crm.log if c[0] != "lookup_order"]
        oracle_effective = [c for c in oracle.log if c[0] != "lookup_order"]
        same_state = crm.state() == oracle.state()
        recs.append({
            "gold": cust.intent,
            "pred": ep.intent_pred,
            "confidence": ep.confidence,
            "has_typo": bool(row["has_typo"]),
            "task_success": same_state and effective == oracle_effective and not ep.escalated,
            "trajectory_match": names == oracle_names,
            "wrong_action": crm.state()["orders"] != oracle.state()["orders"] and not low_conf_escalation,
            "escalated": ep.escalated,
            "clarified": ep.clarified,
            "turns": ep.turns,
        })
        if len(traces) < keep_traces:
            traces.append({"message": cust.first_message, "pred": ep.intent_pred, "confidence": round(ep.confidence, 3), "gold": cust.intent, "trace": ep.trace})
    return pd.DataFrame(recs), traces


def summarize(df: pd.DataFrame) -> dict:
    return {
        "conversations": int(len(df)),
        "task_completion_rate": round(float(df.task_success.mean()), 4),
        "trajectory_match_rate": round(float(df.trajectory_match.mean()), 4),
        "wrong_action_rate": round(float(df.wrong_action.mean()), 4),
        "escalation_rate": round(float(df.escalated.mean()), 4),
        "clarification_rate": round(float(df.clarified.mean()), 4),
        "avg_turns": round(float(df.turns.mean()), 3),
        "intent_accuracy": round(float((df.gold == df.pred).mean()), 4),
    }
