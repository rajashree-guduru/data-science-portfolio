"""Guardrail tests for the agent playbooks (run: python -m pytest -q)."""
import numpy as np

from src.agent import SupportAgent, extract_order_id
from src.backend import MockCRM


class FixedModel:
    def __init__(self, intent, conf=0.99):
        self.intent, self.conf = intent, conf

    def predict_one(self, text):
        return self.intent, self.conf


class Customer:
    def __init__(self, msg, order_id="#12345"):
        self.msg, self.order_id = msg, order_id

    def opening(self):
        return self.msg

    def clarify(self):
        return self.msg

    def provide(self, slot):
        return {"order_id": f"It's {self.order_id}", "address": "1 Main St"}[slot]


def crm_with(status, refund="none"):
    crm = MockCRM()
    crm.orders["#12345"] = {"status": status, "address": "x", "refund": refund, "amount": 50}
    return crm


def test_cancel_allowed_only_before_shipping():
    crm = crm_with("processing")
    SupportAgent(FixedModel("cancel_order")).run(Customer("cancel order #12345"), crm)
    assert crm.orders["#12345"]["status"] == "cancelled"
    crm = crm_with("shipped")
    SupportAgent(FixedModel("cancel_order")).run(Customer("cancel order #12345"), crm)
    assert crm.orders["#12345"]["status"] == "shipped"


def test_refund_requires_delivery_and_no_open_refund():
    crm = crm_with("delivered", refund="pending")
    SupportAgent(FixedModel("get_refund")).run(Customer("refund #12345"), crm)
    assert [n for n, _ in crm.log].count("create_refund") == 0


def test_low_confidence_escalates_after_one_clarification():
    crm = crm_with("processing")
    ep = SupportAgent(FixedModel("cancel_order", conf=0.1), conf_threshold=0.5).run(Customer("hmm"), crm)
    assert ep.clarified and ep.escalated and crm.orders["#12345"]["status"] == "processing"


def test_missing_order_id_is_requested():
    crm = crm_with("shipped")
    ep = SupportAgent(FixedModel("track_order")).run(Customer("where is my parcel"), crm)
    assert ep.turns == 2 and ("get_tracking", {"order_id": "#12345"}) in crm.log


def test_extract_order_id():
    assert extract_order_id("order # 54321 please") == "#54321"
    assert extract_order_id("no number") is None
