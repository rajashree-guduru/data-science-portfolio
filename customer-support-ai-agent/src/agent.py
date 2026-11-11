"""Tool-using customer-support agent.

Loop per conversation:  understand -> (clarify) -> gather slots -> plan -> act -> observe -> respond
* Understanding uses the PyTorch intent model.
* Low-confidence predictions trigger ONE clarifying question, then a human handoff.
* Playbooks decide which tools to call and branch on tool observations
  (e.g. an order that already shipped cannot be cancelled).
* Every step is recorded as a thought / action / observation trace.

The planner is deliberately deterministic so its behaviour can be evaluated
exactly; the `plan()` method is the single place where an LLM planner could be
swapped in.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

ORDER_RE = re.compile(r"#\s?(\d{5})\b")

NEEDS_ORDER = {
    "cancel_order", "change_order", "track_order", "change_shipping_address",
    "track_refund", "get_refund", "get_invoice", "check_invoice",
}
TICKET_INTENTS = {"complaint", "payment_issue", "registration_problems"}


@dataclass
class Episode:
    intent_pred: str | None = None
    confidence: float = 0.0
    turns: int = 1
    clarified: bool = False
    escalated: bool = False
    trace: list[dict] = field(default_factory=list)
    reply: str = ""


def extract_order_id(text: str) -> str | None:
    m = ORDER_RE.search(text)
    return f"#{m.group(1)}" if m else None


def plan(intent: str, crm, order_id: str | None, ask, note=None) -> list[tuple[str, str]]:
    """Execute the playbook for an intent. Returns (thought, outcome) notes;
    `note` is an optional callback so the caller can log each step as it happens."""
    notes = []

    def add(thought, outcome):
        notes.append((thought, outcome))
        if note:
            note(thought, outcome)

    if intent == "contact_human_agent":
        crm.handoff_to_human("customer requested a human")
        add("Customer explicitly asked for a person", "handed off")
        return notes
    if intent in TICKET_INTENTS:
        crm.create_ticket(intent, order_id)
        add("Issue needs follow-up by a specialist team", f"ticket opened ({intent})")
        return notes
    if intent not in NEEDS_ORDER:
        doc = crm.policy_lookup(intent)
        add("Informational request: answer from policy", doc or "no policy found")
        return notes

    order = crm.lookup_order(order_id)
    if order is None:
        # the number in the message may be a typo or an unrelated number: confirm once
        add(f"Order {order_id} not found in CRM, asking customer to confirm the number", "lookup failed")
        order_id = extract_order_id(ask("order_id")) or order_id
        order = crm.lookup_order(order_id)
    if order is None:
        crm.handoff_to_human("order not found")
        add("Order id still not found", "handed off")
        return notes
    add(f"Order {order_id} status is {order['status']}", "checked")

    if intent == "cancel_order":
        if order["status"] == "processing":
            crm.cancel_order(order_id)
            add("Order has not shipped, cancellation allowed", "cancelled")
        else:
            add("Order already left the warehouse, cannot cancel", "explained return option")
    elif intent == "change_order":
        if order["status"] == "processing":
            crm.create_ticket("change_order", order_id)
            add("Edits possible before shipping", "change request ticket opened")
        else:
            add("Order already shipped, cannot edit", "explained")
    elif intent == "track_order":
        crm.get_tracking(order_id)
        add("Tracking requested", "status shared")
    elif intent == "change_shipping_address":
        if order["status"] == "delivered":
            add("Order already delivered, address cannot change", "explained")
        else:
            address = ask("address")
            crm.update_shipping_address(order_id, address)
            add("Address can still change", "address updated")
    elif intent == "track_refund":
        crm.get_refund_status(order_id)
        add("Refund status requested", "status shared")
    elif intent == "get_refund":
        if order["status"] == "delivered" and order["refund"] == "none":
            crm.create_refund(order_id)
            add("Delivered and no open refund: eligible", "refund created")
        else:
            add("Not eligible (not delivered or refund already open)", "explained policy")
    elif intent in {"get_invoice", "check_invoice"}:
        crm.get_invoice(order_id)
        add("Invoice requested", "invoice link shared")
    return notes


class SupportAgent:
    def __init__(self, model, conf_threshold: float = 0.5, max_clarify: int = 1):
        self.model = model
        self.th = conf_threshold
        self.max_clarify = max_clarify

    def run(self, customer, crm) -> Episode:
        ep = Episode()
        msg = customer.opening()
        for attempt in range(self.max_clarify + 1):
            intent, conf = self.model.predict_one(msg)
            ep.trace.append({"thought": "classify intent", "action": "intent_model",
                             "observation": f"{intent} ({conf:.2f})"})
            if conf >= self.th:
                break
            if attempt < self.max_clarify:
                ep.clarified = True
                ep.turns += 1
                msg = customer.clarify()
                ep.trace.append({"thought": "confidence below threshold", "action": "ask_clarifying_question",
                                 "observation": msg})
        ep.intent_pred, ep.confidence = intent, conf
        if conf < self.th:
            crm.handoff_to_human("low confidence")
            ep.escalated = True
            ep.trace.append({"thought": "still unsure after clarifying", "action": "handoff_to_human",
                             "observation": "queued"})
            return ep

        order_id = None
        if intent in NEEDS_ORDER:
            order_id = extract_order_id(msg)
            if order_id is None:
                ep.turns += 1
                order_id = extract_order_id(customer.provide("order_id"))
                ep.trace.append({"thought": "order number missing from message", "action": "ask_for_order_id",
                                 "observation": order_id})

        def ask(slot):
            ep.turns += 1
            answer = customer.provide(slot)
            ep.trace.append({"thought": f"need {slot}", "action": f"ask_for_{slot}", "observation": answer})
            return answer

        plan(intent, crm, order_id, ask,
             note=lambda t, o: ep.trace.append({"thought": t, "action": "tools", "observation": o}))
        ep.escalated = len(crm.handoffs) > 0 and intent != "contact_human_agent"
        ep.reply = ep.trace[-1]["observation"]
        return ep
