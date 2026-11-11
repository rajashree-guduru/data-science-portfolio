"""Mock customer-support backend exposing the tools the agent can call."""
from __future__ import annotations

import copy

import numpy as np

POLICIES = {
    "check_cancellation_fee": "Orders can be cancelled free of charge before they ship; after shipping a $5 restocking fee applies.",
    "check_payment_methods": "We accept Visa, Mastercard, Amex, PayPal and Apple Pay.",
    "check_refund_policy": "Delivered items can be refunded within 30 days of delivery.",
    "delivery_options": "Standard (5-7 days), Express (2-3 days) and Next-day delivery are available.",
    "delivery_period": "Standard delivery takes 5-7 business days after the order ships.",
    "place_order": "Add items to your cart and choose Checkout; you can pay as a guest or signed in.",
    "newsletter_subscription": "Manage newsletter preferences under Account > Notifications.",
    "create_account": "Choose Sign up, enter your email, and confirm the link we send you.",
    "set_up_shipping_address": "Add a default shipping address under Account > Addresses.",
    "recover_password": "Choose 'Forgot password' on the sign-in page to receive a reset link.",
    "edit_account": "Update your profile details under Account > Profile.",
    "switch_account": "Change your plan under Account > Plan; changes apply from the next billing cycle.",
    "delete_account": "Account deletion is available under Account > Privacy and takes effect in 14 days.",
    "contact_customer_service": "Support is available 24/7 by chat, or by phone 8am-8pm.",
    "review": "You can leave a review from Orders > Write a review.",
}


class MockCRM:
    """In-memory order store. Every tool call is logged for evaluation."""

    def __init__(self):
        self.orders: dict[str, dict] = {}
        self.tickets: list[dict] = []
        self.handoffs: list[str] = []
        self.log: list[tuple[str, dict]] = []

    @classmethod
    def with_order(cls, order_id: str, rng: np.random.Generator) -> "MockCRM":
        crm = cls()
        status = str(rng.choice(["processing", "shipped", "delivered"], p=[0.4, 0.3, 0.3]))
        crm.orders[order_id] = {
            "status": status,
            "address": "12 Oak Street, Austin",
            "refund": str(rng.choice(["none", "pending"], p=[0.7, 0.3])) if status == "delivered" else "none",
            "amount": int(rng.integers(15, 300)),
        }
        return crm

    def clone(self) -> "MockCRM":
        c = MockCRM()
        c.orders = copy.deepcopy(self.orders)
        return c

    def _rec(self, name, **kw):
        self.log.append((name, kw))

    # ---- tools -------------------------------------------------------
    def lookup_order(self, order_id):
        self._rec("lookup_order", order_id=order_id)
        o = self.orders.get(order_id)
        return dict(o) if o else None

    def cancel_order(self, order_id):
        self._rec("cancel_order", order_id=order_id)
        o = self.orders[order_id]
        if o["status"] != "processing":
            return {"ok": False, "reason": f"order already {o['status']}"}
        o["status"] = "cancelled"
        return {"ok": True}

    def get_tracking(self, order_id):
        self._rec("get_tracking", order_id=order_id)
        return {"status": self.orders[order_id]["status"]}

    def update_shipping_address(self, order_id, address):
        self._rec("update_shipping_address", order_id=order_id)
        o = self.orders[order_id]
        if o["status"] == "delivered":
            return {"ok": False, "reason": "already delivered"}
        o["address"] = address
        return {"ok": True}

    def get_refund_status(self, order_id):
        self._rec("get_refund_status", order_id=order_id)
        return {"refund": self.orders[order_id]["refund"]}

    def create_refund(self, order_id):
        self._rec("create_refund", order_id=order_id)
        o = self.orders[order_id]
        if o["status"] != "delivered" or o["refund"] != "none":
            return {"ok": False, "reason": "not eligible"}
        o["refund"] = "pending"
        return {"ok": True, "amount": o["amount"]}

    def get_invoice(self, order_id):
        self._rec("get_invoice", order_id=order_id)
        return {"invoice_url": f"https://example.com/invoices/{order_id.strip('#')}"}

    def policy_lookup(self, topic):
        self._rec("policy_lookup", topic=topic)
        return POLICIES.get(topic)

    def create_ticket(self, category, order_id=None):
        self._rec("create_ticket", category=category)
        self.tickets.append({"category": category, "order_id": order_id})
        return {"ticket_id": f"T{len(self.tickets):04d}"}

    def handoff_to_human(self, reason):
        self._rec("handoff_to_human", reason=reason)
        self.handoffs.append(reason)
        return {"queued": True}

    def state(self) -> dict:
        """Side-effect state used to compare agent vs. oracle outcomes."""
        return {
            "orders": copy.deepcopy(self.orders),
            "tickets": sorted(t["category"] for t in self.tickets),
            "handoffs": len(self.handoffs),
        }
