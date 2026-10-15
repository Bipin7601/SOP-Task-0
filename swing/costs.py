"""Zerodha equity-delivery transaction costs."""

from __future__ import annotations


def buy_cost(value: float, c) -> float:
    """Statutory charges on a delivery buy of `value` INR (slippage is applied to the price, not here)."""
    fees = value * (c.brokerage_pct + c.exchange_txn_pct + c.sebi_fee_pct) / 100
    gst = fees * c.gst_pct / 100
    return fees + gst + value * (c.stt_pct + c.stamp_duty_pct) / 100


def sell_cost(value: float, c) -> float:
    """Statutory charges on a delivery sell of `value` INR, including the per-scrip DP charge."""
    fees = value * (c.brokerage_pct + c.exchange_txn_pct + c.sebi_fee_pct) / 100
    gst = fees * c.gst_pct / 100
    return fees + gst + value * c.stt_pct / 100 + c.dp_charge_per_sell


def slip(price: float, c, side: str) -> float:
    """Move a modelled fill price against us by the configured slippage."""
    s = c.slippage_pct / 100
    return price * (1 + s) if side == "buy" else price * (1 - s)
