"""Strategy 4 used as a FILTER on strategies 1-3.

Wraps any strategy: a new position (from flat) is only allowed on the VWAP-confirming side —
longs need last close > session VWAP, shorts need last close < session VWAP — evaluated when
the target is issued (for ORB stop-entries: when the order is placed at 09:35). Exits and
position reductions always pass. For the SPY noise-area strategy the VWAP condition is already
part of the rules (use_vwap=True/False is the with/without comparison).
"""
from __future__ import annotations

from collections import defaultdict

from .common import SessionVWAP


class VWAPFilter:
    def __init__(self, inner):
        self.inner = inner
        self.name = f"{inner.name}+vwap_filter"
        self.blocked = 0

    def prepare_session(self, session, md):
        self.vwaps = defaultdict(SessionVWAP)
        self.last = {}
        return self.inner.prepare_session(session, md)

    def on_bar(self, now, bars, ctx):
        for s, b in bars.items():
            self.vwaps[s].update(b)
            self.last[s] = b.close
        targets = self.inner.on_bar(now, bars, ctx)
        if not targets:
            return targets
        out = {}
        for s, t in targets.items():
            if t.weight != 0 and ctx.position(s) == 0 and s in self.last:
                v = self.vwaps[s].value
                if (t.weight > 0 and not self.last[s] > v) or (t.weight < 0 and not self.last[s] < v):
                    self.blocked += 1
                    continue
            out[s] = t
        return out or None
