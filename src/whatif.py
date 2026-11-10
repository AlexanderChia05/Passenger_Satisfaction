"""What-if search: cost-aware counterfactuals for one passenger and a
greedy service-investment planner for a group of passengers."""
import numpy as np
import pandas as pd

from . import config as C

# Default cost (RM) of raising a service by one star for ONE passenger.
# Illustrative numbers for the prototype; the airline would supply real ones.
DEFAULT_COSTS = {
    "Inflight wifi service": 6, "Departure/Arrival time convenient": 60, "Ease of Online booking": 2,
    "Gate location": 80, "Food and drink": 12, "Online boarding": 3, "Seat comfort": 45,
    "Inflight entertainment": 10, "On-board service": 15, "Leg room service": 70, "Baggage handling": 8,
    "Checkin service": 5, "Inflight service": 15, "Cleanliness": 7,
}


def counterfactual(predict_fn, X: pd.DataFrame, costs: dict, locked=(), target=0.6, max_steps=10):
    """Greedy search for the cheapest +1-star upgrades that lift P(satisfied) to `target`.

    Each step tries +1 on every unlocked service (< 5 stars) and keeps the one
    with the best probability gain per RM. Returns (steps, final_p) where each
    step = (service, from, to, cost, p_before, p_after).
    """
    cur = X.copy().reset_index(drop=True)
    p_now = float(predict_fn(cur)[0])
    steps = []
    for _ in range(max_steps):
        if p_now >= target:
            break
        cands = [s for s in C.RATING_COLS if s not in locked and int(cur.iloc[0][s]) < 5]
        if not cands:
            break
        trial = pd.concat([cur.assign(**{s: int(cur.iloc[0][s]) + 1}) for s in cands], ignore_index=True)
        gain = predict_fn(trial) - p_now
        score = gain / np.array([max(costs.get(s, 1), 0.01) for s in cands])
        best = int(np.argmax(score))
        if gain[best] <= 0:
            break
        s = cands[best]
        steps.append((s, int(cur.iloc[0][s]), int(cur.iloc[0][s]) + 1, costs.get(s, 0), p_now, p_now + float(gain[best])))
        cur = trial.iloc[[best]].reset_index(drop=True)
        p_now += float(gain[best])
    return steps, p_now


def plan_budget(predict_fn, X: pd.DataFrame, costs: dict, budget: float, locked=(), max_steps=12):
    """Greedy fleet plan: each step raises one service by +1 star for every
    passenger in X below 5 stars, choosing the upgrade that converts the most
    passengers to 'satisfied' per RM, until the budget runs out.

    Returns (steps DataFrame, final probabilities).
    """
    cur = X.copy().reset_index(drop=True)
    p = predict_fn(cur)
    spent, rows = 0.0, []
    for _ in range(max_steps):
        best = None
        for s in C.RATING_COLS:
            if s in locked:
                continue
            mask = cur[s].between(0, 4).to_numpy()
            n = int(mask.sum())
            cost = n * costs.get(s, 0)
            if n == 0 or spent + cost > budget:
                continue
            up = cur.copy()
            up.loc[mask, s] = up.loc[mask, s].clip(lower=0) + 1
            p_new = predict_fn(up)
            converted = int(((p_new >= 0.5) & (p < 0.5)).sum() - ((p_new < 0.5) & (p >= 0.5)).sum())
            value = converted / max(cost, 0.01)
            if converted > 0 and (best is None or value > best[0]):
                best = (value, s, up, p_new, cost, converted, n)
        if best is None:
            break
        _, s, cur, p_new, cost, converted, n = best
        spent += cost
        rows.append({"step": len(rows) + 1, "service": s, "passengers upgraded": n, "cost (RM)": cost,
                     "newly satisfied": converted, "RM per convert": cost / converted,
                     "satisfied rate": float((p_new >= 0.5).mean()), "cumulative spend (RM)": spent})
        p = p_new
    return pd.DataFrame(rows), p
