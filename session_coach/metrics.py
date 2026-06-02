"""Session scores: consistency (vs own session) and stability — no pro reference."""

import statistics


def _angle_vector(angles):
    keys = ("shoulder", "elbow", "wrist", "knee", "ankle")
    return [angles.get(k) for k in keys if angles.get(k) is not None]


def consistency_score(shots):
    """
    Higher when same shot types have similar joint angles across the session.
    0-100 scale.
    """
    if len(shots) < 2:
        return 70.0

    by_type = {}
    for s in shots:
        st = s.get("shot_type", "drive")
        vec = _angle_vector(s.get("angles", {}))
        if len(vec) >= 3:
            by_type.setdefault(st, []).append(vec)

    if not by_type:
        return 50.0

    type_scores = []
    for _st, vectors in by_type.items():
        if len(vectors) < 2:
            type_scores.append(75.0)
            continue
        # Mean std dev across joints (lower = more consistent)
        n_joints = len(vectors[0])
        stds = []
        for j in range(n_joints):
            vals = [v[j] for v in vectors if j < len(v)]
            if len(vals) >= 2:
                stds.append(statistics.pstdev(vals))
        if not stds:
            continue
        avg_std = sum(stds) / len(stds)
        # ~0 std -> 100, ~25+ std -> ~0
        type_scores.append(max(0, min(100, 100 - avg_std * 3.5)))

    return round(sum(type_scores) / len(type_scores), 1) if type_scores else 50.0


def stability_score(shots):
    if not shots:
        return 0.0
    vals = [s.get("stability", 0) for s in shots]
    return round(sum(vals) / len(vals), 1)


def session_score(consistency, stability):
    """Combined 0-100 session index."""
    return round(0.6 * consistency + 0.4 * stability, 1)


def rank_highlights(shots, fps):
    """Best = high stability + high wrist_vel; worst = low stability."""
    if not shots:
        return [], []

    ranked = sorted(
        shots,
        key=lambda s: (s.get("stability", 0) * 0.6 + min(s.get("wrist_vel", 0) * 500, 40)),
        reverse=True,
    )
    n = len(ranked)
    if n == 1:
        return [ranked[0]], []

    top_n = min(4, max(1, n // 3))
    best = ranked[:top_n]
    worst = ranked[-top_n:] if n > top_n else []
    return best, worst


def generate_tips(shots, consistency, stability):
    tips = []
    if consistency < 55:
        tips.append({
            "level": "warning",
            "text": "Your technique varies a lot between similar shots. Pick one shot (e.g. clear) and repeat 20 times focusing on the same arm path.",
        })
    if stability < 60:
        tips.append({
            "level": "critical",
            "text": "Balance looks unstable at contact — widen your base, bend knees, and keep your head over your hips.",
        })

    by_type = {}
    for s in shots:
        by_type.setdefault(s.get("shot_type", "drive"), []).append(s)

    if by_type.get("lift") and stability < 70:
        knees = [s.get("angles", {}).get("knee") for s in by_type["lift"] if s.get("angles", {}).get("knee")]
        if knees and statistics.mean(knees) > 150:
            tips.append({
                "level": "critical",
                "text": f"Lifts: knee angle averages {statistics.mean(knees):.0f}° — bend more and reach lower (aim ~130–140° at contact).",
            })

    counts = {t: sum(1 for s in shots if s.get("shot_type") == t) for t in by_type}
    dominant = max(counts, key=counts.get) if counts else None
    if dominant and counts.get(dominant, 0) / max(len(shots), 1) > 0.5:
        tips.append({
            "level": "info",
            "text": f"This session was mostly {dominant}s — add variety (drops/clears) or drill weaker shot types for balanced improvement.",
        })

    if not tips:
        tips.append({
            "level": "success",
            "text": "Solid session consistency. Keep filming weekly to track your own progress.",
        })
    return tips[:5]


def frame_to_timestamp(frame_idx, fps):
    sec = int(frame_idx / fps) if fps else 0
    return f"{sec // 60}:{sec % 60:02d}"
