"""Cluster contact moments into distinct shots."""

SHOT_TYPES = ("smash", "clear", "drop", "drive", "lift")


def cluster_shots(frame_series, fps):
    """
    Group high-velocity / contact-phase frames into shots.
    Returns list of shot dicts with best frame per cluster.
    """
    frames = frame_series.get("frames", [])
    if not frames:
        return []

    contact = [f for f in frames if f.get("phase") == "contact"]
    if not contact:
        sorted_by_vel = sorted(frames, key=lambda x: x.get("wrist_vel", 0), reverse=True)
        contact = sorted_by_vel[: max(1, len(sorted_by_vel) // 8)]

    contact.sort(key=lambda x: x["frame_idx"])
    gap = max(int(fps), 15)
    clusters = []
    current = [contact[0]]

    for f in contact[1:]:
        if f["frame_idx"] - current[-1]["frame_idx"] <= gap:
            current.append(f)
        else:
            clusters.append(current)
            current = [f]
    if current:
        clusters.append(current)

    shots = []
    for cluster in clusters:
        best = max(cluster, key=lambda x: x.get("wrist_vel", 0))
        shots.append({
            "frame_idx": best["frame_idx"],
            "shot_type": best.get("shot_type", "drive"),
            "angles": best.get("angles", {}),
            "stability": best.get("stability", 0),
            "wrist_vel": best.get("wrist_vel", 0),
        })
    return shots


def count_by_type(shots):
    counts = {t: 0 for t in SHOT_TYPES}
    for s in shots:
        st = s.get("shot_type", "drive")
        if st in counts:
            counts[st] += 1
        else:
            counts["drive"] += 1
    return counts
