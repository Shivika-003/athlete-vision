"""
Athlete Vision — Ideal Biomechanical Angle Benchmarks
======================================================
Static, scientifically benchmarked target angles for each badminton shot phase.
"""

IDEAL_BIOMECHANICAL_ANGLES = {
    "smash": {
        "preparation": {"shoulder": 95.0, "elbow": 85.0, "wrist": 145.0, "knee": 135.0, "ankle": 100.0},
        "swing": {"shoulder": 155.0, "elbow": 120.0, "wrist": 160.0, "knee": 115.0, "ankle": 95.0},
        "contact": {"shoulder": 170.0, "elbow": 165.0, "wrist": 170.0, "knee": 145.0, "ankle": 110.0},
        "follow_through": {"shoulder": 120.0, "elbow": 90.0, "wrist": 130.0, "knee": 140.0, "ankle": 105.0}
    },
    "clear": {
        "preparation": {"shoulder": 90.0, "elbow": 80.0, "wrist": 140.0, "knee": 140.0, "ankle": 100.0},
        "swing": {"shoulder": 150.0, "elbow": 115.0, "wrist": 155.0, "knee": 125.0, "ankle": 95.0},
        "contact": {"shoulder": 160.0, "elbow": 155.0, "wrist": 165.0, "knee": 150.0, "ankle": 108.0},
        "follow_through": {"shoulder": 115.0, "elbow": 95.0, "wrist": 135.0, "knee": 145.0, "ankle": 105.0}
    },
    "drive": {
        "preparation": {"shoulder": 75.0, "elbow": 90.0, "wrist": 150.0, "knee": 145.0, "ankle": 100.0},
        "swing": {"shoulder": 110.0, "elbow": 130.0, "wrist": 155.0, "knee": 140.0, "ankle": 98.0},
        "contact": {"shoulder": 125.0, "elbow": 150.0, "wrist": 160.0, "knee": 150.0, "ankle": 105.0},
        "follow_through": {"shoulder": 100.0, "elbow": 110.0, "wrist": 140.0, "knee": 148.0, "ankle": 103.0}
    },
    "drop": {
        "preparation": {"shoulder": 90.0, "elbow": 82.0, "wrist": 142.0, "knee": 138.0, "ankle": 100.0},
        "swing": {"shoulder": 148.0, "elbow": 112.0, "wrist": 150.0, "knee": 130.0, "ankle": 96.0},
        "contact": {"shoulder": 155.0, "elbow": 145.0, "wrist": 140.0, "knee": 148.0, "ankle": 106.0},
        "follow_through": {"shoulder": 110.0, "elbow": 100.0, "wrist": 125.0, "knee": 145.0, "ankle": 104.0}
    },
    "net": {
        "preparation": {"shoulder": 60.0, "elbow": 100.0, "wrist": 145.0, "knee": 120.0, "ankle": 85.0},
        "swing": {"shoulder": 75.0, "elbow": 115.0, "wrist": 150.0, "knee": 105.0, "ankle": 80.0},
        "contact": {"shoulder": 85.0, "elbow": 130.0, "wrist": 155.0, "knee": 95.0, "ankle": 75.0},
        "follow_through": {"shoulder": 70.0, "elbow": 110.0, "wrist": 140.0, "knee": 110.0, "ankle": 82.0}
    },
    "lift": {
        "preparation": {"shoulder": 50.0, "elbow": 90.0, "wrist": 140.0, "knee": 110.0, "ankle": 90.0},
        "swing": {"shoulder": 80.0, "elbow": 120.0, "wrist": 150.0, "knee": 95.0, "ankle": 80.0},
        "contact": {"shoulder": 95.0, "elbow": 140.0, "wrist": 160.0, "knee": 90.0, "ankle": 75.0},
        "follow_through": {"shoulder": 120.0, "elbow": 110.0, "wrist": 145.0, "knee": 120.0, "ankle": 92.0}
    }
}
