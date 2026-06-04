"""
Athlete Vision — Offline Reference Builder
===========================================
Replaces database-driven reference players with offline static benchmarks from ideal_angles.py.
Ensures full backward compatibility with no SQL database queries.
"""

from ai_engine.ideal_angles import IDEAL_BIOMECHANICAL_ANGLES

class DummyReferencePlayer:
    """Mock object mimicking ReferencePlayer model to prevent attribute errors."""
    id = None
    name = "Ideal Standard"
    nationality = "Universal"
    sport = "Badminton"
    hand = "right"
    achievements = "Scientific Biomechanical Standard"
    description = "Form benchmarks optimized for optimal range of motion, power, and injury prevention."
    is_active = True

def seed_reference_from_json(json_path="reference_data/multisense_badminton.json"):
    """Seeding is deprecated as we are now using static in-memory ideal angles."""
    print("[RefBuilder] Seeding skipped: Platform is using offline biomechanical benchmarks.")
    return DummyReferencePlayer()

def get_active_reference_player():
    """Returns a mock pro player to keep comparison templates functioning smoothly."""
    return DummyReferencePlayer()

def get_reference_angles(player_id, shot_type, phase='contact'):
    """Fetch ideal biomechanical angles for a shot phase.
    
    Args:
        player_id: Unused (retained for signature compatibility)
        shot_type: 'smash', 'clear', 'drive', 'drop', 'net'
        phase: 'preparation', 'swing', 'contact', 'follow_through'
        
    Returns:
        dict of joint angles, or None
    """
    shot_type_clean = shot_type.lower() if shot_type else 'drive'
    if shot_type_clean == 'net shot':
        shot_type_clean = 'net'
    # Fallback to drive if shot type not defined in dictionary
    if shot_type_clean not in IDEAL_BIOMECHANICAL_ANGLES:
        shot_type_clean = 'drive'
        
    phases = IDEAL_BIOMECHANICAL_ANGLES.get(shot_type_clean, {})
    return phases.get(phase, None)

def get_all_reference_angles(player_id, shot_type):
    """Fetch ideal biomechanical angles for all phases of a shot.
    
    Args:
        player_id: Unused
        shot_type: 'smash', 'clear', 'drive', 'drop', 'net'
        
    Returns:
        dict of phase -> angles dict
    """
    shot_type_clean = shot_type.lower() if shot_type else 'drive'
    if shot_type_clean == 'net shot':
        shot_type_clean = 'net'
    if shot_type_clean not in IDEAL_BIOMECHANICAL_ANGLES:
        shot_type_clean = 'drive'
        
    return IDEAL_BIOMECHANICAL_ANGLES.get(shot_type_clean, {})

def get_reference_summary(player_id=None):
    """Generates a summary of all target techniques.
    
    Returns:
        dict formatted as shot_type -> phase -> angles/notes
    """
    summary = {}
    for shot_type, phases in IDEAL_BIOMECHANICAL_ANGLES.items():
        summary[shot_type] = {}
        for phase, angles in phases.items():
            summary[shot_type][phase] = {
                'angles': angles,
                'notes': f"Scientific standard for a badminton {shot_type} in {phase} phase."
            }
    return summary

def process_reference_video(video_path, player_id, shot_type_override=None):
    """Processing reference videos is deprecated as the benchmarks are static."""
    print("[RefBuilder] Reference video processing skipped: system is offline.")
    return 0
