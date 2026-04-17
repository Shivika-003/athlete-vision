"""
Athlete Vision 2.0 — Reference Builder
========================================
Processes reference player videos to extract biomechanical angle data,
or seeds the database from the pre-built JSON reference file.
"""

import json
import os
from models import db, ReferencePlayer, ReferenceShotData


def seed_reference_from_json(json_path="reference_data/an_seyoung.json"):
    """Load pre-built reference data from JSON and seed into the database.
    
    This is called on first app startup to populate the reference database
    with An Se-young's biomechanical data.
    
    Args:
        json_path: Path to the reference JSON file
        
    Returns:
        ReferencePlayer object, or None if file not found
    """
    if not os.path.exists(json_path):
        print(f"[RefBuilder] Reference file not found: {json_path}")
        return None
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    player_info = data['player']
    
    # Check if already seeded
    existing = ReferencePlayer.query.filter_by(name=player_info['name']).first()
    if existing:
        print(f"[RefBuilder] Reference player '{player_info['name']}' already exists (id={existing.id})")
        return existing
    
    # Create reference player
    player = ReferencePlayer(
        name=player_info['name'],
        nationality=player_info.get('nationality', ''),
        sport=player_info.get('sport', 'Badminton'),
        hand=player_info.get('hand', 'right'),
        achievements=player_info.get('achievements', ''),
        description=player_info.get('description', ''),
        is_active=True
    )
    db.session.add(player)
    db.session.flush()  # Get player.id before adding shots
    
    # Add all shot data
    shots_data = data['shots']
    count = 0
    for shot_type, phases in shots_data.items():
        for phase, angles in phases.items():
            shot = ReferenceShotData(
                player_id=player.id,
                shot_type=shot_type,
                phase=phase,
                shoulder_angle=angles['shoulder'],
                elbow_angle=angles['elbow'],
                wrist_angle=angles['wrist'],
                knee_angle=angles['knee'],
                ankle_angle=angles['ankle'],
                video_source="pre-built reference data",
                notes=angles.get('notes', '')
            )
            db.session.add(shot)
            count += 1
    
    db.session.commit()
    print(f"[RefBuilder] Seeded {count} reference shots for '{player_info['name']}'")
    return player


def get_active_reference_player():
    """Get the currently active reference player.
    
    Returns:
        ReferencePlayer object or None
    """
    return ReferencePlayer.query.filter_by(is_active=True).first()


def get_reference_angles(player_id, shot_type, phase='contact'):
    """Get reference angles for a specific shot type and phase.
    
    Args:
        player_id: ReferencePlayer ID
        shot_type: 'smash', 'clear', 'drive', 'drop', 'net'
        phase: 'preparation', 'swing', 'contact', 'follow_through'
        
    Returns:
        dict with angle values, or None if not found
    """
    shot = ReferenceShotData.query.filter_by(
        player_id=player_id,
        shot_type=shot_type,
        phase=phase
    ).first()
    
    if shot:
        return shot.to_angles_dict()
    return None


def get_all_reference_angles(player_id, shot_type):
    """Get reference angles for all phases of a specific shot type.
    
    Args:
        player_id: ReferencePlayer ID
        shot_type: 'smash', 'clear', 'drive', 'drop', 'net'
        
    Returns:
        dict of phase -> angles dict
    """
    shots = ReferenceShotData.query.filter_by(
        player_id=player_id,
        shot_type=shot_type
    ).all()
    
    result = {}
    for shot in shots:
        result[shot.phase] = shot.to_angles_dict()
    
    return result


def get_reference_summary(player_id):
    """Get a summary of all reference data for a player.
    
    Returns:
        dict with shot_type -> phase -> angles
    """
    shots = ReferenceShotData.query.filter_by(player_id=player_id).all()
    
    summary = {}
    for shot in shots:
        if shot.shot_type not in summary:
            summary[shot.shot_type] = {}
        summary[shot.shot_type][shot.phase] = {
            'angles': shot.to_angles_dict(),
            'notes': shot.notes
        }
    
    return summary


def process_reference_video(video_path, player_id, shot_type_override=None):
    """Process a reference video to extract angle data.
    
    Uses the same pose analysis pipeline but saves results as reference data
    instead of user analysis.
    
    Args:
        video_path: Path to the reference video file
        player_id: ReferencePlayer ID to store data under
        shot_type_override: Force a specific shot type instead of auto-detect
        
    Returns:
        Number of shot data entries created
    """
    from ai_engine.pose_analyzer import process_video
    
    # Run through the standard pipeline
    results = process_video(video_path, "ref_temp", "processed")
    
    if not results.get('contact_angles'):
        return 0
    
    shot_type = shot_type_override or results.get('shot_type', 'drive')
    phase_angles = results.get('phase_angles', {})
    
    count = 0
    
    # Save contact frame as primary reference
    contact_angles = results['contact_angles']
    if contact_angles:
        shot = ReferenceShotData(
            player_id=player_id,
            shot_type=shot_type,
            phase='contact',
            shoulder_angle=contact_angles.get('shoulder', 0),
            elbow_angle=contact_angles.get('elbow', 0),
            wrist_angle=contact_angles.get('wrist', 0),
            knee_angle=contact_angles.get('knee', 0),
            ankle_angle=contact_angles.get('ankle', 0),
            video_source=os.path.basename(video_path),
            visibility_avg=0.0,
            notes=f"Extracted from video: {os.path.basename(video_path)}"
        )
        db.session.add(shot)
        count += 1
    
    # Save other phase angles
    for phase_name, angles in phase_angles.items():
        if phase_name == 'contact':
            continue  # Already saved above
        if angles:
            shot = ReferenceShotData(
                player_id=player_id,
                shot_type=shot_type,
                phase=phase_name,
                shoulder_angle=angles.get('shoulder', 0),
                elbow_angle=angles.get('elbow', 0),
                wrist_angle=angles.get('wrist', 0),
                knee_angle=angles.get('knee', 0),
                ankle_angle=angles.get('ankle', 0),
                video_source=os.path.basename(video_path),
                notes=f"Extracted from video: {os.path.basename(video_path)}"
            )
            db.session.add(shot)
            count += 1
    
    db.session.commit()
    return count
