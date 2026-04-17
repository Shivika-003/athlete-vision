"""
Athlete Vision 2.0 — Comparison Engine
========================================
Compares user's joint angles against reference player data.
Produces MAE, MSE, similarity scores, per-joint breakdown,
and weakness rankings.
"""

import json
from ai_engine.angle_utils import (
    compute_mae, compute_mse, compute_similarity,
    compute_per_joint_similarity, rank_weaknesses
)
from ai_engine.reference_builder import (
    get_active_reference_player, get_reference_angles, get_all_reference_angles
)


def compare_user_with_reference(user_results):
    """Full comparison of user analysis results with the active reference player.
    
    This is the main entry point called after process_video() completes.
    
    Args:
        user_results: dict from pose_analyzer.process_video()
        
    Returns:
        dict with comparison data:
        {
            'reference_player': 'An Se-young',
            'shot_type': 'smash',
            'similarity_score': 85.2,
            'grade': 'B+',
            'mae': 8.3,
            'mse': 92.1,
            'user_angles': {...},
            'ref_angles': {...},
            'per_joint_similarity': {'shoulder': 90.1, ...},
            'weaknesses': [('wrist', 145, 170, 25.0), ...],
            'phase_comparison': {
                'contact': {'user': {...}, 'ref': {...}, 'similarity': 85.2},
                ...
            }
        }
    """
    ref_player = get_active_reference_player()
    if not ref_player:
        return _no_reference_result(user_results)
    
    shot_type = user_results.get('shot_type', 'drive')
    user_contact_angles = user_results.get('contact_angles', {})
    user_phase_angles = user_results.get('phase_angles', {})
    
    if not user_contact_angles:
        return _no_reference_result(user_results)
    
    # Get reference angles for the detected shot type (contact phase)
    ref_contact_angles = get_reference_angles(ref_player.id, shot_type, 'contact')
    
    if not ref_contact_angles:
        # Try 'drive' as universal fallback
        ref_contact_angles = get_reference_angles(ref_player.id, 'drive', 'contact')
    
    if not ref_contact_angles:
        return _no_reference_result(user_results)
    
    # ─── Compute comparison metrics ───
    mae = compute_mae(user_contact_angles, ref_contact_angles)
    mse = compute_mse(user_contact_angles, ref_contact_angles)
    similarity = compute_similarity(user_contact_angles, ref_contact_angles)
    per_joint = compute_per_joint_similarity(user_contact_angles, ref_contact_angles)
    weaknesses = rank_weaknesses(user_contact_angles, ref_contact_angles)
    grade = _score_to_grade(similarity)
    
    # ─── Phase-by-phase comparison ───
    ref_all_phases = get_all_reference_angles(ref_player.id, shot_type)
    phase_comparison = {}
    
    for phase_name in ['preparation', 'swing', 'contact', 'follow_through']:
        user_ph = user_phase_angles.get(phase_name, {})
        ref_ph = ref_all_phases.get(phase_name, {})
        
        if user_ph and ref_ph:
            ph_sim = compute_similarity(user_ph, ref_ph)
            ph_per_joint = compute_per_joint_similarity(user_ph, ref_ph)
            phase_comparison[phase_name] = {
                'user': user_ph,
                'ref': ref_ph,
                'similarity': ph_sim,
                'per_joint': ph_per_joint
            }
    
    # ─── Per-joint angle differences ───
    joint_diffs = {}
    for joint in ['shoulder', 'elbow', 'wrist', 'knee', 'ankle']:
        u = user_contact_angles.get(joint)
        r = ref_contact_angles.get(joint)
        if u is not None and r is not None:
            joint_diffs[joint] = {
                'user': round(u, 1),
                'reference': round(r, 1),
                'difference': round(u - r, 1),
                'abs_difference': round(abs(u - r), 1),
                'similarity': per_joint.get(joint, 0)
            }
    
    return {
        'reference_player': ref_player.name,
        'reference_player_id': ref_player.id,
        'shot_type': shot_type,
        'similarity_score': similarity,
        'grade': grade,
        'mae': mae,
        'mse': mse,
        'user_angles': user_contact_angles,
        'ref_angles': ref_contact_angles,
        'per_joint_similarity': per_joint,
        'joint_diffs': joint_diffs,
        'weaknesses': [
            {'joint': w[0], 'user_angle': w[1], 'ref_angle': w[2], 'deviation': w[3]}
            for w in weaknesses
        ],
        'phase_comparison': phase_comparison,
        'has_reference': True,
    }


def _no_reference_result(user_results):
    """Return comparison result when no reference data is available."""
    return {
        'reference_player': None,
        'reference_player_id': None,
        'shot_type': user_results.get('shot_type', 'drive'),
        'similarity_score': 0,
        'grade': 'N/A',
        'mae': 0,
        'mse': 0,
        'user_angles': user_results.get('contact_angles', {}),
        'ref_angles': {},
        'per_joint_similarity': {},
        'joint_diffs': {},
        'weaknesses': [],
        'phase_comparison': {},
        'has_reference': False,
    }


def _score_to_grade(similarity):
    """Convert similarity percentage to a letter grade.
    
    Args:
        similarity: 0-100 percentage
        
    Returns:
        Grade string (A+, A, B+, B, C+, C, D, F)
    """
    if similarity >= 95:
        return 'A+'
    elif similarity >= 90:
        return 'A'
    elif similarity >= 85:
        return 'B+'
    elif similarity >= 78:
        return 'B'
    elif similarity >= 70:
        return 'C+'
    elif similarity >= 60:
        return 'C'
    elif similarity >= 50:
        return 'D'
    else:
        return 'F'


def serialize_comparison(comparison_data):
    """Serialize comparison data to JSON string for database storage.
    
    Args:
        comparison_data: dict from compare_user_with_reference()
        
    Returns:
        JSON string safe for database storage
    """
    # Create a serializable copy (remove non-JSON-safe types)
    safe_data = {
        'reference_player': comparison_data.get('reference_player'),
        'shot_type': comparison_data.get('shot_type'),
        'similarity_score': comparison_data.get('similarity_score'),
        'grade': comparison_data.get('grade'),
        'mae': comparison_data.get('mae'),
        'mse': comparison_data.get('mse'),
        'user_angles': comparison_data.get('user_angles'),
        'ref_angles': comparison_data.get('ref_angles'),
        'per_joint_similarity': comparison_data.get('per_joint_similarity'),
        'joint_diffs': comparison_data.get('joint_diffs'),
        'weaknesses': comparison_data.get('weaknesses'),
        'has_reference': comparison_data.get('has_reference'),
    }
    return json.dumps(safe_data)


def deserialize_comparison(json_string):
    """Deserialize comparison data from database JSON string.
    
    Args:
        json_string: JSON string from VideoRecord.comparison_details
        
    Returns:
        dict with comparison data, or empty dict if invalid
    """
    if not json_string:
        return {}
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError):
        return {}
