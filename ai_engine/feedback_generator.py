"""
Athlete Vision 2.0 — AI Coach Insight Generator
=========================================
Generates personalized, 6-component AI coaching feedback for the user.
Components: Issue, Fix, Why, Drill, Cue, Checkpoint.

Each joint/direction has MULTIPLE varied responses to avoid repetition.
"""

import random
import hashlib


def _no_reference_feedback(comparison_data):
    """Fallback when no reference data is available."""
    return {
        'issue': 'No reference player configured. Cannot analyze specific form deviations.',
        'fix': 'Ask your administrator to set a Professional Benchmark (like An Se-young).',
        'why': 'AI Coach Insights require a professional benchmark to compare your angles against.',
        'drill': 'Keep practicing your general mechanics and shadow swings.',
        'cue': 'Focus on consistent form.',
        'checkpoint': 'Your baseline mechanics look acceptable.',
        'similarity': 0,
        'grade': 'N/A'
    }

def generate_feedback(comparison_data):
    """Generate comprehensive coaching feedback from comparison results."""
    
    if not comparison_data.get('has_reference'):
        return _no_reference_feedback(comparison_data)
        
    ref_name = comparison_data.get('reference_player', 'An Se-young')
    shot_label = comparison_data.get('shot_type', 'shot').lower()
    similarity = comparison_data.get('similarity_score', 0)
    grade = comparison_data.get('grade', 'N/A')
    weaknesses = comparison_data.get('weaknesses', [])
    stability = comparison_data.get('stability_score', 70)
    
    # Default positive feedback if form is perfect
    if not weaknesses or (weaknesses[0]['deviation'] <= 8 and stability > 85):
        positives = _get_positive_feedback(shot_label, ref_name, similarity)
        positives['similarity'] = similarity
        positives['grade'] = grade
        return positives
        
    # --- MULTI-JOINT COACHING (Increased Feedback) ---
    # We will pick the TOP 2 weaknesses for a dual-focus improvement plan
    active_weaknesses = weaknesses[:2]
    
    combined_issue = []
    combined_fix = []
    combined_why = []
    combined_drill = []
    combined_cue = []
    checkpoint_items = []
    
    for i, weak in enumerate(active_weaknesses):
        joint = weak['joint']
        user_a = weak['user_angle']
        ref_a = weak['ref_angle']
        diff = weak['deviation']
        direction = "higher" if user_a < ref_a else "lower"
        
        base_advice = _get_joint_advice(joint, direction, shot_label, ref_name)
        
        prefix = f"({i+1}) " if len(active_weaknesses) > 1 else ""
        br = " " if i > 0 else ""
        severity = "Significant" if diff > 25 else "Minor"
        
        combined_issue.append(f"{br}{prefix}{severity} {joint} error: {base_advice['issue']}")
        combined_fix.append(f"{br}{prefix}{base_advice['fix']}")
        combined_why.append(base_advice['why'])
        combined_drill.append(f"{br}{prefix}{base_advice['drill']}")
        combined_cue.append(base_advice['cue'])
        checkpoint_items.append(base_advice['checkpoint'])


    # Add Stability Check (Integrated into Issue & Fix to keep 6-part layout)
    if stability < 75:
        combined_issue.append(f" ⚠️ Stability Alert: Your head/balance drifted during the swing.")
        combined_fix.append(" Keep your nose aligned over your leading foot.")
        combined_drill.append(" Shadow swings on one leg to find your center.")

    # Add "Pro Secret" (Integrated into Checkpoint to keep 6-part layout)
    pro_secret_text = ""
    if ref_name == "An Se-young":
        pro_secret_text = f" 💎 Pro Secret from {ref_name}: Focus on 'Split-step' recovery. She stays low to cover the next shot 30% faster."
    else:
        pro_secret_text = f" 💎 Pro Mindset: Great players focus 80% on 'Ready Position' before the shuttle arrives."

    return {
        'issue': "".join(combined_issue), # Join with empty string as items already have space if needed
        'fix': "".join(combined_fix),
        'why': " ".join(list(dict.fromkeys(combined_why))),
        'drill': "".join(combined_drill),
        'cue': " & ".join(combined_cue),
        'checkpoint': " ".join(checkpoint_items) + pro_secret_text,
        'similarity': similarity,
        'grade': grade
    }




def _get_positive_feedback(shot_label, ref_name, similarity):
    """Return varied positive feedback for good form."""
    options = [
        {
            'issue': f"Your {shot_label} form is excellent and closely matches {ref_name}.",
            'fix': "Maintain this precise joint alignment and keep building consistency.",
            'why': "Optimal biomechanics lead to maximum efficiency and power.",
            'drill': f"Do 20 full-speed {shot_label}s to lock in this muscle memory.",
            'cue': "Stay relaxed and explosive.",
            'checkpoint': "Your shots should feel effortless and powerful."
        },
        {
            'issue': f"Outstanding {shot_label} technique! Your angles closely mirror {ref_name}'s professional form.",
            'fix': "Focus on repeating this exact form under match pressure.",
            'why': "Consistency under pressure separates advanced from intermediate players.",
            'drill': f"Play 3 sets of 10 {shot_label}s against a training partner at full intensity.",
            'cue': "Same form, more speed.",
            'checkpoint': "Your shot quality should remain identical even when fatigued."
        },
        {
            'issue': f"Your biomechanics for the {shot_label} are near-pro level. Very impressive!",
            'fix': "Now work on deception — same body position but vary the shuttle direction.",
            'why': f"At your level, deception becomes the key advantage over pure technique.",
            'drill': f"Hit 15 {shot_label}s to the left, then 15 to the right using the exact same body preparation.",
            'cue': "Hide your intention.",
            'checkpoint': "Your opponent should not be able to read your shot direction until after contact."
        },
    ]
    return random.choice(options)


def _get_joint_advice(joint, direction, shot_label, ref_name):
    """Knowledge base with MULTIPLE varied responses per joint/direction."""
    
    advice_pool = _ADVICE_DATABASE.get(joint, {}).get(direction, [])
    
    if not advice_pool:
        # Generic fallback
        return {
            'issue': f"Your {joint} is misaligned during the {shot_label}.",
            'fix': f"Adjust your {joint} to match {ref_name}'s form.",
            'why': f"Proper {joint} alignment improves shot quality.",
            'drill': f"Practice 10 {shot_label} shadow swings slowly.",
            'cue': f"Watch the {joint}.",
            'checkpoint': f"Your {joint} should feel different during contact."
        }
    
    # Pick a random advice from the pool
    advice_template = random.choice(advice_pool)
    
    # Substitute placeholders
    result = {}
    for key, val in advice_template.items():
        result[key] = val.format(shot=shot_label, ref=ref_name, joint=joint)
    
    return result


# ═══════════════════════════════════════════════════════
# FULL ADVICE DATABASE — Multiple variants per joint
# ═══════════════════════════════════════════════════════

_ADVICE_DATABASE = {
    'shoulder': {
        'higher': [
            {
                'issue': "Shoulder is too low when hitting.",
                'fix': "Reach higher! Let your shoulder go up to meet the shuttle like {ref}.",
                'why': "Higher contact gives you steeper, harder shots to return.",
                'drill': "Do 10 slow shadow swings. Stop at the top to check your stretch.",
                'cue': "Reach for the ceiling.",
                'checkpoint': "Arm fully straight upwards, shoulder near your ear."
            },
            {
                'issue': "Not reaching high enough on your {shot}.",
                'fix': "Pull your elbow up first, then fully straighten your arm.",
                'why': "Hitting higher means more downward power.",
                'drill': "Shadow swing and try to touch the highest mark on a wall.",
                'cue': "Elbow leads, reach high.",
                'checkpoint': "Feel a stretch in your side when you hit."
            },
            {
                'issue': "Short shoulder rotation.",
                'fix': "Turn your body fully so your other shoulder points to the net before swinging.",
                'why': "Full body turn gives you much more power than just using your arm.",
                'drill': "Practice throwing a tennis ball with a big shoulder turn.",
                'cue': "Turn big, then swing.",
                'checkpoint': "Chest faces sideways before you swing, then forward after."
            }
        ],
        'lower': [
            {
                'issue': "Shoulder opening up too early.",
                'fix': "Wait longer! Keep your body sideways until your elbow comes forward.",
                'why': "Turning too early loses all your body power.",
                'drill': "Do 10 slow swings. Only let the shoulder turn at the very end.",
                'cue': "Wait... then turn!",
                'checkpoint': "The hit should feel much heavier and faster."
            },
            {
                'issue': "Swinging too hard with your whole body.",
                'fix': "Relax your shoulder. Let your forearm do more of the fast work.",
                'why': "Over-swinging slows down your recovery for the next shot.",
                'drill': "Hit 15 flat shots without moving your feet at all.",
                'cue': "Keep it compact.",
                'checkpoint': "You step back to center instantly after hitting."
            }
        ]
    },
    'elbow': {
        'higher': [
            {
                'issue': "Elbow is too bent during the hit.",
                'fix': "Straighten your arm sharply exactly when you hit the shuttle.",
                'why': "A straight arm whips the racket much faster for big power.",
                'drill': "Throw a racket cover slowly, snapping your arm straight to release it.",
                'cue': "Snap it straight.",
                'checkpoint': "Hear a louder swoosh sound from your racket."
            },
            {
                'issue': "Hitting with a cramped 'chicken wing' arm.",
                'fix': "Imagine pulling an apple from a very high tree branch.",
                'why': "A bent arm loses your height and cuts your power in half.",
                'drill': "Shadow swing while trying to touch a high target.",
                'cue': "Reach long and high.",
                'checkpoint': "Elbow locks straight just as you hit."
            }
        ],
        'lower': [
            {
                'issue': "Arm is too straight and stiff before you swing.",
                'fix': "Keep a small bend in your elbow while raising your racket.",
                'why': "A slightly bent arm acts like a whip instead of a stiff stick.",
                'drill': "Do loose shadow swings holding just a shuttlecock.",
                'cue': "Loose arm whip.",
                'checkpoint': "Elbow feels relaxed right before the swing."
            },
            {
                'issue': "Locking out your elbow too hard.",
                'fix': "Never fully lock your joint forcefully. Keep a tiny bend.",
                'why': "Hard locking risks injury and makes the shot jerky.",
                'drill': "Use a light resistance band to practice soft arm extensions.",
                'cue': "Soft snap, not locked.",
                'checkpoint': "No pain or shocking feeling in the elbow gap."
            }
        ]
    },
    'wrist': {
        'higher': [
            {
                'issue': "Not enough wrist flick.",
                'fix': "Pull your wrist back early, then snap it hard right as you hit.",
                'why': "The final wrist flick is the secret to ultimate smash speed.",
                'drill': "Sit down and flick 30 shuttles over the net just using your wrist.",
                'cue': "Violent wrist snap.",
                'checkpoint': "Sharp 'pop' sound from the shuttle."
            },
            {
                'issue': "Wrist stays flat and lazy.",
                'fix': "Think of flicking water off your fingers at the shuttle.",
                'why': "Without the flick, you only push. You need pure snap.",
                'drill': "Hold the racket very high on the handle to focus purely on wrist.",
                'cue': "Flick the water.",
                'checkpoint': "Racket head zooms past your hand at the end."
            }
        ],
        'lower': [
            {
                'issue': "Floppy wrist during contact.",
                'fix': "Hold the wrist firm and slightly back until impact.",
                'why': "A floppy wrist makes your shots fly randomly.",
                'drill': "Hit 20 dropshots focusing only on a nice firm wrist.",
                'cue': "Firm and guide.",
                'checkpoint': "Racket face stays incredibly steady."
            },
            {
                'issue': "Releasing wrist snap too early.",
                'fix': "Hold the wrist squeeze until your elbow is totally forward.",
                'why': "Snapping early wastes your power on empty air.",
                'drill': "Do 15 slow swings. Say 'Elbow... Wrist!' out loud.",
                'cue': "Hold... NOW flick.",
                'checkpoint': "Max racket speed hits exactly at the shuttle."
            }
        ]
    },
    'knee': {
        'higher': [
            {
                'issue': "Legs are too stiff.",
                'fix': "Bend your knees deeper before you swing. Load up your power.",
                'why': "Power starts from the ground. Stiff legs equal weak shots.",
                'drill': "Do jump squats, then immediately do a shadow swing.",
                'cue': "Squat and explode.",
                'checkpoint': "Feel the push start in your legs and move to your hand."
            },
            {
                'issue': "Flat-footed hitting.",
                'fix': "Stay bouncy on your toes with soft knees at all times.",
                'why': "Locked knees trap your energy. Soft knees create spring.",
                'drill': "Jump rope briefly, then hit 10 shots with that same bounce.",
                'cue': "Be a spring.",
                'checkpoint': "Leave the ground just a bit during hard shots."
            }
        ],
        'lower': [
            {
                'issue': "Sinking too low in your stance.",
                'fix': "Bend knees, but keep hips high. Don't squat too deep.",
                'why': "Going too low traps you and makes you slow to recover.",
                'drill': "10 rapid shadow swings checking how fast you bounce back.",
                'cue': "Stay light, stay high.",
                'checkpoint': "Instant bounce-back to center after the hit."
            },
            {
                'issue': "Sitting into the shot instead of springing.",
                'fix': "Use a quick, shallow knee dip instead of a slow deep squat.",
                'why': "A fast small dip gives more bounce than a slow heavy one.",
                'drill': "Do 20 tiny, super-fast hops before swinging.",
                'cue': "Quick dip, fast up.",
                'checkpoint': "Knee bend and push finishes in half a second."
            }
        ]
    },
    'ankle': {
        'higher': [
            {
                'issue': "Standing flat-footed.",
                'fix': "Keep heels off the floor. Stay bouncy on the balls of your feet.",
                'why': "Flat feet kill your reaction time completely.",
                'drill': "30 seconds of pure split-stepping with heels up.",
                'cue': "Heels up, spring loaded.",
                'checkpoint': "Constant energy and tension in your calves."
            },
            {
                'issue': "Stiff, stuck ankles.",
                'fix': "Picture standing on hot coals. Keep feet super light.",
                'why': "Light feet let you start moving a fraction of a second faster.",
                'drill': "Do 20 calf raises between sets of shots.",
                'cue': "Light feet, hot coals.",
                'checkpoint': "Dancing on your toes lightly between points."
            }
        ],
        'lower': [
            {
                'issue': "Landing on your toes during a lunge.",
                'fix': "Launch forward but ALWAYS land heel-first on your big lunge step.",
                'why': "Landing on toes wrecks your knees and ruins your balance completely.",
                'drill': "20 slow lunges, exaggerating that heel-strike landing.",
                'cue': "Heel strikes first.",
                'checkpoint': "Front foot fully stops your momentum strongly and safely."
            },
            {
                'issue': "Ankle wobbling inward.",
                'fix': "Keep your foot pointing straight at the shuttle. Hold it stable.",
                'why': "Wobbly ankles cause sprains very easily in badminton.",
                'drill': "Lunge while focusing totally on keeping the foot perfectly straight.",
                'cue': "Foot straight and solid.",
                'checkpoint': "Zero wobble when your weight hits that front foot."
            }
        ]
    }
}


def format_feedback_for_display(feedback_data):
    """
    Format feedback into a strict 6-part pipe-separated string for the template.
    Format:
    ISSUE|FIX|WHY|DRILL|CUE|CHECKPOINT
    """
    if not feedback_data:
        # Fallback format
        return "❌ Issue: Analysis failed|🎯 Fix: Try re-uploading the video|💡 Why: System error|🏋️ Drill: N/A|🧠 Cue: N/A|✅ Checkpoint: N/A"
        
    issue = feedback_data.get('issue', 'N/A')
    fix = feedback_data.get('fix', 'N/A')
    why = feedback_data.get('why', 'N/A')
    drill = feedback_data.get('drill', 'N/A')
    cue = feedback_data.get('cue', 'N/A')
    checkpoint = feedback_data.get('checkpoint', 'N/A')
    
    # Construct exact format requested
    return f"❌ Issue: {issue}|🎯 Fix: {fix}|💡 Why: {why}|🏋️ Drill: {drill}|🧠 Cue: {cue}|✅ Checkpoint: {checkpoint}"
