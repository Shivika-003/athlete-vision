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
    
    # Default positive feedback if form is perfect
    if not weaknesses or weaknesses[0]['deviation'] <= 5:
        positives = _get_positive_feedback(shot_label, ref_name, similarity)
        positives['similarity'] = similarity
        positives['grade'] = grade
        return positives
        
    # Get the single worst joint deviation
    worst = weaknesses[0]
    joint = worst['joint']
    user_a = worst['user_angle']
    ref_a = worst['ref_angle']
    diff = worst['deviation']
    
    direction = "higher" if user_a < ref_a else "lower"
    
    # Use a seed based on angles to get variety but consistency for same analysis
    seed = int(abs(user_a * 100 + ref_a * 10 + diff))
    
    # Retrieve contextual feedback blueprint (randomly chosen from pool)
    base_advice = _get_joint_advice(joint, direction, shot_label, ref_name, seed)
    
    # Scale feedback text based on severity (diff > 25 is major)
    severity_prefix = "Major issue:" if diff > 25 else "Slight deviation:"
    
    issue_text = f"{severity_prefix} Your {joint} angle is {diff:.0f}° {direction} than {ref_name}'s ({user_a:.0f}° vs {ref_a:.0f}°). {base_advice['issue']}"
    fix_text = base_advice['fix']
    why_text = base_advice['why']
    drill_text = base_advice['drill']
    cue_text = base_advice['cue']
    checkpoint_text = base_advice['checkpoint']

    return {
        'issue': issue_text,
        'fix': fix_text,
        'why': why_text,
        'drill': drill_text,
        'cue': cue_text,
        'checkpoint': checkpoint_text,
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


def _get_joint_advice(joint, direction, shot_label, ref_name, seed):
    """Knowledge base with MULTIPLE varied responses per joint/direction."""
    
    rng = random.Random(seed)
    
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
    advice_template = rng.choice(advice_pool)
    
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
                'issue': "Your shoulder stays too low at contact, causing your {shot} to lose height and reach.",
                'fix': "Reach higher and let your hitting shoulder rotate upward to meet the shuttle like {ref}.",
                'why': "Extending the shoulder fully ensures a higher contact point, giving you steeper angles and more margin over the net.",
                'drill': "Do 10 slow {shot} shadow swings, pausing exactly at the highest point to feel the stretch in your shoulder.",
                'cue': "Reach for the ceiling.",
                'checkpoint': "Your shoulder should be fully extended upwards, almost touching your ear at contact."
            },
            {
                'issue': "Your hitting arm isn't reaching high enough, costing you contact height on the {shot}.",
                'fix': "Think of pulling your elbow up first, then extending the forearm. Your shoulder follows the elbow.",
                'why': "A higher contact point means you can hit downward with more angle, making your {shot} harder to return.",
                'drill': "Stand against a wall and mark the highest point your racket can reach. Now try to beat that mark 20 times.",
                'cue': "Elbow leads, shoulder follows.",
                'checkpoint': "You should feel a deep stretch through your lat muscle on the hitting side."
            },
            {
                'issue': "Your shoulder rotation is incomplete — you're punching the shuttle instead of throwing at it.",
                'fix': "Rotate your torso fully so your non-hitting shoulder points toward the net before swinging.",
                'why': "Full trunk rotation generates 40% more power than arm-only swings. {ref} uses this extensively.",
                'drill': "Practice 15 overhead throws with a tennis ball, exaggerating the shoulder turn before each throw.",
                'cue': "Turn fully, then uncoil.",
                'checkpoint': "Your chest should face sideways before the swing and fully forward after follow-through."
            },
            {
                'issue': "You're not engaging your shoulder girdle enough during the {shot} preparation.",
                'fix': "Draw your racket arm back further during windup. Let your shoulder blade pinch toward your spine.",
                'why': "Pre-stretching the shoulder muscles stores elastic energy that gets released explosively at contact.",
                'drill': "Hold the racket behind your head in trophy position for 5 seconds, then snap 10 {shot}s from there.",
                'cue': "Load the slingshot.",
                'checkpoint': "You should feel tension between your shoulder blades just before swinging."
            },
        ],
        'lower': [
            {
                'issue': "Your shoulder is over-rotated or too stiff during the {shot}.",
                'fix': "Relax your shoulder and keep your chest slightly more square to the shuttle before contact, matching {ref}.",
                'why': "Over-rotating strips power from your swing and slows down your recovery time for the next shot.",
                'drill': "Hit 15 flat drives against a wall, focusing exclusively on a compact, half-swing shoulder rotation.",
                'cue': "Keep it compact.",
                'checkpoint': "You should feel less strain in your shoulder joint after hitting."
            },
            {
                'issue': "Your shoulder is opening up too early, leaking power before contact.",
                'fix': "Delay your shoulder rotation — keep it coiled until your elbow starts coming forward.",
                'why': "Early rotation means the kinetic chain breaks. The power from your legs and core gets wasted.",
                'drill': "Do 10 {shot}s in slow motion, counting '1-2-3' where 3 is when the shoulder finally opens.",
                'cue': "Late rotation, big power.",
                'checkpoint': "The shuttle should feel heavier and faster off your racket face."
            },
            {
                'issue': "You're over-committing your upper body into the {shot}, pulling you off balance.",
                'fix': "Keep 60% of the rotation in your forearm and wrist, not your entire trunk.",
                'why': "Over-rotation makes recovery to center court slower, leaving you vulnerable to counter-attacks.",
                'drill': "Practice 20 half-swing {shot}s from a stationary position, keeping your feet completely planted.",
                'cue': "Compact upper body.",
                'checkpoint': "You should be able to take one step back to center immediately after hitting."
            },
        ],
    },
    
    'elbow': {
        'higher': [
            {
                'issue': "Your elbow is bent too much when you hit the {shot}, cramping your swing.",
                'fix': "Straighten your arm out more as you swing upward. Snap the elbow forward exactly at contact like {ref}.",
                'why': "A straighter arm provides a longer lever, which mathematically creates more racket head speed and power.",
                'drill': "Hold a racket cover and do 20 overhead throws, focusing entirely on snapping the elbow straight at the release point.",
                'cue': "Snap it straight.",
                'checkpoint': "You should hear a louder 'whip' sound from your racket strings cutting the air."
            },
            {
                'issue': "You're hitting with a chicken wing elbow — bent and cramped during the {shot}.",
                'fix': "Imagine reaching up to grab something from a high shelf. That's how extended your arm should be at contact.",
                'why': "A bent elbow at contact reduces your reach by 15-20cm and cuts your power by nearly half.",
                'drill': "Tape a string to a high point on the wall. Practice touching it with your racket tip 20 times during shadow swings.",
                'cue': "Reach high, arm long.",
                'checkpoint': "Your elbow should be nearly locked at the exact moment of shuttle contact."
            },
            {
                'issue': "Your arm is collapsing inward during the {shot} contact — the elbow drops too early.",
                'fix': "Keep your elbow high and drive it forward before snapping the forearm. Think javelin, not punch.",
                'why': "{ref} keeps the elbow at shoulder height until the last moment, creating a whip-crack acceleration.",
                'drill': "Do 15 {shot}s with your eyes closed, focusing on the feeling of a high elbow throughout the swing.",
                'cue': "High elbow, late snap.",
                'checkpoint': "Someone watching from the side should see your elbow stay above your shoulder during the swing."
            },
            {
                'issue': "Not enough elbow extension means your {shot} lacks penetration and depth.",
                'fix': "After contact, your arm should fully straighten and point toward your target for a split second.",
                'why': "Full extension through the ball is what separates club players from competitive athletes.",
                'drill': "Hit 20 {shot}s and freeze your arm position after each one. Check if it's fully extended.",
                'cue': "Extend through the shuttle.",
                'checkpoint': "Your follow-through should end with a straight arm pointing in the direction of your shot."
            },
        ],
        'lower': [
            {
                'issue': "Your arm is entirely rigid and locked out during the {shot} preparation.",
                'fix': "Maintain a slight bend in your elbow while drawing the racket back. Let the arm act like a whip.",
                'why': "A locked elbow prevents the elastic 'whiplash' effect, forcing you to use only brute muscle rather than technique.",
                'drill': "Do 15 relaxed shadow swings holding just a shuttlecock, throwing it over the net with a relaxed, bent throwing arm.",
                'cue': "Loose arm, whip it.",
                'checkpoint': "Your elbow should feel relaxed and elastic during the preparation phase."
            },
            {
                'issue': "Your arm is too straight during the wind-up, making your {shot} stiff and telegraphed.",
                'fix': "Bend your elbow to roughly 90° during preparation, then explode into extension at contact.",
                'why': "A bent-to-straight motion creates elastic energy. A straight-to-straight motion is just pushing.",
                'drill': "Practice the 'L-shape' drill: start with your arm in an L, then snap 15 {shot}s from that position.",
                'cue': "Bend first, then explode.",
                'checkpoint': "You should feel a clear two-phase motion: load (bend) then release (straighten)."
            },
            {
                'issue': "Your elbow is hyper-extending which risks injury and reduces control on the {shot}.",
                'fix': "Keep a micro-bend in the elbow even at full extension. Never fully lock the joint.",
                'why': "Hyper-extension puts extreme stress on the elbow ligaments and causes string tension errors.",
                'drill': "Wrap a light resistance band around your elbow and hit 20 {shot}s. It will teach you to stop at natural extension.",
                'cue': "Soft lock, not hard lock.",
                'checkpoint': "You should feel zero joint pain in your elbow after a session."
            },
        ],
    },
    
    'wrist': {
        'higher': [
            {
                'issue': "You aren't using enough wrist snap on the {shot}, relying too much on your arm.",
                'fix': "Cock your wrist backward during preparation, and violently snap it forward exactly as you hit the shuttle.",
                'why': "The wrist snap is the final link in the kinetic chain and generates up to 30% of {ref}'s explosive {shot} speed.",
                'drill': "Sit in a chair and flick 30 shuttles over the net using ONLY your wrist and forearm, no shoulder movement.",
                'cue': "Violent snap.",
                'checkpoint': "The shuttle should leave your racket with a sharp, crisp 'pop' sound."
            },
            {
                'issue': "Your wrist stays flat through the {shot} — there's no acceleration at the end of your swing.",
                'fix': "Think of flicking water off your fingertips. That's the wrist motion you need at contact.",
                'why': "Without wrist pronation, you're hitting with your arm only. {ref} adds 20-30% more speed with wrist alone.",
                'drill': "Hold the racket at the very top of the handle and hit 20 {shot}s. This forces wrist-only power.",
                'cue': "Flick the water off.",
                'checkpoint': "Your racket head should visibly accelerate faster than your arm at the moment of contact."
            },
            {
                'issue': "Your wrist is too passive — it's just following your arm instead of leading the final burst.",
                'fix': "Pre-load your wrist by cocking it backward, then release it like a mousetrap at the exact contact point.",
                'why': "The wrist creates the final whip in the kinetic chain. Without it, all your leg and core power gets wasted.",
                'drill': "Practice 25 wrist flicks with just a shuttlecock (no racket), throwing it as high as possible using only wrist motion.",
                'cue': "Mousetrap release.",
                'checkpoint': "You should feel a distinct snap sensation in your forearm tendons at contact."
            },
        ],
        'lower': [
            {
                'issue': "Your wrist is over-extended or too floppy during the {shot}.",
                'fix': "Keep your wrist firm and slightly cocked until the exact moment of impact. Avoid breaking the wrist too early.",
                'why': "A floppy wrist completely destroys accuracy and timing, leading to unpredictable slice or framed shots.",
                'drill': "Hit 20 dropshots focusing completely on a firm, guided wrist rather than a snapping motion.",
                'cue': "Firm and guide.",
                'checkpoint': "The racket face should remain perfectly flat and stable upon contact with the shuttle."
            },
            {
                'issue': "You're releasing your wrist too early in the {shot} swing, losing control.",
                'fix': "Hold the wrist cocked until your elbow is fully forward, THEN release. Timing is everything.",
                'why': "Early wrist release means the snap happens before the shuttle arrives — power goes into thin air.",
                'drill': "Do 15 slow-motion {shot}s, counting 'elbow-wrist' out loud to train the sequence.",
                'cue': "Hold... NOW snap.",
                'checkpoint': "Contact should happen at the peak of your wrist acceleration, not before or after."
            },
            {
                'issue': "Your wrist angle is unstable through contact, causing inconsistent {shot} placement.",
                'fix': "Grip the racket more firmly with your last three fingers. This naturally stabilizes the wrist.",
                'why': "A loose bottom grip lets the racket twist on impact, sending shots unpredictably left or right.",
                'drill': "Squeeze a tennis ball 50 times with your racket hand, then immediately hit 10 {shot}s.",
                'cue': "Firm bottom three fingers.",
                'checkpoint': "Your shots should land within a 1-meter radius of your target consistently."
            },
        ],
    },
    
    'knee': {
        'higher': [
            {
                'issue': "Your legs are too straight and stiff during the {shot} movement.",
                'fix': "Bend your knees noticeably deeper before you swing. Load your weight into your legs like {ref}.",
                'why': "All power in badminton starts from the ground up. Stiff legs mean you are entirely relying on arm strength.",
                'drill': "Do 15 explosive jump squats. On landing, immediately trigger a shadow swing while pushing up from the floor.",
                'cue': "Sink and explode.",
                'checkpoint': "Your quads should burn. You should feel the power traveling from your legs, through your core, to your arm."
            },
            {
                'issue': "You're standing too tall during the {shot}. No leg drive is being used.",
                'fix': "Drop your center of gravity by bending both knees to roughly 130°. Push up as you swing.",
                'why': "{ref} generates ground reaction force by pushing up from bent knees — this adds 25% more power to overhead shots.",
                'drill': "Practice 20 {shot}s starting from a deep squat position. Explode upward as you swing.",
                'cue': "Push the ground away.",
                'checkpoint': "You should feel your calf muscles firing as you hit the shuttle."
            },
            {
                'issue': "Your knees aren't loading properly before the {shot} — zero athletic stance.",
                'fix': "Get into a ready position with knees bent BEFORE the shuttle comes. Don't bend and swing simultaneously.",
                'why': "Pre-loading the legs means the power is stored and ready. Bending while swinging wastes energy on two tasks at once.",
                'drill': "Do 10 split-step-to-{shot} sequences. Split step first, then bend, then swing. Three distinct phases.",
                'cue': "Ready position first.",
                'checkpoint': "You should feel grounded and stable before every single shot."
            },
            {
                'issue': "You're hitting flat-footed with locked knees, removing all explosiveness from your {shot}.",
                'fix': "Stay bouncy on your toes with soft knees. Feel like a spring, not a statue.",
                'why': "Locked knees transfer zero energy from the ground. You're swinging with arms only — that's 50% of your power gone.",
                'drill': "Jump rope for 2 minutes, then immediately hit 10 {shot}s. The bouncy feel should carry over.",
                'cue': "Be a spring, not a stick.",
                'checkpoint': "You should leave the ground slightly during powerful overhead {shot}s."
            },
        ],
        'lower': [
            {
                'issue': "You are sinking too deeply into your knees, slowing your recovery after the {shot}.",
                'fix': "Keep your knees bent, but stay in a higher, more athletic stance. Don't let your hips drop below the knee line.",
                'why': "Dipping too low traps your center of gravity, making it impossible to quickly return to the center of the court.",
                'drill': "Perform 10 fast-paced shadow swings, focusing specifically on how fast you can push instantly back to the center.",
                'cue': "Stay light, stay high.",
                'checkpoint': "You should be able to instantly bounce backward off your loaded leg."
            },
            {
                'issue': "Your knee bend is too deep — you're sitting into the shot, not springing from it.",
                'fix': "Think 'athletic crouch' not 'squat'. Your knees should bend to about 140°, not below 120°.",
                'why': "Over-bending makes the push-up phase too slow. {ref} stays in a quick-fire position, never going too deep.",
                'drill': "Place a chair behind you. If your butt touches it during {shot} practice, you're too low. Do 15 reps.",
                'cue': "Athletic, not deep.",
                'checkpoint': "You should be able to change direction instantly after every shot."
            },
            {
                'issue': "Excessive knee flexion is absorbing your power instead of transferring it upward.",
                'fix': "Focus on a quick, shallow dip and explosive extension rather than a slow, deep bend.",
                'why': "A fast shallow dip creates more elastic energy than a slow deep squat. Speed of bend matters more than depth.",
                'drill': "Do 20 mini-jumps (2-inch hops) and hit {shot}s. Train your body to use quick, small knee bends.",
                'cue': "Quick dip, fast up.",
                'checkpoint': "Your knee bend and extension should happen in under half a second."
            },
        ],
    },
    
    'ankle': {
        'higher': [
            {
                'issue': "You are flat-footed during the {shot}.",
                'fix': "Get your heels off the ground. Stay entirely on the balls of your feet during the preparation phase.",
                'why': "Being flat-footed kills your reaction time. Staying on your toes activates your calf muscles into a permanent 'spring' state.",
                'drill': "Do 30 seconds of rapid split steps on the balls of your feet, never letting your heels touch the floor.",
                'cue': "Heels up, spring loaded.",
                'checkpoint': "You should feel a continuous bounce and tension in your calf muscles."
            },
            {
                'issue': "Your ankles are stiff and locked, reducing court mobility during the {shot}.",
                'fix': "Keep your ankles flexible and springy. Imagine standing on hot coals - stay light and ready to move.",
                'why': "{ref}'s explosive movement starts from active ankles. Stiff ankles add 0.3 seconds to your reaction time.",
                'drill': "Do 20 calf raises between each set of {shot} practice. Keep those ankle muscles activated.",
                'cue': "Light feet, hot coals.",
                'checkpoint': "You should be dancing on your toes between every rally point."
            },
            {
                'issue': "Your feet are planted too firmly, making you react late to the shuttle during {shot} exchanges.",
                'fix': "Keep a micro-bounce going at all times. Your feet should never be completely still on court.",
                'why': "Static feet require extra time to initiate movement. A constant micro-bounce keeps muscles pre-loaded.",
                'drill': "Practice 'ghosting' — shadow badminton with nonstop foot movement for 3 minutes straight.",
                'cue': "Never stop moving.",
                'checkpoint': "Even when waiting, your feet should have a slight bounce rhythm."
            },
        ],
        'lower': [
            {
                'issue': "You are stepping awkwardly on your toes during the lunge for the {shot}.",
                'fix': "On your final lunge step, always land heel-first and roll to the toe, creating a braking mechanism.",
                'why': "Landing toe-first on a lunge places extreme stress on the knee joint and completely ruins your stopping balance.",
                'drill': "Do 20 slow-motion lunges. Exaggerate extending your foot out and striking the floor with your heel first.",
                'cue': "Heel strike to brake.",
                'checkpoint': "Your front foot should be perfectly flat and anchored when executing the shot."
            },
            {
                'issue': "Your ankle is rolling inward during the {shot} lunge, risking injury.",
                'fix': "Focus on landing with your foot pointing toward the shuttle. Keep the ankle stable and neutral.",
                'why': "Ankle inversion during lunges is the #1 cause of badminton ankle sprains. Neutral foot = safe foot.",
                'drill': "Do 15 lunges with a resistance band around your ankles. This trains stability under load.",
                'cue': "Foot straight, ankle stable.",
                'checkpoint': "You should feel no wobble in your ankle during any lunge step."
            },
            {
                'issue': "Your ankle dorsiflexion is excessive — your toes are coming up too high during {shot} footwork.",
                'fix': "Keep your foot closer to flat during movement. Only the balls of your feet should drive the push-off.",
                'why': "Over-dorsiflexion slows your push-off phase and wastes energy on unnecessary ankle movement.",
                'drill': "Practice 20 shuttle runs (6 meters) focusing on quick, flat pushoffs without lifting your toes.",
                'cue': "Quick and flat.",
                'checkpoint': "Your footwork should feel smooth and gliding, not stomping."
            },
        ],
    },
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
