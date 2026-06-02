# SessionCoach — Full Implementation Plan (Simple English)

**Project name:** SessionCoach (or Badminton Practice Coach)  
**Time you have:** 5 days × 6 hours = **30 hours**  
**Goal:** Help badminton players improve without paying for expensive coaches.  
**College:** Final year major project  

---

## 1. What are you building? (In one paragraph)

A **website** where a player uploads a **2–5 minute practice video**. The app:

1. Finds **how many shots** they played (smash, clear, drop, etc.)
2. Gives a **session score** based on **their own consistency** (not “be like a world champion”)
3. Shows **best moments** and **needs improvement** as short video clips
4. Gives **3–5 simple tips** (bend knees, stay balanced, repeat same arm path)
5. Saves history so they can see **progress over weeks**

You are **not** replacing a human coach. You are a **cheap training diary + video helper**.

---

## 2. Why remove “reference player” comparison?

Your sir said:

- Every player’s body is different  
- One pro’s angles are not “the only correct form”  
- You cannot prove your angle numbers are 100% correct from a phone video  

**New rule:** Compare the player **to themselves** in the same session and over past sessions.  
That is fair, honest, and still useful.

---

## 3. Which datasets to use? (Important)

There is **no single perfect file** with video + angles + shot labels + quality score all together.  
You use **different datasets for different jobs**.

### For your 5-day project (MVP — what you actually need)

| Priority | Dataset | Download from | What you use it for | Do you need full download? |
|----------|---------|---------------|---------------------|----------------------------|
| **Must cite in report** | **ShuttleSet / ShuttleSet22** | [GitHub: CoachAI-Projects](https://github.com/wywyWang/CoachAI-Projects) | Proof that shot labels exist in research; 18 shot types, hit timing | **No** for MVP — cite in report + optional small sample |
| **Must cite in report** | **MultiSenseBadminton** | [Nature paper + figshare](https://www.nature.com/articles/s41597-024-03144-z) | Real joint angles + stroke types from sensors (for “related work”) | **No** for MVP |
| **For manual testing** | **Your own 2–3 videos** | Phone camera at club/home | Count shots by hand vs app — **required for viva** | **Yes** — you record these |
| **Optional demo** | **1 pre-processed demo session** | You create once pipeline works | Viva demo without waiting | **Yes** — you make this |

**For 5 days:** You do **NOT** need to download 50 GB ShuttleSet.  
You build the app with **MediaPipe + rules**, and write in the report: *“Future work: train on ShuttleSet labels.”*

### For future / if you get extra time after submission

| Dataset | Link | Use later |
|---------|------|-----------|
| **Shuttlecock Trajectory** | [HackMD guide](https://hackmd.io/@TUIK/rJkRW54cU) | Better “when did the shuttle get hit” |
| **BadminSense BADS_CLL** | [GitHub](https://github.com/taizhouchen/BadminSense_Dataset) | Shot quality scores from coaches (848 clips) |
| **RichardPinter/badminton_shot_type** | [GitHub](https://github.com/RichardPinter/badminton_shot_type) | Pre-trained models on ShuttleSet |
| **BFMD** | [arXiv paper](https://arxiv.org/abs/2603.25533) | Full match + pose keypoints (newest) |

---

## 4. What data goes inside YOUR app (your own database)

You create this when users upload videos. No external “reference JSON” needed.

### Table: `User`
- username, email, password  

### Table: `PracticeSession`
| Field | Meaning |
|-------|---------|
| `video_file` | Uploaded mp4 path |
| `upload_date` | When uploaded |
| `duration_min` | Video length |
| `status` | processing / done / failed |
| `total_shots` | e.g. 47 |
| `session_score` | 0–100 combined score |
| `consistency_score` | How similar same shot types look |
| `stability_score` | Balance at contact |
| `report_json` | Full details (see below) |

### Inside `report_json` (one session example)

```json
{
  "shot_counts": {"smash": 12, "clear": 10, "drop": 8, "drive": 9, "lift": 5, "net": 3},
  "strong_shots": 29,
  "weak_shots": 18,
  "highlights": {
    "best": [{"time": "0:42", "type": "smash", "score": 91, "clip": "best_1.mp4"}],
    "worst": [{"time": "1:18", "type": "lift", "score": 34, "clip": "worst_1.mp4"}]
  },
  "tips": ["Bend knees more on lifts...", "..."],
  "shots": [{"frame": 1200, "type": "smash", "stability": 85, "elbow": 142}]
}
```

This is **your** clean labeled data — built from each upload.

---

## 5. How the app works (step by step)

```
Player uploads video
        ↓
Open video with OpenCV (read frames)
        ↓
MediaPipe Pose → body joints each frame
        ↓
Find "contact" moments (fast wrist movement / swing peak)
        ↓
Group contacts 1 second apart = separate shots
        ↓
Classify each shot: smash / clear / drop / drive / lift / net (rules from body position)
        ↓
Calculate angles at contact (shoulder, elbow, wrist, knee, ankle)
        ↓
Score session:
   - Consistency = same shot type → similar angles?
   - Stability = head balanced over hips?
        ↓
Pick best 3–4 and worst 3–4 shots (high vs low stability in THIS video)
        ↓
Cut short clips with FFmpeg/OpenCV
        ↓
Generate tips from weak points
        ↓
Save to database + show report page
```

**No step compares to An Se-young or any pro.**

---

## 6. How you score performance (simple formulas)

### Consistency (0–100)
- Take all **smashes** in the video  
- If elbow/knee angles are **similar every time** → high score  
- If every smash looks different → low score  
- Same for clears, drops, etc.  
- Average across shot types  

**Meaning for player:** “Your technique is repeatable” = good for match play.

### Stability (0–100)
- At each shot: is the head centered over the hips?  
- Average across all shots  

**Meaning:** “You stay balanced when hitting.”

### Session score
```
Session score = 60% consistency + 40% stability
```

### Best shot
- Highest stability + good wrist speed in **this session**

### Worst shot
- Lowest stability in **this session**

---

## 7. Tech stack (keep it simple)

| Part | Tool | Why |
|------|------|-----|
| Website | **Flask** (Python) | Fast, you already know it |
| Database | **SQLite** | No server setup |
| Video read | **OpenCV** | Standard |
| Body tracking | **MediaPipe Pose** | Free, works offline |
| Charts | **Chart.js** | Donut + bar charts |
| Clips | **OpenCV** video writer | Cut 3–5 sec segments |
| AI help | **Cursor** | Write code faster — you still test |

**Do NOT use** Claude Vision API as main brain (hard to defend in viva).  
**Do NOT rebuild** in React Native in 5 days.

---

## 8. Folder structure (clean project)

```
major project/
├── app.py                    # Flask routes
├── models.py                 # User + PracticeSession
├── session_coach/            # NEW — all analysis code
│   ├── pose_track.py         # MediaPipe on video
│   ├── shots.py              # find & group shots
│   ├── metrics.py            # scores & tips
│   ├── clips.py              # cut best/worst videos
│   └── pipeline.py           # runs everything
├── templates/
│   ├── dashboard.html
│   ├── upload.html
│   ├── session_report.html   # main result page (your mockup)
│   └── methodology.html      # how it works (for sir)
├── static/css/style.css
├── uploads/                  # user videos
├── processed/                # clips
├── data/demo/                # pre-made demo for viva
└── docs/
    ├── IMPLEMENTATION_PLAN.md   (this file)
    └── MANUAL_VALIDATION.md     (your hand counts vs app)
```

**Remove or hide:** Reference Pro page, An Se-young JSON, similarity % to pro.

---

## 8. Screens you must build

### Screen 1 — Login / Register
Already exists — keep it.

### Screen 2 — Dashboard
- Upload new session button  
- List of past sessions with date + score  
- Small line chart: last 5 session scores  

### Screen 3 — Upload
- Drag & drop mp4  
- Tips: “Film from side, full body visible, 2–5 min, 30+ fps”  
- Processing spinner  

### Screen 4 — Session Report (main mockup)
- Header: player name, date, duration, “47 shots detected”  
- 3 cards: Session score | Strong shots | Needs work  
- Bar chart: score per shot type  
- Donut: shot mix %  
- Grid: Best 4 clips | Worst 4 clips  
- Box: AI coaching tips (rule-based, not GPT required)  

### Screen 5 — Methodology
- Pipeline diagram  
- “We compare you to yourself, not to one pro”  
- Limitations: phone angle, MediaPipe estimate  
- Datasets cited: ShuttleSet, MultiSenseBadminton  

---

## 9. Five-day plan (6 hours each day)

### Day 1 — Backend brain (6 h)
| Hour | Task |
|------|------|
| 1 | Create `PracticeSession` in `models.py` |
| 2 | Finish `session_coach/pipeline.py` — video in, JSON out |
| 3 | Test in terminal: `python -m session_coach.cli video.mp4` |
| 4 | Fix errors (MediaPipe, empty video, no person detected) |
| 5 | Wire background worker in `app.py` |
| 6 | Write ½ page report: Introduction |

**Done when:** One video produces JSON with shot counts and scores.

---

### Day 2 — Upload + report page (6 h)
| Hour | Task |
|------|------|
| 1 | Upload route + save file |
| 2 | Build `session_report.html` — cards + shot bars |
| 3 | Chart.js donut for shot mix |
| 4 | Connect report to `report_json` from DB |
| 5 | Loading / failed states |
| 6 | Report section: System design |

**Done when:** Upload → wait → see numbers on screen.

---

### Day 3 — Clips + tips (6 h)
| Hour | Task |
|------|------|
| 1 | `clips.py` — cut 4 sec around best/worst frame |
| 2 | Show video cards on report page |
| 3 | Tips from `metrics.generate_tips()` |
| 4 | Test 2 real videos |
| 5 | Fix wrong shot counts if possible |
| 6 | Start `MANUAL_VALIDATION.md` table |

**Done when:** Play good/bad clips in browser.

---

### Day 4 — Dashboard + demo + cleanup (6 h)
| Hour | Task |
|------|------|
| 1 | Session history on dashboard |
| 2 | Progress chart (last 5 sessions) |
| 3 | Create demo session (pre-processed, instant open) |
| 4 | `methodology.html` + remove Reference Pro from menu |
| 5 | Simple UI polish (dark theme like mockup) |
| 6 | Report: Implementation + Testing |

**Done when:** Full flow works + demo button for viva.

---

### Day 5 — Report, slides, viva prep (6 h)
| Hour | Task |
|------|------|
| 1 | Finish report (8–12 pages) |
| 2 | PowerPoint 10 slides |
| 3 | Manual validation table (3 videos) |
| 4 | Record 2-minute demo video (backup) |
| 5 | Fix last bugs |
| 6 | Practice viva answers (see section 11) |

**Done when:** Submit report + working demo.

---

## 10. Testing table (required for college)

Record this in `docs/MANUAL_VALIDATION.md`:

| Video | You counted shots | App counted | Match? | Notes |
|-------|-------------------|-------------|--------|-------|
| Demo 1 | | | | |
| Your practice 1 | | | | |
| Your practice 2 | | | | |

Honesty is good: “App detected 36/40 shots (90%)” is better than fake 100%.

---

## 11. Viva — answers in simple words

**Q: Where does data come from?**  
> User uploads their video. We extract pose with MediaPipe. Shot labels use body-position rules. For research we cite ShuttleSet and MultiSenseBadminton. We do not copy one champion’s angles.

**Q: How do you know angles are correct?**  
> They are estimates from AI pose detection, not lab sensors. We use them for relative feedback (consistency), not medical accuracy.

**Q: How is best/worst decided?**  
> Best = most stable, controlled shots in that session. Worst = least stable. Compared to the player’s own video, not a pro.

**Q: How does this help players without coaches?**  
> Shows what they did in practice, which shots they repeat, where form breaks down, and short clips to review — like a coach’s video feedback but free.

**Q: Limitations?**  
> Phone angle, lighting, heuristic shot types. Future: train on ShuttleSet for better labels.

---

## 12. Report chapter outline (copy for Word doc)

1. **Introduction** — Problem: coaching cost; Solution: SessionCoach  
2. **Objectives** — Detect shots, score consistency, show clips, track progress  
3. **Literature survey** — ShuttleSet, CoachAI, MediaPipe, MultiSenseBadminton (1–2 pages)  
4. **System analysis** — User needs, use cases  
5. **System design** — Block diagram, database, modules  
6. **Implementation** — Flask, session_coach pipeline, UI  
7. **Testing** — Manual validation table  
8. **Results** — Screenshots  
9. **Limitations & future work** — Datasets, TrackNet, mobile app  
10. **Conclusion**  

---

## 13. What to tell AI (Cursor) each day

Copy these when you sit down:

**Day 1:**  
> “Complete session_coach/pipeline.py and PracticeSession model. No reference player. Test CLI.”

**Day 2:**  
> “Add upload route and session_report.html with Chart.js from report_json.”

**Day 3:**  
> “Implement clip cutting and video cards on report page.”

**Day 4:**  
> “Dashboard history chart, methodology page, hide reference routes.”

**Day 5:**  
> “Help write testing section and viva Q&A from IMPLEMENTATION_PLAN.md.”

---

## 14. Success checklist (before submission)

- [ ] User can register and login  
- [ ] User can upload mp4  
- [ ] Report shows shot counts + session score  
- [ ] Best/worst clips play  
- [ ] At least 3 tips shown  
- [ ] Demo session opens in 1 click  
- [ ] Methodology page explains no pro comparison  
- [ ] Report cites ShuttleSet + MultiSenseBadminton  
- [ ] Manual validation table filled  
- [ ] Reference player feature removed or hidden  

---

## 15. One-line project synopsis (for college form)

> SessionCoach is a web application that analyzes badminton practice videos using computer vision to count shot types, measure technique consistency and balance, highlight best and weakest moments as clips, and provide actionable training tips — helping amateur players improve affordably without professional coaching fees.

---

**Start here:** Day 1 → run pipeline on one video → everything else builds on that.
