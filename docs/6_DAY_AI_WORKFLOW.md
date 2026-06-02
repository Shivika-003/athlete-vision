# SessionCoach — 6 hours/day × 5 days (AI-assisted build)

Use Cursor AI as a **pair programmer**, not as magic. You review, test, and demo.

## Daily rhythm (repeat every day)

| Block | Time | You do | AI prompt style |
|-------|------|--------|-----------------|
| 1 — Plan | 30 min | Read yesterday output; list 3 tasks | "Today I need X, Y, Z. What's the smallest order?" |
| 2 — Build | 3 h | Run app after each feature; paste errors | "Fix this traceback: …" / "Add route for …" |
| 3 — Test | 1 h | Upload 1 real video; note wrong counts | "Shot count is 12 but should be ~20. Improve contact detection." |
| 4 — Report | 1 h | Screenshots + 1 report section | "Write limitations paragraph for Section 7" |
| 5 — Commit | 30 min | `git add` + short message | — |

## Day 1 (6 h) — Pipeline works
- [ ] Run: `python -m session_coach.cli test uploads/sample.mp4`
- [ ] JSON has: shot_counts, session_score, highlights[]
- **AI:** "Wire PracticeSession model and worker to session_coach.pipeline"

## Day 2 — Upload + report UI
- [ ] Upload page → processing → session report (cards + bars)
- **AI:** "Build session_report.html from this JSON schema: …"

## Day 3 — Clips + tips
- [ ] 3 best + 3 worst video cards play in browser
- **AI:** "Fix clip paths and ffmpeg fallback in session_coach/clips.py"

## Day 4 — Dashboard + demo data
- [ ] History chart (last 5 sessions)
- [ ] Pre-load `data/demo/demo_session.json` for viva
- **AI:** "Add methodology.html explaining consistency score"

## Day 5 — Report + viva
- [ ] 8-page report + 10 slides + 2 min screen recording
- **AI:** "Manual vs automatic validation table template"

## Good AI prompts (copy-paste)

```
Implement only session_coach/pipeline.py — no reference player, 
score = consistency vs own session. Reuse ai_engine.angle_utils.
```

```
Add Flask route POST /session/upload and GET /session/<id> 
using PracticeSession.report_json. Do not change auth routes.
```

```
Professor asked: where does data come from? 
Add methodology page: MediaPipe pose, heuristic shot labels, 
future ShuttleSet — 200 words, student tone.
```

## What NOT to ask AI in 5 days
- "Build TrackNet from scratch"
- "Rewrite entire app in React Native"
- "Make 90% accuracy guarantee"

## Viva backup
- Always open **Demo Session** first (no upload wait)
- Keep `docs/MANUAL_VALIDATION.md` with your hand-counted shots
