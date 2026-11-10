# Mitosis

**One winning ad in, a tested batch of on-brand variations out.**

Mitosis takes a proven Meta ad (the video plus its script) and multiplies it into dozens of variations. Each one gets a new hook, a new voice, a new opening location, and all three placement formats. The hero stays the same person on screen, every asset passes a quality gate before anyone sees it, and every cent and second is accounted for.

```
winner.mp4 + script ──► ingest ─► understand ─► lock hero ─► write hooks ─► design batch ─► produce ─► QA gate ─► report / exports
                         shots     creative      identity      archetypes     pairwise      keyframe    identity     HTML report
                         words     DNA           embedding     policy guard   covering      i2v clip    technical    Meta bulk CSV
                         stems                   references    bandit priors  budget cap    voice/STS   safe zones   AE script
                                                                                            mix + ASS   policy       ZIP bundle
                                                                                            9:16 4:5 1:1 vision     n8n via REST
```

## Why it exists

Ad spend grows only as fast as the team ships new creatives. Iterating on a winner (new hook, new voice, new format) is repetitive work that eats the hours people should spend on new ideas. Mitosis does the repetition and leaves people with the decisions.

## What makes it production-grade

| Problem in real creative ops | How Mitosis handles it |
| --- | --- |
| The hero drifts between shots and variants | Face embeddings (YuNet + SFace) of the winner build a hero pack. Every generated keyframe is checked **before** paying for video, and every generated clip is checked frame by frame for drift and flicker. A failed check re-rolls the seed automatically. |
| Voice swaps break lip-sync on talking-head footage | Talking-head winners keep their original performance: the body is re-voiced with **speech-to-speech** (same timing, new timbre). Only voiceover-style winners get full TTS. The choice is automatic. |
| Meta rejections cost money and account health | A policy guard lints every hook (personal attributes, sexual-wellness rules, health claims, before/after, sensationalism), repairs what it can and blocks the rest, before generation and again on the final render. |
| Models change every month | Every model is a line in `config/models.yaml`. Each media kind is a **fallback chain** with retries, circuit breakers and per-model concurrency limits. Swapping Kling for the next model is a config change. |
| "How much does one variation cost?" | Every call goes through a ledger: cost, latency, provider, stage, attempt. Offline runs record **production-equivalent prices**, so cost per variant is known before spending anything. |
| Paying twice for the same asset | Content-addressed cache. A keyframe, clip or voice line is generated once and reused across formats, re-runs and variants. |
| Tests that can't be read | The planner builds a pairwise covering design across hook × voice × location, so results can be attributed to factors. Hooks are allocated with Thompson sampling over priors learned from past Meta results. |
| Text under the Reels UI | Captions and headlines are positioned and verified against per-format safe zones (Reels keeps the top 14% and bottom 35% clear). |
| Designers and marketers need to use it without an engineer | A web panel: upload, pick voices and locations, estimate cost, multiply, review, approve. n8n flows run the same thing from a Google Drive drop. |

## Quick start

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev,live]"
.venv/bin/mitosis doctor
.venv/bin/mitosis sample                      # builds samples/winner.mp4 + script + music stem
.venv/bin/mitosis plan briefs/sample.yaml     # hooks, policy blocks, batch design, cost estimate
.venv/bin/mitosis run briefs/sample.yaml --open
.venv/bin/mitosis serve                       # web panel on http://127.0.0.1:8765
.venv/bin/mitosis lint "Struggling with ED? Your wife deserves better"
```

With no API keys, every model chain falls back to local stand-ins (macOS `say`, ffmpeg Ken Burns, rubberband voice shift). The whole pipeline still runs end to end and produces real, watchable files. Add keys to `.env` and the same run goes live:

```bash
cp .env.example .env    # ANTHROPIC_API_KEY, FAL_KEY, ELEVENLABS_API_KEY, RUNWAYML_API_SECRET, ...
.venv/bin/mitosis run briefs/sample.yaml --live
```

## Results on the sample brief

Offline run, 10 variants x 3 placements, 4 workers, MacBook:

| | Fresh run | Re-run (cache warm) |
| --- | --- | --- |
| Renders | 30 (10 variants x 9:16, 4:5, 1:1) | 30 |
| QA gate | 10 passed, 0 flagged, 0 failed | 10 passed |
| Wall time | 137 s (13.7 s per variant) | 135 s (13.4 s per variant) |
| Production-equivalent cost | $5.62 ($0.56 per variant) | $0.00 |
| Cache hits / avoided spend | 4 / $0.19 | 41 / $5.81 |

The planner budgets $0.98 per variant before anything runs. The measured $0.56 covers keyframes, image-to-video, voice and transcription. Lip-sync and Claude calls only exist on the live path, so the planner figure is the one to budget with.

Offline, ffmpeg rendering dominates wall time, so the cache shows up in spend rather than seconds. Live, every cache hit skips minutes of video generation.

Policy guard on real-world copy:

```
$ mitosis lint "Struggling with ED? Your wife deserves better" "Lose 20 pounds and she'll want you again"
blocked  Struggling with ED? Your wife deserves better
         repair Intimacy changes over the years, and that's normal
blocked  Lose 20 pounds and she'll want you again
         no compliant rewrite, hook dropped
```

## Brief

```yaml
winner:
  id: W017
  video: ../samples/winner.mp4          # textless master, captions are rebuilt per variant
  script: ../samples/winner_script.txt
music: ../samples/music_bed.wav         # optional; otherwise stems are separated when demucs is installed
variants: 10
aspects: ["9:16", "4:5", "1:1"]
voices: [warm_30s, deep_40s, casual_20s, calm_coach, partner_pov]
locations: [original, living_room_evening, kitchen_morning, parked_car, balcony_golden_hour, home_office]
hook_count: 6
budget_usd: 25
max_attempts: 2
```

## Layout

```
mitosis/
  pipeline.py        stage orchestration, retries, parallel variants
  registry.py        model chains, fallback, circuit breakers, cost
  cache.py ledger.py content-addressed cache, cost/time ledger
  providers/         anthropic, fal (Nano Banana, Seedream, Kling, Seedance, Veo), elevenlabs, runway, higgsfield, comfyui, offline
  media/             ffmpeg engine: probe, shots, frames, stems, alignment, captions, reframing, assembly
  intelligence/      structure, creative DNA, policy guard, hooks, planner, priors, pre-launch score
  qa/                identity lock, technical checks, safe zones, vision judge, gate
  report/            HTML report, Meta bulk CSV, After Effects script, bundle
  service/           FastAPI + web panel
config/              models, brand standards, voices, locations, policy rules
workflows/           n8n (Drive → Mitosis → Slack/Meta, daily learning loop), ComfyUI (FLUX + PuLID keyframe)
```

## Tests

```bash
.venv/bin/python -m pytest
```
