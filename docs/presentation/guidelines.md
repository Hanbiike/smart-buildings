# Video presentation codex (guidelines)

> Binding rules for producing the 3-minute demo video. Stack: HyperFrames
> (HTML rendered to video) + Three.js 3D + SOTA TTS voice. See
> [01](01-hackathon-demo-deep-research.md) for the evidence behind the story
> beats, [02](02-competition-reference.md) for the competition facts, and
> [judging](judging.md) for the Meta-hackathon rubric (when that is the target).

## 1. Concept (locked)

- The demo is a 3D walkthrough of ONE building: morning to evening, one day of
  project operation, across varied weather conditions.
- The same building recurs in every weather/time beat so the viewer reads
  cause and effect, not a montage of places.
- Everything else (hook cards, problem stats, architecture, metrics, CTA) is
  2D motion graphics. 3D is reserved for the demo; 2D never substitutes for it.

## 2. Mandatory 3D beats (all must appear)

1. Building exterior and interiors with continuous camera motion (orbit,
   fly-through, dolly — no static 3D plates).
2. Day-cycle lighting: dawn, midday, dusk, night on the same views.
3. X-ray/cutaway revealing what the policy controls (AHUs, VAV boxes, zones,
   hot-water loop).
4. Zoom from building scale down to a controlled element (damper, reheat
   valve, setpoint) at the moment the policy acts.
5. People reactions: occupants visibly comfortable as a result of control
   (the payoff shot of each weather beat).
6. Weather variants on the same building (heat, cold, swing/shoulder at
   minimum; exact set fixed in SPEC.md).

## 3. Story skeleton and clock (from [01](01-hackathon-demo-deep-research.md))

Total cap 180 seconds; target 170 seconds (2:50).

- 0:00-0:20 hook with numbers, product visible in the first seconds.
- 0:20-0:45 problem and persona (30% problem / 70% solution overall).
- 0:45-2:15 the 3D walkthrough (Section 2), wow moment lands by 1:45.
- 2:15-2:40 how it works (2D architecture + concrete stack names).
- 2:40-3:00 impact metrics, next steps, CTA, team tag.
- Narration budget about 150 words/minute with 15% slack for pauses and demo.
- If the target is the Meta hackathon, weight beats to its rubric: Impact 30%,
  Use of Meta AI Tooling 25%, Technical Depth 20%, Demo 25% ([judging](judging.md)).

## 4. 3D production rules

- Registry first: search `npx hyperframes catalog --query "<beat>" --json`
  before hand-building any 3D block, camera move, or transition; report gaps
  via `feedback --search-miss` instead of silently hand-rolling.
- Three.js work follows the `threejs` skill hub: read exactly the topic file
  the task touches (fundamentals, geometry, materials, lighting, textures,
  animation, loaders, shaders, postprocessing, interaction).
- Deterministic picture: every visible value is a function of timeline time
  (no `Date.now()`, unseeded random, fetches, timers, infinite repeats).
  Same timeline position renders the same frame, every time.
- Lighting sells the day cycle: grade by time of day; night beats must stay
  readable (no crushed blacks on controlled elements).
- Text safety: 3D labels and HUD inside the 90% action-safe box, key text
  inside 80% title-safe.

## 5. Voice (SOTA TTS)

- Voiceover is mandatory: full narration script, no silent explanation gaps,
  no on-screen text left unvoiced.
- Quality bar: SOTA neural TTS — natural prosody, stable long-form delivery,
  one consistent voice for the whole film.
- Pipeline: locked script first, then TTS, then timeline alignment, then mix
  (duck music under voice). Re-render voice only from script diffs.
- Voice asset resolves through `media-use` (HeyGen Voice Turbo free with
  sign-in, other HeyGen voices/cloning on credits, or local `npx hyperframes
  tts` Kokoro on device). The exact engine is chosen at SPEC time against the
  bar above and recorded in SPEC.md; no engine is prescribed here.

## 6. Compute (RTX 5090)

- Verified host GPU (2026-11-02): `NVIDIA GeForce RTX 5090 D v2, 24455 MiB,
  driver 595.99.02`.
- The 5090 carries all heavy local work: HyperFrames local renders and
  snapshot previews, TTS inference when a local engine is chosen, and any
  local generative media (FLUX via mflux, LTX clips) if the media plan calls
  for them.
- Cloud/credit spend (HeyGen voices, generated music/images) is planned per
  the media-use credit rules and never silently substituted with weaker
  free options.

## 7. Music and sound

- Music and sound on visible actions (film over ~15 s that tells something),
  unless silence is explicitly requested.
- Resolve via `media-use`: 10k+ track catalog, Openverse/bundled SFX, local
  MusicGen only if asked.
- Mix: fades, gain, ducking under voice; every `<audio>` keeps its `id` for
  the mix; timed wrapper or media, never both.

## 8. HyperFrames contract (gates, not suggestions)

- Root `index.html` holds variables and mounts one scene per sub-composition
  from `compositions/`; media and fonts in `assets/`; 1920x1080 at 30 fps.
- Every timed element: `class="clip"`, `data-start`, `data-duration`.
- One paused GSAP timeline per composition, registered in
  `window.__timelines`; no `play()`/`pause()`/seeking of media.
- Asset paths `assets/...` without leading `/`; bundled fonts or `@font-face`
  from `assets/fonts/`; grades via `media-treatment`, never CSS filters.
- After the last edit: `npx hyperframes check --json` — fix every error,
  weigh every warning. Then `snapshot --review` and REVIEW.md. Render only
  after check and review pass and the Studio preview is approved.

## 9. Definition of done

- [ ] 170 s target / 180 s hard cap; per-beat second budgets sum to the cap.
- [ ] All Section 2 beats present and readable (building, camera moves,
  day cycle, x-ray, control zoom, people payoff, weather variants).
- [ ] Wow moment lands by 1:45; working product visible in the first seconds.
- [ ] Concrete impact numbers on screen and in voice; concrete stack names.
- [ ] One consistent SOTA voice, fully mixed with music and SFX.
- [ ] Section 10 retention layer applied (hook test, stakes ladder,
  two re-engagement spikes, loop map, abrupt ending).
- [ ] `check --json` clean, REVIEW.md written, preview approved, file rendered.

## 10. Retention layer (MrBeast doctrine, adapted for a jury demo)

Source research: the leaked MrBeast production guide (memo), the
[Lex Fridman #351](https://podscript.ai/podcasts/lex-fridman-podcast/351-mrbeast-future-of-youtube-twitter-tiktok-and-instagram/)
retention interview, the EditorDuel retention-architecture analysis, the
SozAI/Thumblore/MonitorYT hook distillations, and the 180-second-demo guides
(Descript, prodSens.live, Bullpen, Clueso, ContentCreators, AIR, VidLocalizer).
Rule of use: borrow the structure, never the hype. Numbers in brackets are
adapted from long-form doctrine to this 170-second jury film.

### 10.1. Non-negotiable doctrine

- First seconds decide. State the title promise and the stakes inside the
  first 5 seconds; no logo, greeting, intro card, or context dump before
  the hook (context kills hooks). Context + stakes + one open curiosity gap
  all land inside the first ~20 seconds.
- Packaging first. Decide the title and the poster frame before building;
  the first frame must confirm that promise (same object, color, scale).
- Stakes ladder. Every segment is bigger than the last; the payoff lives
  only at the end; the ending is abrupt at peak energy — never signal the
  end early, and the last 15 seconds must feel special.
- No dead air. Something visual changes every 3-5 seconds (cut, zoom,
  overlay, camera move). Jury adaptation: the original doctrine says 2-4 s,
  but a jury must read charts, so number slides hold 4-6 s and the pace
  serves the story (per MrBeast's own 2024 walkback of overstim editing).
- Planned re-engagement spikes at ~1/3 and ~2/3 of runtime (~55 s and ~115 s
  here): a stakes raise, a weather flip, or a visual reveal that resets
  attention structurally, not by cutting faster.
- Curiosity loops. Every payoff immediately opens the next loop; the film
  returns to the original hook visually or verbally past the midpoint;
  close every loop by the end.
- Crazy progression. Front-load progress instead of telling linearly
  (compress the day into the first acts so the viewer is invested early).
- Micro-hooks. One emphasis word at a time; key phrases for stakes/numbers
  hold 2-5 s (long enough to read twice); captions stay short and synced.
- Sound is pacing: a constant bed, SFX only on transitions and actions,
  strategic silence before key moments, music well under narration, a short
  pause after each key beat.
- Energy cycling. Alternate high-energy spectacle beats with quiet
  low-energy beats so the film never plateaus.
- Effort rule. Half the production effort goes to the hook and the first
  30 seconds: a great hook with average content beats the reverse.

### 10.2. Jury guardrails (what NOT to take)

- No hype delivery, screaming, or meme sounds over the TTS; narration stays
  calm, expert, and genuine (never monotone, never animated-shouty).
- No fake stakes, clickbait promises, or viewer-trapping; every promise is
  backed by a metric shown later (alignment beats cleverness).
- No per-building-tuning or technical-overload tangents; memorable beats
  comprehensive (judges watch dozens of pitches).
- No subscribe-style outro or "thanks for watching" wind-down; the CTA is
  one line on the end card.

### 10.3. Insertion map for this film (170 s)

- 0-6 s hook: building + promise title + visual proof; stakes in one line.
- 6-20 s finance: stakes ladder (day, month, year), overlay beat every
  3-4 s, voice micro-hook ("how — at 0:45").
- 20-30 s title card: short brand beat AFTER the hook, never before it.
- 30-70 s comfort/control rooms: compressed day arc, actuator close-ups,
  warm-versus-cold contrast between adjacent rooms.
- ~55 s spike 1: storm arrival / VAV macro + a question overlay
  ("will the policy hold?").
- 70-105 s: low-energy night versus high-energy equipment spectacle
  (show the system's scale as input bias).
- ~115 s spike 2: peak-load alarm with the live agent response.
- 105-145 s metrics: ladder comfort, energy, carbon, reward — one number
  at a time, each bigger than the last, payoff teased as one final digit.
- 145-160 s claim: one result sentence fullscreen with the voice.
- 160-170 s end card: abrupt cut on the music+payoff peak; logo and links
  only, no wind-down.
