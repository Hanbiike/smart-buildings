# REVIEW.md — pass 1 (2026-11-02)

## Creative calls

- Built storyline A ("One building, one day"); B (result-first zoom) and C
  (split before/after) not built. A matches the codex walkthrough verbatim.
- Registry blocks used: snow (cold snap), starfield (night). Hand-built the
  3D world (no building block exists; gap reported via feedback), all 2D
  scenes, and decimal counters (count-up is integer-only; logo-sting needs a
  brand mark we do not have).
- Voice: Kokoro `am_adam` on CUDA (HeyGen unsigned, non-interactive; user
  directed 5090 use). Music: MusicGen-medium, 7 segments, 2.5 s crossfades,
  seam flux below interior baseline.
- End card carries the starter-kit URL, no invented team name (none supplied).

## Measured vs looked

- Measured: `check --json` ok:true, 0 errors; determinism (t=95 twice,
  identical md5); audio durations/pacing math; music seam flux; contact
  sheets + review beats/seams/phone sheets.
- Looked at: every sheet above; individual PNGs for 158/162 (sheet cells
  lied — frames are correct, sheets can lag on late 2D scenes).
- Nobody checked: how it SOUNDS (no playback; mix verified structurally
  only). SFX choices (bundled, intent-matched) were never auditioned.

## Notes (pass 1)

- P0 (fixed): t88 VAV label filled the frame at macro range → proximity fade
  by camera distance.
- P0 (fixed): end-card rule struck through wrapped URL → URL 56 px, nowrap.
- P1 (fixed): s09/s10/s11 opened on empty ground ~0.3 s → entrances −0.2 s.
- P1 (fixed): dead_stretch in s09/s10 → wire pulses + landing rule; s11
  deliberate hold carries data-hold.
- P2 (kept): 135 cut night-3D → light-2D is abrupt; voice ("How?") bridges it.

## Weakest shot / more time

- t92 macro approach is dark (storm); coils carry t95+. With more time: faint
  cool fill on the VAV box in 88-95.
- People are faceless capsules; fine at film distance, not for close-ups.

## Warnings kept on purpose

- gsap_repeated_fromto_without_baseline (ov-hook object counters): plain
  objects, explicit from, seek-safe.
- composition_file_too_large (world3d): single module scope is load-bearing.
- canvas_content_at_edge: full-bleed 3D by design.
- text_box_overflow info (ov-title), container_overflow info (starfield):
  verified harmless in frames.

## Pass 2 + delivery (2026-11-02)

- Re-check after fixes: ok:true, 0 errors, same 3 kept warnings.
- Verified: 2D entrances land by +0.3 s; VAV label shrinks/fades on approach
  (85 readable, 88 exiting, 90 gone).
- Draft render (GPU, 47 s): 170.0 s, 1080p30, h264+aac. Mix analysis: bed had
  a 9 s near-silent night passage → score normalized (dynaudnorm + loudnorm
  I=-26); re-render: integrated -17 LUFS, true peak -2.8, 0 clips, one 0.6 s
  dip left (accepted as phrasing breath).
- Decoded frames match snapshots (encode integrity confirmed).
- Delivery: renders/demo-final.mp4 (71.5 MB, 170.0 s). Contact-sheet cells can
  lag on late 2D scenes — 2D verified via individual PNGs instead.

## Revision v2 (2026-11-03) — user feedback pass

Feedback: pacing feels slow; too few background buildings; no interior room
overviews with people reactions (too cold / cold / comfort / hot / too hot);
captions not beautiful.

- City: 110 instances kept, pushed to r 315-555 and capped at h 58 so the
  skyline reads dense behind the hero without dwarfing it at the night pull-back.
- Interior: camera now dives into the floor-2 hero room (keys t33/t38/t44);
  five figures carry the five comfort states with body colour (blue->neutral->
  red), shiver/hunch/sway micro-motion and warm emissive near comfort, then
  converge to neutral over 40-44 s under an ALL COMFORTABLE banner. Zone cells
  fade to ~7% during 30-47 s so the room is actually visible.
- Labels rebuilt: canvas width is measured per text (no more fixed 512 px
  sprite), so neighbouring room labels no longer paint over each other;
  all five state labels read white at t34-41 snapshots. ALL COMFORTABLE is
  gated to 42.5-46.5 s and to the x-ray factor, so it no longer leaks onto
  the facade after the camera exits.
- Captions: generated karaoke track (66 groups after merging standalone
  punctuation tokens), one group visible at a time, dark pill
  rgba(10,16,24,0.92), Inter 600 56 px, amber #e8590c spoken-word fill.
  Registry block caption-pill-karaoke was evaluated and not adopted (it
  builds timed DOM from script and uses a light pill off-brand); its
  gap-based word spacing and hard group kill informed this implementation.
  Words are inline-block with 0.26 em margins because inter-span text nodes
  were dropped by the captions runtime (words rendered joined).
- Pacing: opening push-in key moved 20 s -> 14 s and closer; night eco HUD
  (ov-eco, 105-133 s) adds three staggered readouts.
- Gates: check ok:true, 0 errors (3 kept warnings, all previously
  dispositioned); snapshots at 3/12/34/37/41/44/47.22/59/100/120/128/160.56
  inspected; draft render 170.0 s, mix -17 LUFS, peak -2.8 dBTP, one 0.6 s
  breath at 60.9 s (kept, phrasing).
- Delivery v2: renders/demo-final.mp4 overwritten (76.6 MB, 170.0 s,
  1080p30 h264+aac). Decoded frames at 37/95/130 match snapshots; t=95
  extracted twice, identical md5 (determinism holds).

## Revision v3 (2026-11-03) - dial + control rooms

Feedback: no dial to read what is happening; unclear what the policy can
and cannot control - show it in separate rooms.

- Day dial (ov-dial, mounted 0-135 s, top right): analog clock with hour and
  minute hands, hourly digital readout, outdoor temperature and weather
  state on one dark instrument card, tagged SIMULATED DAY. Day keys are
  anchored to the voice beats (05:42 dawn, ~06:30 "Six AM" startup, ~12:30
  "Noon" heat wave, 16:00 cold snap, evening/night after). No registry clock
  exists (gap logged); mk-usage-arc is a percentage gauge, not a clock.
- Two new floor-2 rooms next to the hero room, camera glides 44-62 s:
  WE CONTROL (ceiling VAV box, damper blade sweeping open 45-55 s, glowing
  reheat coil, falling airflow cones, label "damper - reheat - setpoint")
  and WE DON'T CONTROL (sun-blasted window with heat shimmer plane, a person
  walks in uninvited 48-55 s, label "sun - weather - people"). X-ray window
  extended to 58 s and zone cells stay faded until 59 s so both rooms read.
- Check: ok:true, 0 errors across lint/runtime/layout/contrast/motion/audio.
  New kept warnings: rotation_pivot_drift on the minute hand (a clock hand
  cannot spin about its bbox centre - false positive by construction) and
  text_below_floor on the 20 px SIMULATED DAY tag (service label, not copy).
  Pre-existing craft line_unread notes on the end card re-checked: all end
  lines are static from 166.6 s, 3.4 s of read time; kept.
- Draft frames at 8/49/53/72/100/120/130 s inspected from the rendered file.

## Revision v4 (2026-11-03) - White Model trial

Applied the deep-research winner (Projection on White Model) as a material
pass over the existing film, grammar only, no engine port:
hero building, roof plant and the 110-block city are now matte white
(#F3F3F1 facade / 0xEDEDEA sides / 0xE3E3DE city, roughness ~0.9) on a dark
satin ground (0x4A4F55 day -> 0x22262C night). Colour is reserved for data:
comfort-state figures, zone cells, lit windows, labels, the day dial and
captions. Interior walls/desks lightened to match. Voice, timings, camera
and 2D scenes unchanged. Check ok:true, 0 errors; frames at 3/25/37/53/72/
100/120/141/149 s inspected from snapshots and the draft render.
- Fix (user report): the red boiler box on the roof read as an unexplained
  red object in the white model - repainted to the model material; the hero
  VAV damper blade is now neutral grey and only glows orange inside its
  labelled macro window (85-100 s).
- Fix (user report): control rooms read as labels only. Camera now closes
  in on each room (49-58 s); each room gets a glowing outline frame (amber
  for WE CONTROL, grey for WE DON'T CONTROL), a front damper blade that
  glows as it opens, bigger falling airflow cones, and explicit chips:
  SETPOINT 22 C vs NO ACTUATOR. Root cause of the first invisible-frames
  attempt: the zone cells write depth, hiding transparent props inside
  their volume - frames and cones now render with depthTest off.

## Revision v5 (2026-11-03) - scene variety

User: not everything in one scene; something new every ~5 s.
- New full-screen 2D finance scene (6.2-20.4 s, under the voice-over about
  energy share): a facility manager watches bills fly into an HVAC furnace,
  a dollar counter burns up, 40-60% lands, he scratches his head, and a
  calendar flips back to SINCE 1980. The 40% hook chip now plays only over
  the dawn exterior (0-6.2 s); the 60% and 15% hook chips were retired
  because the finance scene carries those numbers.
- s09-arch rebuilt from abstract cards into an equipment showcase on white:
  air handler (spinning fan), VAV damper (sweeping blade), reheat coil
  (glowing), boiler (flame) - each with a live slider and a percentage the
  policy sets - plus a read-only sensor strip ("listens, never rewrites").
- Check ok:true, 0 errors. New kept warning: tile sub-labels at 26 px
  (captions, not body copy).

## Goal iteration 1 - hook promise (Section 10: 0-6 s)

Gap: the hook stated stakes (37% carbon) but never promised the payoff.
Fix: ov-hook gained a THE PAYOFF card (1.2-5.8 s): "5.2x lower loss" with
animated bars 0.06439 -> 0.01236 (real numbers from docs/ladder.md).
Check ok:true, 0 errors; t=3 snapshot: building centered, stakes bottom
left, promise top center-right, dial clear of both. Verdict: pass.

## Goal iteration 2 - re-engagement spikes (Section 10: ~55 s, ~115 s)

Gap: no planned attention-reset spikes.
Fix: ov-spike1 (52-61 s) "A storm is coming. Will the policy hold?" placed
low-center so it never covers the room labels; ov-spike2 (111-120 s)
"Peak load at night. Watch it answer." over the night pull-back.
Check ok:true, 0 errors; both snapshots read clean with the dial, eco HUD
and captions. Verdict: pass.

## Goal iteration 3 - finance ladder + honesty tag (Section 10: 6-20 s)

Gap: finance scene had a single counter, no stakes ladder, and its dollar
numbers carried no honesty label.
Fix: ladder line "$132/d -> $4k/mo -> $48k/yr" (internally consistent:
132*365 = 48,180) under the counter, plus an ILLUSTRATIVE tag top-left.
Two placement misses fixed on the way (ladder first crossed the character,
then the caption pill). Check ok:true, 0 errors; draft frames at 3/12/55/
115/167 s inspected from the rendered file; mix -17 LUFS, peak -2.8 dBTP.
Verdict: pass.

## Goal verification - second consecutive clean pass

- check ok:true, 0 errors; all warnings previously dispositioned (large
  composition files, opacity-only fromTo baselines in ov-hook/s09-arch,
  minute-hand pivot drift, sub-28 px service labels).
- Snapshot sweep at 3/12/25/37/52/59/72/100/115/128/141/149/160.56/167 s:
  no remarks. Section 10 audit: promise in first 5 s, stakes ladders,
  spikes at 52-61 and 111-120 s, curiosity loops opened/closed by the
  spikes, abrupt end card with a one-line CTA.
- Delivery render verified: 170.000 s, h264+aac 1080p30, t=95 md5 stable
  across two extractions, mix -17 LUFS / -2.8 dBTP.

## Revision v6 (2026-11-03) - no slow zooms, comfort split out

User: slow zooms invite skipping; control rooms should take the slot where
comfort states were shown; people states deserve their own detailed scene.
- 3D restructured: the hero comfort room lost its five figures/labels (now
  a plain shell); the camera dives straight to the control rooms (30-41 s)
  and the disturbance room (41-52 s) under the "damper and reheat valve"
  voice line. All control/disturbance windows shifted ~12 s earlier.
- Slow legs split with intermediate keys: 0-7-14, 14-24-30, 70-82-95,
  105-112-120, 120-128-135 - camera now changes direction every 6-13 s.
- New 2D scene s-comfort (58-70 s, silent music beat): five detailed cards
  with faces (frown/flat/smile), shiver, sway, sweat drops, snowfall and
  heat shimmer, then all five converge to neutral with green checks.
- ov-spike1 shortened to 52-58 s so it never covers the comfort scene.
- Check ok:true, 0 errors; frames at 32/38/44/60/61/66/68/82/112 s checked.

## Revision v7 (2026-11-03) - control rooms become 2D s-control, new s-how

User: the "what we control / what we don't" beat should be the separate
scene from the end, not 3D rooms.
- 3D control/disturbance rooms deleted from world3d (shells, VAV, coil,
  labels, frames, chips, walk/heat animation); camera keys 35/41/47
  retargeted to an exterior x-ray sweep; roomFade removed (zone cells stay
  visible through the 30-52 cutaway).
- New 2D s-control mounted 30-45 s (z 9, over 3D): four dial-safe actuator
  tiles (AHU, damper, reheat, boiler) + dark strip with read-only sensors
  and the don't-control line (sun / weather / people). Title matches the
  user wording: "What we control, what we don't".
- s09-arch removed; new 2D s-how mounted 135-147 s for V09: Read (zone
  grid + live temp) -> Think (neural net, 3 activation waves) -> Command
  (AHU/VAV/boiler bars) + "same brain, any building." strip.
- Fixes: caption/strip collisions cleared (s-control strip up + shorter
  line 2; s-how strip up 12 px); net waves rewritten set/to (no lint
  overlap warnings); read label raised to the 28 px type floor.
- Trade-off note: guidelines Sec.2 beat 3 (x-ray reveals controlled
  elements) now reads via exterior cutaway + zone cells + AHU/zone labels
  (visible 45-52 s) plus the damper macro at 85-100 s; room interiors are
  gone per explicit user direction.
- Check ok:true, 0 errors; frames at 32/38/44/47/50/136/141/146 reviewed.

## Revision v8 (2026-11-03) - s-control instant start + state-NN-control band

User: run everything at once; the scene must open immediately with
control/don't-control; add neuron activation state -> NN -> control.
- s-control rewritten: tiles enter near-simultaneously (0.2-0.5 s) and all
  actuators spin up together; strip with BOTH lines visible from 0.8 s.
- New pipeline band (600-720 px): STATE 39 zone sensors -> 3-5-4 neural net
  with 4 looping activation waves (green in, amber through/out) ->
  CONTROL damper/reheat/fan/boiler. Tiles compacted to 336 px to fit.
- Strip moved to 736-844 px; caption clearance re-verified at 31/34/38 s.
- Check ok:true, 0 errors; frames at 31/34/38/44 reviewed.

## Revision v9 (2026-11-03) - s-control cut moves to 26 s, slow reveal restored

User: not faster tiles — cut to the scene sooner; the silence after V03
"run them all" (~26 s) was dead air.
- s-control mount 30 -> 26 s (ends 41 s, aligned with V04 voice end 40.6 s);
  V04 voice/captions untouched — the scene now builds during the former
  silence and is complete when the voice starts at 30.4 s.
- Internal timing reverted to the staged cascade (tiles 0.4-2.2, NN band
  2.6 with 4 waves from 3.0, strip 3.2, don't-control line 4.2).
- ov-title trimmed 20-30 -> 20-26 s with fade at local 5.5; 2D->3D whoosh
  au-w1 moved 45 -> 41 s to the new out-cut.
- Check ok:true, 0 errors; frames at 24/26/27/30/38/41/42 reviewed.

## Revision v10 (2026-11-03) - s-control split into building/equipment phases

User: at "This is our building" the equipment tiles were already showing —
picture/voice mismatch (it predated v9, the early cut only exposed it).
- s-control is now two phases split at the sentence boundary 34.42 s
  (caption-verified: "air handlers." -> "Every room has a damper").
- Phase A (local 0-8.4): THE BUILDING card — 3 floor slabs with 39 zone
  cells lighting in cascade, 4 AHU blocks with spinning fans, count-ups
  3/39/4, dark strip + LIVE badge; exits 7.9-8.3.
- Phase B (local 8.5-15): THE EQUIPMENT — title swap + 4 tiles fast
  cascade, sliders, NN band (2 waves), sensors/don't-control strip.
- LIVE badge moved from card corner (overlapped AHU4) into strip-A.
- Check ok:true, 0 errors; frames at 28/33/35/37/40 reviewed.

## Revision v11 (2026-11-03) - s-comfort fixes read as real state changes

User: checks landed while faces/tags/effects stayed put — the fix was
unreadable (only a subtle body recolor happened).
- Each fixed card now: face warms to comfort tone + frown/flat mouth swaps
  to a smile, tag crossfades to green COMFORT, bottom band flips green,
  then the check pops (same 7.6-9.6 stagger).
- Effects retimed to end before the fixes (snow 6.0, heat 7.0, sweat 7.2)
  with an explicit dissolve at 7.2 — no frozen drops beside fixed cards;
  shivers extended to live until their card's fix (7.2/7.7).
- Check ok:true, 0 errors; frames at 60/64/66/68 reviewed.

## Revision v12 (2026-11-03) - captions lowered

User: captions sat too close to scene content.
- .cap-g bottom 92 -> 56 px (36 px lower, still inside action-safe).
- Check ok:true, 0 errors; frames at 8/33/76/150/166 reviewed — clear of
  finance art, control strips, metrics numerals and the end card.

## Revision v13 (2026-11-03) - visible 6 AM equipment startup

User: at "Six AM, the building wakes" the voice described spinning up
equipment but the 3D showed nothing starting (the damper wave ran at 52+,
after the words; AHU fans were static smooth cylinders).
- AHU fans rebuilt as spinner groups with a contrasting blade bar that
  spins up 46-52 s (accelerating ramp, then steady) with an amber emissive
  pulse 46-58 s; all four roof units visibly come alive.
- Morning damper wave retimed 52-66 -> 47-58 s so zones sweep blue->amber
  under "dampers sweep open ... every zone glides into its comfort band".
- Hero VAV blade ramps 46-52 s (was 45-70) with an emissive highlight;
  AHU/zone labels extended to 52 s.
- Camera keys 47/53 retargeted: push to the roof for the spin-up, drift,
  then pull back out (no leg over 7 s).
- JS syntax-checked (node --check), check ok:true, 0 errors; frames at
  46/48/50/52/54 reviewed.

## Revision v14 (2026-11-03) - dial time/temp resynced to events

User: dial temperature did not match what was happening.
- Root causes: "Noon" spoken at 70.4 s while the dial showed 11:00/22 C;
  snow/breath played at 16 C; real cold (3-4 C) arrived only at 103 s.
- New schedule: 12:00 + 29 C land exactly on V06 "Noon" (70.4 s); 31 C
  peak 80-88 s; crash to 4 C at 88 s together with the COLD SNAP wx flip,
  clouds and rain; 3 C with NIGHT from 112 s. The 16 C step is deleted.
- Morning (06:00 Six AM) and evening (17:00/18:00 + "Evening" at 105.4 s)
  boundaries preserved; hour-hand tweens recomputed so hands hit exact
  hours at every new label boundary (minute hand stays decorative).
- Check ok:true, 0 errors; dial crops at 70/71/80/88/95/105/112 reviewed.

## Revision v15 (2026-11-03) - s-meta "Built with Meta AI" beat + 178 s

User: claim Use-of-Meta-AI (simulator scores, Muse wrote the rest —
rules + nets + this video) and open the tech spec a little.
- New 8 s scene s-meta at 147-155: real occ_gated7.py excerpt types into
  a code pane, chips RULE-BASED / NEURAL NETS / PYTORCH / THIS VIDEO.
  Stack verified in repo (torch BC, d=64 transformer, 72-step window,
  occ_*/pid rule stack) — no invented names. No model version claimed
  (unverifiable) — "Muse" + "Meta AI" only.
- New V09B voice (Kokoro am_adam, 6.23 s): "Meta AI did the building:
  rules and neural nets in PyTorch — written with Muse, down to this
  video." Captions capg-66..68 aligned from Parakeet word stamps.
- End block shifted +8 (s10 155, s11 168, s12 173, V10-V12, c5-c7, all
  captions >= 147); root 178 s. Bed extended 172 -> 178 s from sibling
  segment _seg6 (level-matched -9.2 dB, 2.5 s crossfade; tail -24.9 dB
  vs -26.3 dB) — same generative family, residual seam risk noted.
- Retention doctrine intact: spikes still at thirds (52-58, 111-120 of
  178), last 15 s = claim + end card, abrupt ending.
- Longer root reshuffled the check sample grid and exposed two latent
  issues in untouched scenes (both fixed): caption pill covered the
  fin-pct sub-line at 9.889 s (fin-pct 110 px -> 96 px, top 760 -> 742);
  dial SIMULATED DAY tag contrast 1.54:1 through the translucent card
  (card alpha 0.86 -> 0.95, tag color solid #d7dee5).
- s-meta polish from snapshots: code line 07 shortened to
  `enumerate(ctrls)` (overflowed the card), code line-height 44 -> 40
  (line 10 was clipped: 82 + 10*44 = 522 > 500 card height).
- check15 ok:true, 0 errors, 0 warnings; reviewed 148/151/154 (s-meta),
  160 (s10), 170 (s11), 176 + 177.75 (s12 end card).
- Delivery renders/demo-final.mp4: 56.3 MB, 178.000 s, 1920x1080@30,
  h264+aac, 5340 frames, md5 89805cf154a9be898222ca1eb329a19c.
  Integrated -17.1 LUFS, true peak -2.8 dBTP. Decoded frames at 151 /
  177.5 match snapshots. Mix energy healthy on the Meta beat
  (147-155: mean -19.2 dB, peak -3.0) and across the bed seam
  (170-178: mean -21.2 dB, peak -3.0) — no dropout.

## Revision v16 (2026-11-03) - own GitHub URL on the end card

User: show our own repo link instead of the starter kit.
- s12-end URL swapped to github.com/Hanbiike/smart-buildings (accent on
  the org, same pattern as before). Voice never says the URL, so no
  re-record. check16 ok:true, 0/0; decoded frame at 176 confirms.
- Delivery renders/demo-final.mp4 re-rendered: 56.3 MB, 178.000 s,
  md5 17fbf4923361164d69445dc1babe8c81.
