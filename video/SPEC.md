# SPEC.md — "One building, one day" (storyline A)

## Template

GOAL:    A hackathon judge feels "this thing runs buildings" and remembers
         one-policy-across-weather; then scores the demo high and opens the repo.
OUTPUT:  HyperFrames, 1920x1080, 30 fps.
TIME:    170 s x 30 fps = 5100 frames. Music target 96 BPM (beat 0.625 s =
         18.75 f); true beat grid from `beats --json` after the track
         resolves; layout changes on the bar (4 beats = 2.5 s).
LOOK:    2D ground #EDF0F3 flat, ink #11161B, accent thermal amber #E8590C
         (one word per shot max, payoff numerals, end-card mark only).
         Display Outfit 700 at 120 px, tracking -2%; body Inter 400 at 40 px;
         numerals JetBrains Mono 700, min 64 px. 3D medium: architectural
         model — warm key upper-left 3200 K, cool rim 6500 K, contact shadows,
         roughness >= 0.45, env intensity 0.5, bloom on brightest points only.
         Day-cycle sky rig: dawn #FFD9A0, noon #BFE0F2, dusk #F2A35E,
         night #101B2A. Grain 2%, vignette 6% via treatment pass (never CSS).
OBJECTS: office block mirroring building_train_1 (3 floors, 39 zones, 4 AHUs
         serving 10/10/9/10); AHU cabinets with fan + coil; per-zone VAV box
         with damper blade + reheat coil; HWS boiler + pump; low-poly
         occupants (comfort = warm rim + relaxed posture); sun/sky rig with
         heat shimmer, rain, snow emitters; HUD numerals (damper %, zone K,
         kW, comfort drift).
MOTION:  enter power2.out 12 f; camera power3.inOut; exit power2.in 10 f;
         data ticks linear; springs none (instrument feel, not playful).
         Pushes 12-18 f with motion blur; holds let equipment act on screen.
         One hero move per shot; secondary moves 50-70% travel, +3-6 f delay.
SHOTS:   (frame ranges @30fps; sound; handoff)
         S01 f0-150 hook aerial: pre-dawn push to facade, "40%" lands f60-120,
         music in + airy whoosh; push continues into S02.
         S02 f150-600 problem orbit: slow 90-degree arc, "40-60% HVAC" f240,
         "15% global energy" f420; low pulse under voice; arc ends facing
         entrance, match-cut to S03.
         S03 f600-900 persona morning: manager enters lobby f660, title "one
         policy, every building" f780-870; footstep foley + theme motif;
         title word "building" becomes the x-ray model of S04.
         S04 f780-1230 2D equipment (s-control over 3D): cut lands on V03
         "run them all"; actuator tiles + state-NN-control band + strips
         build during 26-30 s; voice joins the complete picture at f912;
         hard cut back to 3D morning at f1230.
         S05 f1350-2100 morning ramp: dive to AHU start f1410, dampers sweep
         open f1500-1800, zones settle to band; air rush + chime per settled
         floor; camera rises through the roof into noon.
         S06 f2100-2850 noon heat: shimmer exterior f2160, x-ray cooling,
         damper zoom macro f2400-2700 (hero zoom to controlled element);
         heat hum under, chime on hold; macro pulls back as clouds gather.
         S07 f2850-3150 WOW people payoff: cold-snap swing, reheat coils glow,
         occupants warm-rimmed f3000, "0.0 K drift" lands f3060; score swell,
         held 2 s; glow cools into evening.
         S08 f3150-4050 night eco: dusk-to-night timelapse f3150-3450, eco
         setback widening bands f3600, last occupant leaves comfortable f3900;
         theme resolves, crickets bed; roof pull-up transitions to 2D.
         S09 f4050-4410 how-it-works: 2D read→think→command diagram
         (zone grid + live temp → neural net waves → actuator bars);
         "same brain, any building." strip; last value holds into metrics.
         S09B f4410-4650 built-with-meta: real occ_gated7.py excerpt types
         in, chips (rule-based / neural nets / PyTorch / this video);
         V09B voice (Kokoro, Parakeet-aligned captions).
         S10 f4650-5040 metrics: numerals land one per ~110 f, chime each;
         last numeral holds into claim line.
         S11 f5040-5190 claim: "One policy. Every building." single line,
         amber on "Every"; full stop of music, voice only; line wipes into
         the mark.
         S12 f5190-5340 end card: mark performed (fill rises), CTA button +
         repo URL held 4 s min, team tag; resolve sting. Total 178 s.
BANNED:  all spec.md defaults, plus: product UI tilted in 3D; floating cards
         or side panels as scenes; particle decoration; glow/bloom beyond
         brightest points; three-cards-in-a-row; KPI grid; docked logo;
         "unlock/seamless/supercharge/game-changer/revolutionize/effortless";
         any invented dashboard or unmeasured speedup claim; metrics beat
         rendering with unlabelled non-real numbers.
QA:      f0 (never black, window + 40%), f120 (numeral readable), f600 (persona
         reads at 1080p), f1050 (x-ray labels >= 28 px), f2400 (damper macro
         fills >= 40% height), f3060 (0.0 K drift + warm occupants), f3450
         (night readable, no crushed blacks), f4140 (arch nodes legible),
         f4740 (metrics labelled), f4980 (mark + CTA held in safe boxes).

## Production notes (locked decisions)

- Voice: local Kokoro via `npx hyperframes tts` on RTX 5090 (HeyGen unsigned,
  non-interactive; user directed 5090 use). Script cap ~360 spoken words.
  Escalate engine choice only if review flags voice quality.
- Music: local MusicGen; SFX: bundled offline set. No paid items.
- Real-number sources: docs/presentation/02-competition-reference.md only.
  Policy performance metrics are TBD-PLACEHOLDER until measured from our repo.
