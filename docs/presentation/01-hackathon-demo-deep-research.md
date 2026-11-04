# Deep research: mandatory blocks of a 3-minute hackathon demo video

> Verbatim record of the `deep-research` workflow run of 2026-11-02.
> Research question: which blocks must a 3-minute hackathon demo video
> contain, based on major-hackathon winner history of the last 2 years.
> Claim statements and agent notes were translated from Russian to English
> per repo rule R9; evidence excerpts, source titles, and URLs are verbatim.
> Importance tags (`core_answer`, `supporting_context`, `edge_case`) are
> the workflow's own labels, preserved as is.
>
> Workflow terminal value: status `no_evidence`, sourceCount 0, with
> evidenceRefs pointing at the 5 resolved results below. That verdict is
> the aggregator's, not the per-result evidence status — each result
> carries its own excerpts and sources, recorded here as is.

## Winner pitch structure and timing (9 claims)

### R2-1 [core_answer]

Mandatory winner 3-minute pitch skeleton: Hook -> Problem -> Solution -> Demo -> Impact/What's next -> Close & team.

- > Many winning pitches follow almost exactly the Hook → Problem → Solution → Demo → Impact → Close skeleton, because it reliably fits the clock and lands every point judges score.
  Source: `https://raw.githubusercontent.com/wafi574/30-days-of-hackathon/main/23_Day_Pitch_Structure/23_pitch_structure.md`

### R2-2 [core_answer]

Standard 3-minute timing: 30s problem+solution, 90s live demo, 30s stack/how-it-works, 30s what's next.

- > In a three-minute slot: 30 seconds on the problem and solution, 90 seconds on the live demo, 30 seconds on your tech stack, 30 seconds on what comes next
  Source: `https://angelhack.com/blog/10-tips-to-help-you-rock-your-next-hackathon-demo/`
- > 0:00 to 0:30 Introduce the team, name the problem, and give the solution in one line.
  Source: `https://angelhack.com/blog/10-tips-to-help-you-rock-your-next-hackathon-demo/`
- > 0:30 to 2:00 Run the live or recorded demo, which is the part judges came for.
  Source: `https://angelhack.com/blog/10-tips-to-help-you-rock-your-next-hackathon-demo/`

### R2-3 [core_answer]

Demo is the centerpiece: winners show a working product within the first 60 seconds; the demo gets the most time; judges remember what they saw.

- > Winners show a working demo within the first 60 seconds. The demo is the centerpiece, not an afterthought. Judges remember what they saw, not what they heard.
  Source: `https://raw.githubusercontent.com/juliusbrussee/pitchr/main/knowledge/curated/hackathon-pitch-dataset/hackathon_winner_patterns.md`
- > Notice the **demo gets the most time** — judges want to see it *work*, not hear about it.
  Source: `https://raw.githubusercontent.com/wafi574/30-days-of-hackathon/main/23_Day_Pitch_Structure/23_pitch_structure.md`

### R2-4 [supporting_context]

Transition from the problem straight into the demo with no slide interlude ('let me show you'): the demo serves as proof of the solution.

- > Winners transition from problem statement directly into "let me show you" — the demo becomes the solution proof. No slide interlude between problem and working product.
  Source: `https://raw.githubusercontent.com/juliusbrussee/pitchr/main/knowledge/curated/hackathon-pitch-dataset/hackathon_winner_patterns.md`

### R2-5 [core_answer]

Impact/metrics block is mandatory in concrete numbers ('saves 3 hours per week', 'reduces error rate by 40%'); vague phrasing scores poorly.

- > Winners state impact in concrete terms: "saves 3 hours per week," "reduces error rate by 40%," "serves 10,000 students." Vague impact ("helps people") scores poorly.
  Source: `https://raw.githubusercontent.com/juliusbrussee/pitchr/main/knowledge/curated/hackathon-pitch-dataset/hackathon_winner_patterns.md`

### R2-6 [supporting_context]

Technical credibility block: name concrete tools, APIs, protocols, and architectural decisions, not 'we used AI'.

- > Winners mention specific tools, APIs, protocols, or architectural decisions. "We used GPT-4 with RAG on a vector DB" beats "we used AI."
  Source: `https://raw.githubusercontent.com/juliusbrussee/pitchr/main/knowledge/curated/hackathon-pitch-dataset/hackathon_winner_patterns.md`

### R2-7 [supporting_context]

Video ending: concrete CTA (try it, link, vote) plus a short team intro; a generic 'thanks for listening' loses momentum.

- > Winners end with a specific ask: "Try it at hack.app," "Star our repo," "Vote for us at booth 7." Generic endings ("thanks for listening") lose momentum.
  Source: `https://raw.githubusercontent.com/juliusbrussee/pitchr/main/knowledge/curated/hackathon-pitch-dataset/hackathon_winner_patterns.md`

### R2-8 [edge_case]

Major-hackathon 2025 winner example (RiskWise, Best Overall Microsoft AI Agents Hackathon): real enterprise problem + sophisticated architecture + polished UI with visualizations + clear business value narrative.

- > **Why it won:** End-to-end excellence — real enterprise problem (supply chain risk), sophisticated multi-agent architecture, polished React/Next.js UI with visualizations, heavy Azure service integration, clear business value narrative.
  Source: `https://slop.sus.cat/tools/hackathons-2026/webapp/reports/past-winners-microsoft.html`

### R2-9 [supporting_context]

Three minutes is the most common demo slot (range 2-5 minutes); scope is one demo path: one user, one action, one visible result.

- > Most hackathon demos run between two and five minutes, with three minutes the most common slot
  Source: `https://angelhack.com/blog/10-tips-to-help-you-rock-your-next-hackathon-demo/`
- > Scope to one demoable path. One user, one action, one visible result. Anything that never reaches the screen scores nothing.
  Source: `https://angelhack.com/blog/5-steps-from-idea-to-hackathon-winning-app/`

Agent notes:

> Takeaway for the 3-minute demo: mandatory blocks are hook (20s), problem (25-30s), one-line solution (25-30s), live demo (60-90s, centerpiece), tech implementation/stack (30s), impact metrics + what's next (20-30s), CTA + team (15s). Patterns derived from analysis of 60 winners (NASA Space Apps / ETHGlobal / TreeHacks / HackMIT and others) plus AngelHack operating experience (hundreds of hackathons). Separately confirmed by the RiskWise example (Microsoft AI Agents Hackathon 2025).

Sources:

- Hackathon Winner Patterns — 10 patterns from 60 verified winners [reported_analysis, unknown (repo file, fetched 2026-11-02)]
  `https://raw.githubusercontent.com/juliusbrussee/pitchr/main/knowledge/curated/hackathon-pitch-dataset/hackathon_winner_patterns.md`
- Day 23 — The 3-Minute Pitch Structure [reported_analysis, unknown (repo file, fetched 2026-11-02)]
  `https://raw.githubusercontent.com/wafi574/30-days-of-hackathon/main/23_Day_Pitch_Structure/23_pitch_structure.md`
- 10 Hackathon Demo Tips From Behind The Scenes - AngelHack [reported_analysis, 2026-08-25 (Last Updated)]
  `https://angelhack.com/blog/10-tips-to-help-you-rock-your-next-hackathon-demo/`
- How To Win A Hackathon: 5 Steps From An Idea To A Great Demo - AngelHack [reported_analysis, 2026-09-08 (Last Updated)]
  `https://angelhack.com/blog/5-steps-from-idea-to-hackathon-winning-app/`
- Microsoft Hackathons — Past Winners & Judging Analysis [reported_analysis, 2026-03-19 (Last updated)]
  `https://slop.sus.cat/tools/hackathons-2026/webapp/reports/past-winners-microsoft.html`

## Screencast production (10 claims)

### R3-1 [core_answer]

A screencast with audio narration beats a marketing-style video: judges score demonstrated functionality, not production quality.

- > we recommend creating a screencast (a video screen capture with audio narration) because it can help judges, voters, and other users to understand exactly how your app works, without the gimmicks.
  Source: `https://oecsthinktankhackathon.devpost.com/forum_topics/4994-as-we-approach-pitch-and-demo-day-here-are-some-tips`
- > says a clear screencast beats a polished marketing video — judges score demonstrated functionality, not production values
  Source: `https://github.com/asrayg/hackathon-agent-skills/blob/HEAD/video-director/SKILL.md`

### R3-2 [core_answer]

The working-prototype demo is the center of the video: about 90 seconds of 3 minutes (0:30-2:00); at least half the runtime is real screen recording of the product.

- > Run the live or recorded demo, which is the part judges came for.
  Source: `https://angelhack.com/blog/10-tips-to-help-you-rock-your-next-hackathon-demo/`
- > keep the demo itself at roughly half the total runtime.
  Source: `https://angelhack.com/blog/10-tips-to-help-you-rock-your-next-hackathon-demo/`
- > ≥50% of runtime is real screen recording of the actual product.
  Source: `https://github.com/asrayg/hackathon-agent-skills/blob/HEAD/video-director/SKILL.md`

### R3-3 [core_answer]

Proof it works: show the product running on real inputs (localhost/emulator), not slides — a rough but working build scores above polished slides.

- > Show a demo of the product running by using some input. Show what it does as much as possible so judges can get an idea of what it does and how it helps.
  Source: `https://info.devpost.com/blog/6-tips-for-making-a-hackathon-demo-video`
- > Something runs live in front of the judges, even partially, and a rough build that functions will score above a polished set of slides describing one.
  Source: `https://angelhack.com/blog/10-tips-to-help-you-rock-your-next-hackathon-demo/`

### R3-4 [core_answer]

Bug fallback plan: pre-record a clean demo run locally and replace network calls with local fixtures/cache.

- > Record a clean run of your demo the night before, save it locally, and swap in local fixtures for anything that depends on a network call.
  Source: `https://angelhack.com/blog/10-tips-to-help-you-rock-your-next-hackathon-demo/`
- > Slow steps pre-loaded or cached, so agent responses and long queries don't create dead air
  Source: `https://angelhack.com/blog/10-tips-to-help-you-rock-your-next-hackathon-demo/`

### R3-5 [core_answer]

Scripted workflow: budget the video by seconds (intro/idea/components/demo/close), write a script, and rehearse it before recording.

- > what I do is take into consideration how many seconds I want to use for the introduction, how many seconds I want to use to explain the idea of the project, how many seconds I want to use to explain the components of the project, how many seconds for the video demo, and usually the last seconds is for closing or farewell.
  Source: `https://info.devpost.com/blog/6-tips-for-making-a-hackathon-demo-video`
- > Write out a script of what you'll say/show in your video, rehearse it before recording
  Source: `https://oecsthinktankhackathon.devpost.com/forum_topics/4994-as-we-approach-pitch-and-demo-day-here-are-some-tips`

### R3-6 [supporting_context]

Hook in the first seconds: elevator pitch and product essence immediately, like a movie trailer; never open with the team intro or logo.

- > treat the opening seconds of your video as you would a movie trailer or TV ad: pique your audience's interest.
  Source: `https://oecsthinktankhackathon.devpost.com/forum_topics/4994-as-we-approach-pitch-and-demo-day-here-are-some-tips`
- > Hook in the first 10 seconds — the problem at its most dramatic or the product's single best moment. Never open with the team intro or the logo
  Source: `https://github.com/asrayg/hackathon-agent-skills/blob/HEAD/video-director/SKILL.md`

### R3-7 [supporting_context]

Voiceover must reveal the invisible (architecture, pipeline, hard part), not narrate visible form-filling.

- > Judges watch and listen at the same time, so don't spend words describing what's already visible.
  Source: `https://angelhack.com/blog/10-tips-to-help-you-rock-your-next-hackathon-demo/`
- > Make sure your demo video is highly visual, and interactive (show your project in use).
  Source: `https://info.devpost.com/blog/6-tips-for-making-a-hackathon-demo-video`

### R3-8 [supporting_context]

Pre-recording checklist: seeded demo data (no empty state), bookmarked URLs, closed tabs and notifications, verified screen resolution and font size.

- > Demo data seeded and loaded so judges see a populated product rather than an empty state
  Source: `https://angelhack.com/blog/10-tips-to-help-you-rock-your-next-hackathon-demo/`
- > Screen resolution and font size checked so judges at the back can read what you're showing
  Source: `https://angelhack.com/blog/10-tips-to-help-you-rock-your-next-hackathon-demo/`

### R3-9 [supporting_context]

Editing is mandatory: cut failed parts, do multiple takes — the video does not have to be one take.

- > Edit out the messy stuff, if you can, and don't be afraid to do multiple takes.
  Source: `https://oecsthinktankhackathon.devpost.com/forum_topics/4994-as-we-approach-pitch-and-demo-day-here-are-some-tips`

### R3-10 [edge_case]

Recording setup: OBS Studio for desktop, the team's best microphone, YouTube upload in advance (2-3 hours) with judge access verified.

- > Use OBS Studio to record your desktop and audio.
  Source: `https://info.devpost.com/blog/6-tips-for-making-a-hackathon-demo-video`
- > get the team member that has the best microphone to record.
  Source: `https://info.devpost.com/blog/6-tips-for-making-a-hackathon-demo-video`
- > uploading your video could take anywhere from a few minutes to several hours or more.
  Source: `https://oecsthinktankhackathon.devpost.com/forum_topics/4994-as-we-approach-pitch-and-demo-day-here-are-some-tips`

Agent notes:

> All quotes read via WebFetch of full page texts. Devpost 6 Tips — advice from winners of 150+/25/10 hackathons; AngelHack — organizer of hundreds of hackathons, 2026 update; OECS/Devpost best practices — the platform's official old playbook (2015, still valid in substance); video-director — community guide synthesizing Devpost+MLH. Not covered: win counts tied specifically to screencasts (sources give practices, not statistics).

Sources:

- 6 Tips for making a winning hackathon demo video — Devpost [direct_evidence, unknown]
  `https://info.devpost.com/blog/6-tips-for-making-a-hackathon-demo-video`
- 10 Hackathon Demo Tips From Behind The Scenes — AngelHack [direct_evidence, 2026-08-25]
  `https://angelhack.com/blog/10-tips-to-help-you-rock-your-next-hackathon-demo/`
- Video-making best practices — Devpost forum (OECS ThinkTank Hackathon) [direct_evidence, 2015-12]
  `https://oecsthinktankhackathon.devpost.com/forum_topics/4994-as-we-approach-pitch-and-demo-day-here-are-some-tips`
- video-director SKILL.md — hackathon-agent-skills (community) [opinion, unknown]
  `https://github.com/asrayg/hackathon-agent-skills/blob/HEAD/video-director/SKILL.md`

## Scene order and timing templates (9 claims)

### R4-1 [core_answer]

Standard 3-minute demo scene order: hook -> problem -> demo -> technical depth -> close, with the demo taking the largest share of runtime.

- > Five beats, in order: hook (the problem, sharply stated) → problem (who it's for, why it matters) → demo (the app actually working -- the largest share of the runtime) → technical depth (the thing a technical judge needs to see, e.g. the real sponsor-tech call site) → close (name, tagline, live URL).
  Source: `https://raw.githubusercontent.com/rogerjeasy/win-hackathon/main/skills/demo-video-script/SKILL.md`
- > Just focus on the most important information—what your project does, how it solves the problem, and how it works.
  Source: `https://info.devpost.com/blog/6-tips-for-making-a-hackathon-demo-video`

### R4-2 [core_answer]

Proven 'golden script' 3-minute timing: 0:00-0:30 problem hook, 0:30-1:15 solution in action, 1:15-2:00 killer feature (wow moment), 2:00-2:40 architecture, 2:40-3:00 impact and next steps.

- > 0:00–0:30: Problem Hook & Human Pain Point. 0:30–1:15: Core Solution in Action (First click to result). 1:15–2:00: Killer Feature ("The Wow Moment"). 2:00–2:40: Technical Architecture & Differentiation. 2:40–3:00: Impact & Next Steps.
  Source: `https://raw.githubusercontent.com/teamddpratham/hackathon-os/main/.agents/skills/demo-engineering/SKILL.md`

### R4-3 [core_answer]

Alternative demo-heavy template: 0:00-0:20 hook with numbers, 0:20-0:45 live persona/pain, 0:45-2:15 live demo (90s, wow moment by 1:45), 2:15-2:40 how it works, 2:40-3:00 close.

- > | 0:00–0:20 | Hook — the problem, with numbers | 2–3 sourced statistics... | 0:20–0:45 | The face | YOUR real business/persona and their exact pain... | 0:45–2:15 | Live demo — the golden path | One flow, rehearsed camera path. The named wow moment lands by 1:45... | 2:15–2:40 | How it works | One breath: agentic loop + policy-bounded autonomy + sponsor tools used... | 2:40–3:00 | Close |
  Source: `https://raw.githubusercontent.com/supawichza40/encoder-xero-app-and-agent-hackathon/main/template/PITCH.md`

### R4-4 [core_answer]

Show/tell balance: 'show, then tell' — a real working product on screen beats slides; reach a working thing on screen fast.

- > Show, then tell. Real demo footage beats slides. Get to a working thing on screen fast.
  Source: `https://raw.githubusercontent.com/jazztinn/homerun/main/.claude/skills/hackathon-pitch-video/SKILL.md`
- > Make sure it's a demo! Show your project or app in action and include more information in your text description if needed.
  Source: `https://info.devpost.com/blog/6-tips-for-making-a-hackathon-demo-video`

### R4-5 [supporting_context]

Open with the pitch and a quick project overview before details; key information goes in the first seconds.

- > Start with your pitch and a quick overview of your project before diving in and presenting it. Include the problem you're solving (if any), frameworks/libraries you used, etc.
  Source: `https://info.devpost.com/blog/6-tips-for-making-a-hackathon-demo-video`
- > Start the video with your elevator pitch. Explain what your app does and how it addresses the hackathon in the first few seconds of your video.
  Source: `https://info.devpost.com/blog/6-tips-for-making-a-hackathon-demo-video`

### R4-6 [supporting_context]

When scripting, budget about 150 words per minute of narration plus about 15% slack for pauses and demo; every beat gets a time budget and word count.

- > Every beat gets a time budget and a word count (~150 wpm narration). Total spoken words must fit the target length with ~15% slack for pauses and demo time.
  Source: `https://raw.githubusercontent.com/jazztinn/homerun/main/.claude/skills/hackathon-pitch-video/SKILL.md`
- > what I do is take into consideration how many seconds I want to use for the introduction, how many seconds I want to use to explain the idea of the project, how many seconds I want to use to explain the components of the project, how many seconds for the video demo, and usually the last seconds is for closing or farewell.
  Source: `https://info.devpost.com/blog/6-tips-for-making-a-hackathon-demo-video`

### R4-7 [supporting_context]

Real submission storyboard example: 0:00-0:30 live control-loop demo, 0:30-1:30 metrics dashboard, 1:30-2:30 architecture and guardrails, 2:30-2:50 wrap-up; target runtime 2:50 with a 3:00 cap.

- > Target Video Duration: 2 minutes 50 seconds (Maximum Cap: 3:00) ... | 0:00 – 0:30 (30 sec) | 1. Live Control Loop Demo | ... | 0:30 – 1:30 (60 sec) | 2. Dashboard Walkthrough | ... | 1:30 – 2:30 (60 sec) | 3. Closed-Loop Architecture & Guardrails | ... | 2:30 – 2:50 (20 sec) | 4. Wrap-Up & Conclusion |
  Source: `https://raw.githubusercontent.com/pritish3110/honeywell_hackathon/main/docs/DEMO_VIDEO_SCRIPT.md`

### R4-8 [supporting_context]

Show technical depth briefly without getting stuck: some technical aspects but no overload; do not cram in too much information.

- > Spend a little time showing the technical aspects of what you worked on but don't get too bogged down in the technical aspects.
  Source: `https://info.devpost.com/blog/6-tips-for-making-a-hackathon-demo-video`
- > Make your presentation clear and concise. Don't try to pack in way too much information or speed up the audio to fit within a time limit.
  Source: `https://info.devpost.com/blog/6-tips-for-making-a-hackathon-demo-video`

### R4-9 [edge_case]

Give every shot an explicit seconds budget summing to the cap (usually 180s); check platform video requirements as a checklist and write each into a specific shot.

- > Write each shot's seconds deliberately, not as an afterthought: the demo beat should be the largest single allocation... Read recon.json.submission_requirements for every video-related hard requirement and write each one into a specific shot, then verify after editing that the shot actually says it.
  Source: `https://raw.githubusercontent.com/rogerjeasy/win-hackathon/main/skills/demo-video-script/SKILL.md`

Agent notes:

> All 6 sources read in full. Consensus on scene order (hook -> problem -> demo -> tech depth -> close) across all templates; the demo is the longest block (45-90s); the wow moment lands no later than mid-minute-two. Only the first-45-seconds timing details vary.

Sources:

- 6 Tips for making a winning hackathon demo video — Devpost blog (quotes from multi-time winners) [direct_evidence, unknown]
  `https://info.devpost.com/blog/6-tips-for-making-a-hackathon-demo-video`
- win-hackathon: demo-video-script SKILL.md — 5-beat sub-3-minute video structure [reported_analysis, unknown]
  `https://raw.githubusercontent.com/rogerjeasy/win-hackathon/main/skills/demo-video-script/SKILL.md`
- hackathon-os: demo-engineering SKILL.md — 3-Minute Golden Script timing [reported_analysis, unknown]
  `https://raw.githubusercontent.com/teamddpratham/hackathon-os/main/.agents/skills/demo-engineering/SKILL.md`
- encoder-xero PITCH.md — 3-minute pitch clock table + wow moment rule [reported_analysis, unknown]
  `https://raw.githubusercontent.com/supawichza40/encoder-xero-app-and-agent-hackathon/main/template/PITCH.md`
- homerun: hackathon-pitch-video SKILL.md — timed script, 150wpm, show-then-tell [reported_analysis, unknown]
  `https://raw.githubusercontent.com/jazztinn/homerun/main/.claude/skills/hackathon-pitch-video/SKILL.md`
- Honeywell hackathon DEMO_VIDEO_SCRIPT.md — real timed 2:50 storyboard example [direct_evidence, unknown]
  `https://raw.githubusercontent.com/pritish3110/honeywell_hackathon/main/docs/DEMO_VIDEO_SCRIPT.md`

## Judging rubrics (13 claims)

### R5-1 [core_answer]

A typical major-hackathon jury rubric is a reweighting of five axes: innovation, technical execution, impact, theme relevance, presentation.

- > Nearly every rubric is a reweighting of the same five axes: innovation · technical execution · impact · relevance to theme · presentation.
  Source: `https://github.com/crisiswastaken/skills/blob/HEAD/hackathon-idea-generator/references/judging-reality.md`

### R5-2 [core_answer]

TAIKAI recommends 6 judging criteria: creativity and innovation, technical execution, functional MVP, problem-solving and relevance, impact and potential, final pitch.

- > Criteria 1: Creativity and innovation Criteria 2: Technical execution Criteria 3: Functional MVP (Minimum Viable Product) Criteria 4: Problem-solving and relevance Criteria 5: Impact and potential Criteria 6: Final pitch
  Source: `https://taikai.network/en/blog/hackathon-judging`

### R5-3 [core_answer]

TAIKAI stresses: a functional MVP need not be perfect but must show core functionality and validate the concept; an idea is only as good as its execution.

- > a functional MVP is an important criterion to consider when evaluating hackathon projects. The MVP doesn't have to be perfect or feature-complete, but it should at least showcase the core functionality of the project and validate the concept.
  Source: `https://taikai.network/en/blog/hackathon-judging`
- > Technical quality is another core aspect when assessing a hackathon project. After all, an idea is only as good as its execution.
  Source: `https://taikai.network/en/blog/hackathon-judging`

### R5-4 [core_answer]

NASA Space Apps Challenge (2025 season) judges on: potential impact, creativity/innovation, challenge relevance plus NASA resource integration, presentation.

- > NASA Space Apps Projects will be evaluated on: Project Potential Impact (Does it address a major or a minor problem? (To what extent does it inspire others?)) Creativity (Does it demonstrate innovation? Novel Concept? Does it improve something in a meaningful way?) Relevance (Does the project address the challenge? Does it integrate NASA resources well?) Presentation (Does it convey solution, structure and propose a compelling idea to the audience?)
  Source: `https://luma.com/wudwutav`

### R5-5 [supporting_context]

The full NASA Space Apps rubric has five criteria: Impact, Creativity, Validity (scientific/technical soundness), Relevance, Presentation.

- > NASA Space Apps | Impact, Creativity, Validity, Relevance, Presentation | Validity = will it actually work scientifically; Relevance = responsive to the specific challenge submitted under
  Source: `https://github.com/crisiswastaken/skills/blob/HEAD/hackathon-idea-generator/references/judging-reality.md`

### R5-6 [core_answer]

ETHGlobal judges on: creativity, functionality, technical difficulty, impact; only work done during the event counts.

- > ETHGlobal | Creativity, Functionality, Technical difficulty, Impact | Technical difficulty is explicit; only work done during the event counts
  Source: `https://github.com/crisiswastaken/skills/blob/HEAD/hackathon-idea-generator/references/judging-reality.md`

### R5-7 [core_answer]

TreeHacks 2025 (Stanford) used three criteria: creativity, technical complexity, social impact.

- > TreeHacks 2025 | Creativity, Technical Complexity, Social Impact
  Source: `https://github.com/crisiswastaken/skills/blob/HEAD/hackathon-idea-generator/references/judging-reality.md`

### R5-8 [core_answer]

Official CGU Ethical AI Hackathon rubric: five equally weighted criteria at 20% each — socially responsible impact of innovation, ease of use/transparency, innovative technology, user safety and trust, presentation and storytelling (1-5 scale).

- > Socially Responsible Impact in Innovation | 20% | Ease of Use, Transparency, and Explainability | 20% | Use of Innovative Technology and Intelligent Adaptation | 20% | User Safety and Trust | 20% | Presentation and Storytelling | 20%
  Source: `https://research.cgu.edu/hackathon/home/judging-rubric/`

### R5-9 [core_answer]

HackTech 2025 jury (Cyprus, 100+ developers, 30,000 euro prize fund) scored four items: innovation, technical execution, presentation, real-world potential.

- > A mixed jury from IT and business, to government officials, firefighters, and emergency response representatives evaluated the projects on innovation, technical execution, presentation and real-world potential.
  Source: `https://pr.wvcjournal.com/article/HackTech-2025-Cyprus-Largest-Open-Hackathon-Wraps-Up-with-Strong-Focus-on-Wildfire-Solutions/6926fe2d476361ebf71e395f`

### R5-10 [supporting_context]

In online/async hackathons (Devpost, ETHGlobal partner prizes) the video submission may be all a judge ever sees; experienced winners spend 15%+ of total hackathon time on the submission artifact, structuring video at roughly 60% explaining / 40% demoing.

- > The video may be all a judge ever sees. One 22-time winner spends ≥15% of the entire hackathon on the submission artifact alone, and structures videos at roughly 60% explaining / 40% demoing.
  Source: `https://github.com/crisiswastaken/skills/blob/HEAD/hackathon-idea-generator/references/judging-reality.md`

### R5-11 [supporting_context]

A working demo carries heavy weight at every event — slides alone never win; creativity/originality appears in every rubric; presentation quality is a multiplier (the best hack loses if poorly demoed).

- > Working demo is weighted heavily at every event — slides alone never win. Creativity/originality appears in every rubric — novel framing matters. Technical depth is expected but must be balanced with usability. Real-world impact separates winners from technically impressive but impractical hacks. Presentation quality is a multiplier — the best hack loses if poorly demoed.
  Source: `https://github.com/juliusbrussee/pitchr/blob/HEAD/knowledge/curated/hackathon-pitch-dataset/hackathon_judging_criteria.md`

### R5-12 [supporting_context]

The Mercer/Mettl framework names key factors: innovation and originality, technical execution, challenge relevance, UX, scalability, impact and value, presentation clarity.

- > Innovation and originality: The strongest entries bring fresh thinking to the table. Technical execution: In a hackathon, time is short, so seeing a well‑built, stable, and efficient solution speaks volumes about a team's skill and discipline.
  Source: `https://blog.mettl.com/hackathon-judging-criteria/`
- > Impact and value: The best ideas change something for the better; whether that's adding revenue, streamlining a process, saving costs, or making a real social impact. Clarity of presentation: Even the most innovative ideas can be overlooked if they're poorly explained.
  Source: `https://blog.mettl.com/hackathon-judging-criteria/`

### R5-13 [edge_case]

MLH by default uses 4 equally weighted criteria — no weak axis.

- > MLH default | 4 equally weighted criteria | Balanced; no weak axis
  Source: `https://github.com/crisiswastaken/skills/blob/HEAD/hackathon-idea-generator/references/judging-reality.md`

Agent notes:

> Video-item prioritization by weight: innovation and technical execution usually carry the most weight (about a quarter each), impact/value about a fifth, pitch/presentation 10-20% but acting as a multiplier. Exact percentage weights of individual 2024 Devpost hackathons (e.g. Innovation 25% / Technical 25% / Impact 20% / Presentation 20%) appear in search snippets, but Devpost pages returned HTTP 403 on fetch, so these numbers are NOT included in claims as unconfirmed by direct reading. The NASA PDF Instructional Guidance for Judging downloaded as a binary and its text was not read directly — NASA criteria are instead confirmed by the RiverHacks x NASA Space Apps 2025 event page (October 2025) and the rubric summary table.

Sources:

- judging-reality.md — rubric archetypes (MLH, TreeHacks 2025, NASA, ETHGlobal, TAIKAI) [discussion, unknown, references 2025 events]
  `https://github.com/crisiswastaken/skills/blob/HEAD/hackathon-idea-generator/references/judging-reality.md`
- TAIKAI — Hackathon judging: 6 criteria to pick winning projects [reported_analysis, 2024-11 (article images dated Nov 20, 2024)]
  `https://taikai.network/en/blog/hackathon-judging`
- CGU Ethical AI Hackathon — Official Judging Rubric (5x20%) [direct_evidence, current (accessed Nov 2026)]
  `https://research.cgu.edu/hackathon/home/judging-rubric/`
- RiverHacks x NASA Space Apps Challenge — event page with judging criteria [direct_evidence, 2025-10-04 (event date)]
  `https://luma.com/wudwutav`
- HackTech 2025 wraps up (PR Newswire) — jury criteria [reported_analysis, 2025-11-26]
  `https://pr.wvcjournal.com/article/HackTech-2025-Cyprus-Largest-Open-Hackathon-Wraps-Up-with-Strong-Focus-on-Wildfire-Solutions/6926fe2d476361ebf71e395f`
- pitchr — hackathon_judging_criteria.md (ETHGlobal, NASA, LA Hacks, Hack the North) [discussion, unknown (compiled reference)]
  `https://github.com/juliusbrussee/pitchr/blob/HEAD/knowledge/curated/hackathon-pitch-dataset/hackathon_judging_criteria.md`
- Mercer/Mettl — Hackathon judging framework guide [reported_analysis, unknown (guide, accessed Nov 2026)]
  `https://blog.mettl.com/hackathon-judging-criteria/`

## Losing mistakes and stop-list (8 claims)

### R6-1 [core_answer]

A drawn-out intro kills the score: judges form opinions in the first seconds, so the video must open with the working product, not credentials, team intro, or problem setup.

- > AngelHack's 10 tips emphasize the 8-second hook: human attention span is 8 seconds. If the judge doesn't understand what they're seeing by second 8, they're mentally moving to the next submission. Front-load the payoff. Never open with credentials, team introductions, or problem statements — open with the tool running.
  Source: `https://raw.githubusercontent.com/tasumermaf/rhombic/main/scripts/VIDEO_CREATIVE_BRIEF.md`
- > Judges can see 30 to 100 demos in one day and form an opinion in the first 30 seconds, well before they see any code.
  Source: `https://www.masaischool.com/blog/how-to-win-an-ai-hackathon-in-2026-full-guide/`

### R6-2 [core_answer]

Slideware with no product is a common losing cause: the video showing a working product beats the video explaining a clever idea; slides must take about 30 seconds max.

- > The video that shows something working beats the video that explains something clever. Every time.
  Source: `https://raw.githubusercontent.com/tasumermaf/rhombic/main/scripts/VIDEO_CREATIVE_BRIEF.md`
- > Never put more than 10 to 15 words on a pitch slide. If the judge is reading, they aren't listening. Slides frame the problem, show one statistic, or hold a single idea, and then you move into the solution. In most pitches, slides should take 30 seconds max.
  Source: `https://raw.githubusercontent.com/idkwhatimd0ing/hackathonstarterkit/main/content/generated/md/blog/hackathon-pitch-mistakes-la-hacks.md`

### R6-3 [core_answer]

Missing problem and tech-first delivery: teams lead with the stack instead of the problem, show features instead of the main user flow, spending seconds on the problem and minutes on features nobody asked for.

- > Leading with tech stack instead of the problem. Judges connect with the "why" before the "how."
  Source: `https://raw.githubusercontent.com/yashkoparde/hackathon101/main/docs/06-COMMON-MISTAKES.md`
- > Spend roughly 30% of your pitch time on the problem and 70% on the solution. Most losing teams invert that ratio they spend five seconds on the problem and the rest showing off features nobody asked for.
  Source: `https://www.masaischool.com/blog/how-to-win-an-ai-hackathon-in-2026-full-guide/`
- > The judge doesn't need your landing page, auth screen, settings panel, or database schema unless one of them is central to the problem. Show the problem and the user flow that proves you solved it.
  Source: `https://raw.githubusercontent.com/idkwhatimd0ing/hackathonstarterkit/main/content/generated/md/blog/hackathon-pitch-mistakes-la-hacks.md`

### R6-4 [core_answer]

Running over time: judges do not finish videos longer than about 3 minutes; breaking the limit is one of the most penalized mistakes.

- > Devpost's canonical advice (the platform that runs most major hackathons) distills to six rules: start with what your project does in the first few seconds, write a script before recording, keep it under 3 minutes, focus on demo over explanation, respect the time limit, and get someone with a decent mic to record.
  Source: `https://raw.githubusercontent.com/tasumermaf/rhombic/main/scripts/VIDEO_CREATIVE_BRIEF.md`
- > Going over time. Judges will cut you off mid-sentence. A shorter, complete pitch is much better than a long one that gets interrupted.
  Source: `https://raw.githubusercontent.com/yashkoparde/hackathon101/main/docs/06-COMMON-MISTAKES.md`

### R6-5 [supporting_context]

Poor audio and flat, energy-free delivery lower the score; the person with the best microphone records, and delivery must be confident and energetic.

- > get someone with a decent mic to record
  Source: `https://raw.githubusercontent.com/tasumermaf/rhombic/main/scripts/VIDEO_CREATIVE_BRIEF.md`
- > Smile. It's obvious, but teams forget it the second judging starts. A flat, monotone delivery leaves the judge to figure out what's exciting on their own.
  Source: `https://raw.githubusercontent.com/idkwhatimd0ing/hackathonstarterkit/main/content/generated/md/blog/hackathon-pitch-mistakes-la-hacks.md`

### R6-6 [supporting_context]

On-camera stop-list: do not read slide bullets, explain the obvious, apologize for rough edges, walk through code line by line, dive into deep technical detail, or rush from nerves.

- > Don't read every bullet point on slides ... Don't explain obvious things ("this is the architecture diagram...") ... Don't apologize for anything ("this might be slow...") ... Don't show code line-by-line (boring for judges) ... Don't go into deep technical details (save for Q&A) ... Don't speak too fast (common mistake when nervous)
  Source: `https://github.com/sgharlow/ai-matcher-aws-hackathon/blob/HEAD/docs/FINAL_SUBMISSION_STRATEGY.md`
- > Apologizing for unfinished parts. Present what works confidently. Frame gaps as an intentional future roadmap, not excuses.
  Source: `https://raw.githubusercontent.com/yashkoparde/hackathon101/main/docs/06-COMMON-MISTAKES.md`

### R6-7 [supporting_context]

Demo with no backup plan: a live API call with no cached response dies on the first rate-limit; a missing fallback video left judges unable to see what the project does at all.

- > Live API call with no fallback / One rate-limit and your demo dies mid-pitch / Cache a real response, use it as backup
  Source: `https://www.masaischool.com/blog/how-to-win-an-ai-hackathon-in-2026-full-guide/`
- > Always record a demo video. The LA Hacks Wi-Fi was spotty, and some teams couldn't run their demo because the app or the network died. Without a backup video, I had no clear way to see what the project actually did.
  Source: `https://raw.githubusercontent.com/idkwhatimd0ing/hackathonstarterkit/main/content/generated/md/blog/hackathon-pitch-mistakes-la-hacks.md`

### R6-8 [edge_case]

Edge sloppiness signals: obviously fake demo data ('Test123'), happy-path-only demo, vague benefit claims with no concrete numbers.

- > Obviously fake demo data. "Test123", "asdf@asdf.com," everything dated today. Judges read sloppy data as a signal the team didn't care.
  Source: `https://raw.githubusercontent.com/yashkoparde/hackathon101/main/docs/06-COMMON-MISTAKES.md`
- > No specific numbers or scenario. Vague claims ("this saves time") are forgettable; specific ones ("cuts a 20-minute task to 90 seconds") stick.
  Source: `https://raw.githubusercontent.com/yashkoparde/hackathon101/main/docs/06-COMMON-MISTAKES.md`

Agent notes:

> Stop-list for the 3-minute demo (all items confirmed by sources read in full): 1) never open with credentials/team/long intro — product on screen in the first seconds; 2) no slideware — demo of the working product, slides <=30s and <=10-15 words; 3) problem first (about 30% of time), then solution, one main user flow; 4) fit about 3 minutes; 5) good microphone, live confident voice; 6) do not read slides, explain the obvious, apologize, walk code line by line, or rush; 7) have a fallback (screen recording/cached responses), realistic demo data, and concrete benefit numbers. Could not read: Devpost FAQ and BizThon Medium article (403), three GitHub raw files (404 — wrong branch); their content was excluded from findings.

Sources:

- How to win an AI Hackathon in 2026 (Full Guide) — Masai School [reported_analysis, 2026-09-03]
  `https://www.masaischool.com/blog/how-to-win-an-ai-hackathon-in-2026-full-guide/`
- 9 Hackathon Pitch Mistakes I Saw at LA Hacks (judge first-hand account) [direct_evidence, 2026-09-14]
  `https://raw.githubusercontent.com/idkwhatimd0ing/hackathonstarterkit/main/content/generated/md/blog/hackathon-pitch-mistakes-la-hacks.md`
- VIDEO_CREATIVE_BRIEF — rhombic-agent, Nous Research Hermes Agent Hackathon (synthesis of Devpost/TechCrunch/AngelHack advice) [reported_analysis, 2026-03]
  `https://raw.githubusercontent.com/tasumermaf/rhombic/main/scripts/VIDEO_CREATIVE_BRIEF.md`
- Common Mistakes — hackathon101 guide [discussion, unknown]
  `https://raw.githubusercontent.com/yashkoparde/hackathon101/main/docs/06-COMMON-MISTAKES.md`
- FINAL_SUBMISSION_STRATEGY — AWS AI Agent Hackathon team strategy (What NOT to Do in Demo Video) [discussion, 2025]
  `https://github.com/sgharlow/ai-matcher-aws-hackathon/blob/HEAD/docs/FINAL_SUBMISSION_STRATEGY.md`

Totals: 49 claims, 27 source entries.
