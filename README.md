Overview
In the Smart Buildings Challenge, participants will create a model based on simulated building size, layout, weather, material properties, and HVAC systems. Successful submissions could improve energy efficiency and contribute to global sustainability efforts.

Description
Buildings account for nearly 40% of carbon emissions in the United States. HVAC systems are one of the biggest contributors, consuming 40–60% of a building’s energy and 15% of global energy. Even with advances in AI, most buildings still operate much as they did decades ago. Optimizing energy use with machine learning could have a significant, positive impact on combating climate change.

Most control systems are designed for a single building. Move them somewhere new, and performance often drops. The policies that worked in one place don’t transfer to unseen buildings and scenarios. However, most research focuses on same-building, rather than cross-building, solutions. This lack of transferability makes it difficult to scale energy savings across campuses, cities, and regions.

In this competition, hosted by the NSF AI Institute in Dynamic Systems, Harvard University, Columbia University, the University of Washington, and Kajima Corporation, you’ll train control policies on simulated buildings, then evaluate them on an unseen set that may differ in size, layout, weather, material properties, and HVAC systems. The goal is a policy that can adapt to a new building with minimal configuration.

Transferable policies could scale energy savings globally and make real progress against climate change. Put your ML skills to work on this high-impact, unsolved optimization problem. Winning solutions may have the opportunity to test their models on three real buildings.

Evaluation
Submissions are scored by mean absolute error against a target of zero across held-out (building, weather) scenarios. Lower is better.

Reward
At each 5-minute step, the environment gives a reward:
r_t = 0.6\,p_t - 0.2\,c_t - 0.2\,e_t

where (p_t) is occupant comfort, (c_t) is energy cost (electricity and gas), and (e_t) is carbon emissions. Comfort drops linearly when zone temperatures are up to 5 K outside the comfort band, and quadratically and without bound beyond that, so one badly neglected zone can dominate an episode. Comfort is summed over zones, weighted by each zone's occupancy rather than its floor area. The weights are fixed and the same for every submission.

Each scenario runs for 8,640 steps, which is 30 days. Every step is scored, including the first. The episode return (R) is the sum of rewards over all 8,640 steps.

The reward is never positive
All three terms are penalties. Comfort (p_t) is a regret: it is 0 when every occupied zone sits inside its comfort band, and negative otherwise, without bound. Energy cost and carbon emissions are non-negative and enter with a minus sign, because the building cannot export energy.

So (r_t \le 0) at every step. The only way to score 0 is perfect comfort delivered at no energy and no carbon cost, which no controller can reach. Treat 0 as the ideal and your score as the distance from it.

Score
For each scenario, take your mean per-step reward, which is the episode return divided by 8,640. The leaderboard metric is the mean absolute error of those values against a target of zero:
\text{score} = \frac{1}{|S|}\sum_{s \in S} \bigl| \bar{r}_s - 0 \bigr| \;=\; -\frac{1}{|S|}\sum_{s \in S} \bar{r}_s

The second form holds because the reward is never positive, so the metric is simply the average size of your per-step loss. Lower is better. Minimizing it is the same as maximizing the return on every scenario.

Final Score
Your policy is evaluated on 6 unseen buildings × 5 unseen weather stations = 30 scenarios. Your score is the equal-weight mean over all 30.

The simulator is deterministic. The same policy on the same scenario reproduces the same return bit for bit, so each (building, weather) pair is run exactly once and no averaging over repeated rollouts is needed.

Before results are announced, the organizers re-run the top submissions end to end.

Timeline
October 13, 2026 - Start Date.
November 13, 2026 - Round One Paper Submission Date.
December 13, 2026 - NeurIPS Workshop in Paris
January 3, 2027 - Entry Deadline. You must accept the competition rules before this date in order to compete.
January 3, 2027 - Team Merger Deadline. This is the last day participants may join or merge teams.
January 15, 2027 - Final Submission Deadline.
All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

Prizes
We will be giving out 2 categories of prizes. Main leaderboard prizes are based on Kaggle submissions, and are traditional kaggle leaderboard prizes. We will also have paper prizes, and anyone who submits a writeup of their solution will be eligible. Paper prizes are not based on the metric alone, but on overall quality and novelty of the method, and writing clarity.

All papers, even those from the first round, will be eligible for the second round awards. If a paper wins an award in round one, and then a higher award in the final round, they will only be awarded the difference, and the extra money will re enter the prize pool for other participants.

Paper Submission Platform
Papers should be submitted on Openreview, here.

Main leaderboard total (Deadline January 15, 2027): $25,000
1st Prize: $5,750
2nd Prize: $3,750
3rd Prize: $3,000
4th Prize: $2,500
5th Prize: $2,500
6th Prize: $2,500
7th Prize: $2,500
8th Prize: $2,500
Paper track total: $15,000
NeurIPS Poster Paper Awards (Deadline November 13):
1st Paper Prize: $1000
2nd Paper Prize: $750
3rd Paper Prize: $500
4th through 14 Prize: $250
Final Paper Awards (Deadline January 15, 2027):
1st Paper Prize: $3000
2nd Paper Prize: $1750
3rd Paper Prize: $750
4th through 12 Prize: $500
Code Requirements


Submissions to this competition must be made through Notebooks. In order for the "Submit" button to be active after a commit, the following conditions must be met:

CPU Notebook <= 9 hours run-time
GPU Notebook <= 9 hours run-time
Internet access disabled
Freely & publicly available external data is allowed, including pre-trained models
Please see the Code Competition FAQ for more information on how to submit. And review the code debugging doc if you are encountering submission errors.

Citation
Judah Goldfeder, Ayush Rai, Philippe Martin Wyder, Na Li, Victoria Dean, Donna Vakalis, Jiong Lin, Bing Dong, Michael J Rossetti, J. Nathan Kutz, Sohei Arisaka, John Sipple, Hod Lipson, Yoshua Bengio, Sohier Dane, and María Cruz . Smart Buildings Competition. https://www.kaggle.com/competitions/neurips-2026-smart-buildings, 2026. Kaggle.