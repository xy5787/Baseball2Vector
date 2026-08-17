# Phase 4 — External Validation Pilot (Qualitative, n=6)

**This is a pilot, not a validation result.** No correlation, effect size, or significance test is
computed across these six players — six points cannot support one, and doing so would overstate
what this check can show. The purpose is narrower: does Baseball2Vector's Z-score-encoded profile
for a player, in their first MLB season(s), broadly agree or disagree with an independent,
publicly published, pre-debut scouting assessment of the same player? Agreement and disagreement
are both reported; disagreements are not explained away.

## Why these six players

All six were top-of-scale prospects with **specific, numerically graded, publicly cited** 20-80
scouting reports available before or at their MLB debut, and all have at least one qualifying
(≥100 PA) season in `data/v3_results_clean.csv` (2021–2025). This combination — a recent, public,
numeric pre-MLB grade *and* enough MLB PA to compute a Baseball2Vector vector — is fairly rare;
it excludes both established veterans (whose formative scouting reports, if they exist publicly at
all, are a decade-plus old and rarely reported with digits) and current prospects who haven't
debuted yet. Grades were located via web search this session and are cited to a specific source
page per player in `pilot_table.csv`; no grade was estimated or guessed. Where a source did not
report a specific digit for a tool, that cell is blank in the table rather than filled in.

## Two structural caveats that apply to every row

1. **Different reference populations.** Public scouting grades are calibrated against the full
   population of minor-league prospects (most of whom never reach the majors). Baseball2Vector's
   20-80 scale is calibrated only against players who *already* qualified for MLB playing time
   (≥100 PA) in 2021–2025 — a much narrower, more elite population. A "60" on each scale is not the
   same percentile of the same population. The comparisons below should be read as a check on each
   player's *relative profile shape* (which tool reads as their carrying tool vs. their weakest,
   within Baseball2Vector's own scale), not as a claim that the two 60s are the same number.
2. **Plate Discipline has no public analogue.** Neither Baseball America nor MLB Pipeline grades
   patience/approach as a separate 20-80 tool in the sources found (it is sometimes folded
   qualitatively into the Hit-tool narrative, e.g. "advanced approach," but never given its own
   digit). Baseball2Vector's `Discipline` column is reported in the table for completeness but is
   never checked against a public number — only against qualitative narrative, when available.
   Defense is graded publicly as two separate tools (Field, Arm) that Baseball2Vector collapses
   into one `Defense` column; several disagreements below trace to this taxonomy mismatch rather
   than to Baseball2Vector being simply wrong.

## Per-player notes

**Bobby Witt Jr.** — BA grades (2019–21 era, [player page](https://www.baseballamerica.com/players/8310-bobby-witt/)):
Hit 60 / Power 60 / Run 60 / Arm 60 / Field 60, Overall FV 65 — a flat, plus-across-the-board
profile with no standout tool. His 2022 rookie-season Baseball2Vector vector disagrees sharply
with that flatness: Speed is his clear standout (64.0) while Defense and Discipline are his
weakest tools (38.0, 34.1) — not the uniform profile scouts projected. By his 2024 breakout season,
though, four of five tools (Contact 66.7, Power 59.7, Defense 59.5, Speed 52.6) cluster in the
high-50s/60s, much closer to the "plus everywhere" scouting picture — except Discipline, which
stays a clear laggard (41.2) throughout, a gap the standard BA taxonomy never had a tool to flag
since it doesn't grade discipline separately. Read together: the flat scouting profile was a better
description of Witt's *third* MLB season than his first.

**Julio Rodríguez** — BA ([player page](https://www.baseballamerica.com/players/7235-julio-rodriguez/)):
Hit 70, 80-grade raw power, Arm 70; Run and Field described only qualitatively as "average." His
2022 rookie vector: Power 58.6 (his highest tool, directionally consistent with the 80-grade raw
power projection, though naturally far short of it — raw power ceiling vs. in-game production is
a known gap), Speed 54.5 and Defense 47.4 both roughly track a scouted "average" Run/Field. The
clearest disagreement is Contact: Baseball2Vector reads it as almost exactly average (49.9)
against a scouted 70-grade Hit tool — plausible given Rodríguez's high rookie-season strikeout
rate, a common gap between pre-MLB hit-tool projections and first-year MLB contact production.

**Adley Rutschman** — BA ([player page](https://www.baseballamerica.com/players/4052-adley-rutschman/)):
Hit 70 was the only specific digit found; MLB Pipeline described him only qualitatively as having
"plus grades in each of the four tools that matter for catchers." His 2022 rookie vector: Contact
58.1 is directionally consistent with the 70-grade Hit tool (above average, though not as extreme).
Defense is his clear standout (63.0), consistent with the plus-defender/plus-arm catching
reputation scouts emphasized, even though no specific public Field/Arm digit was available to check
against directly. Discipline (57.8) is his second-highest Baseball2Vector tool, plausible given his
well-documented college approach (more walks than strikeouts at Oregon State) — but since no
outlet grades discipline on the 20-80 scale, this is a plausibility note, not a checked prediction.

**Gunnar Henderson** — BA ([player page](https://www.baseballamerica.com/players/8124-gunnar-henderson/)):
Hit 60 / Power 70 / Run 60 / Arm 70 / Field 60 — Power and Arm are his co-standout grades. Using
his first full-PA season (2023; his 2022 debut was only 132 PA and too small a sample to trust) —
Speed (55.5) edges out Power (54.4) as his single highest Baseball2Vector tool, a mild disagreement
with Power being the scouted standout, though Power is still clearly his second-best tool, so the
overall picture is closer to "two above-average tools" than a clean match or a clean miss. Defense
(46.6) reads below average despite a scouted 60 Field / 70 Arm — again plausibly a taxonomy
mismatch, since Baseball2Vector's single Defense column can't separately register elite arm
strength the way BA's split Field/Arm grades can.

**Corbin Carroll** — BA ([player page](https://www.baseballamerica.com/players/8111-corbin-carroll/)):
Hit 60 / Power 60 / Run 80 / Arm 45 / Field 60, Overall FV 65 — Speed is an extreme standout
(80-grade, the top of the scale) and Arm is his one below-average tool. His first full season
(2023; 2022 was only 115 PA) shows the cleanest agreement in this panel: Speed (78.8) is by far his
highest Baseball2Vector tool, closely tracking the extreme scouted run grade. The clearest
disagreement is Defense (40.9, his lowest tool) against a scouted average-plus 60 Field grade —
plausibly explained by his single below-average input (a 45-grade Arm) pulling down
Baseball2Vector's combined Defense column, again the Field/Arm taxonomy mismatch flagged above.

**Elly De La Cruz** — BA ([story](https://www.baseballamerica.com/stories/elly-de-la-cruz-pegs-the-top-of-the-scouting-scale/)):
Power 70 / Run 70 / Arm 70, Hit reported as a 40-45 range (no specific Field digit found). His 2023
debut vector: Contact (44.1) is indeed his lowest tool alongside Discipline, consistent with the
scouted 40-45 Hit-tool weakness being his one clear flaw. Speed (60.7) is his highest tool,
consistent with the scouted 70-grade run tool. Power (51.7), however, reads only marginally above
average against a scouted 70 — the clearest disagreement in this row, plausibly the same raw-power-
vs-game-power gap noted for Rodríguez: De La Cruz's *raw* power (batting-practice ceiling, what
scouts grade) has always been described as separate from and ahead of his *game* power (in-game
production), which is what Baseball2Vector's ISO/SLG/Barrel%-based Power tool actually measures.

## Overall pattern (qualitative only)

Across six players: Speed/Run shows the most consistent agreement (Witt, Carroll, De La Cruz all
show Baseball2Vector Speed tracking a scouted standout or average run grade reasonably well).
Power is the most consistent point of *disagreement*, generally reading lower in Baseball2Vector
than the scouted raw-power grade — plausibly because scouted power grades often describe raw,
batting-practice ceiling while Baseball2Vector's Power tool is built entirely from in-game
production stats (ISO, SLG, Barrel%, HardHit%, HR/FB), which is a real and defensible difference in
what is being measured, not necessarily a Baseball2Vector shortcoming. Defense disagreements
recur (Witt rookie year, Henderson, Carroll) and plausibly trace to the taxonomy mismatch: public
grades split Field and Arm, Baseball2Vector's Defense tool does not. None of this should be read as
more than a pattern worth further, larger-sample investigation — six players, six idiosyncratic
rookie seasons, and two different, non-commensurable 20-80 scales is not a validation sample.
