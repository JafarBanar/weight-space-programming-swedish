# Analyses of saved outputs and a calibrated null for the weight reading

## 1. Calibrated null for the weight reading

200 control directions built from the same 34 function-word tokens paired at random. Most negative neuron z-score per control direction: median -6.4, most extreme -14.6. Real direction: -9.2 (L23/N981) and -7.9 (L22/N1821). Share of control directions with any neuron at or beyond −7.9: 22.0%; at or beyond −9.2: 11.0%. So a large z alone is not rare: a neuron that promotes one token produces it.

Consistency across the 17 pairs (mean per-pair write divided by its standard error): L23/N981 t = -8.3 (rank 7 of 250,880, 15/17 pairs with the Danish sign); L22/N1821 t = -7.4 (rank 19, 17/17). Most negative t over all neurons for the control directions: median -6.5, most extreme -11.4; share of control directions with any neuron as consistent as L23/N981: 6.0%, as L22/N1821: 19.0%.

## 2. Round 1 by template

Three templates account for 39 of the 48 unedited failures.

| Set | Template | Unedited failures (of 20) | W0 fixed / broken | W0a fixed / broken |
|---|---|---|---|---|
| sv_A | Förklara kort vad {t} är. | 0 | 0 / 2 | 0 / 0 |
| sv_A | Skriv tre meningar om {t}. | 13 | 11 / 0 | 12 / 0 |
| sv_A | Vad är det viktigaste man bör veta om {t}? | 0 | 0 / 1 | 0 / 0 |
| sv_A | Kan du berätta lite om {t}? | 11 | 11 / 0 | 11 / 0 |
| sv_A | Ge mig två intressanta fakta om {t}. | 5 | 2 / 0 | 4 / 0 |
| sv_A | Varför spelar {t} roll i samhället? | 0 | 0 / 0 | 0 / 0 |
| sv_A | Hur skulle du förklara {t} för ett barn? | 0 | 0 / 0 | 0 / 0 |
| sv_A | Sammanfatta kort historien bakom {t}. | 0 | 0 / 2 | 0 / 0 |
| sv_B | Vilka är för- och nackdelarna med {t}? | 0 | 0 / 0 | 0 / 0 |
| sv_B | Skriv en kort dikt om {t}. | 15 | 7 / 0 | 7 / 0 |
| sv_B | Vad tänker du på när du hör ordet {t}? | 0 | 0 / 0 | 0 / 0 |
| sv_B | Beskriv {t} med enkla ord. | 3 | 3 / 5 | 3 / 0 |
| sv_B | Jag vill lära mig mer om {t}. Var ska jag börja? | 0 | 0 / 1 | 0 / 0 |
| sv_B | Vilka vanliga missförstånd finns det om {t}? | 1 | 1 / 2 | 1 / 0 |
| sv_B | Hur har synen på {t} förändrats de senaste hundra åren? | 0 | 0 / 2 | 0 / 0 |
| sv_B | Nämn tre saker som hänger ihop med {t}. | 0 | 0 / 0 | 0 / 0 |

## 3. Grammar pairs restricted to equal token counts

195 of 237 synthetic pairs and 184 of 200 native pairs have the same number of tokens in both sentences.

| Condition | All pairs | Equal-length pairs |
|---|---|---|
| B0_baseline | 73.0% | 71.8% |
| W0_weights_only | 72.6% | 72.8% |
| W0a_only_L23N981 | 73.4% | 72.8% |
| W0b_only_L22N1821 | 68.8% | 68.2% |
| F1_lora | 73.0% | 70.8% |
| r2_native_B0 | 75.0% | 78.8% |
| r2_native_W0a | 76.0% | 78.8% |
| r2_native_W0 | 75.0% | 77.7% |

## 4. MKQA Swedish questions: gold answer found, answer in Swedish, both

202 of 300 questions have a short gold answer (string match against the dataset's Swedish and English answers and aliases).

| Condition | Gold answer found | Answer in Swedish | Both |
|---|---|---|---|
| B0 | 8.9% | 49.0% | 5.0% |
| W0a | 8.9% | 49.5% | 4.5% |
| W0 | 8.4% | 45.0% | 3.5% |
| P1 | 9.9% | 95.0% | 9.4% |
| P2_same_language | 7.9% | 75.2% | 6.4% |

## 5. Round-1 Danish prompts answered in Danish, with and without three imprecise topic translations

| Condition | All 160 | Without genbrug / hjemmearbejde / ligestilling (136) |
|---|---|---|
| B0_baseline | 82.5% | 79.4% |
| W0_weights_only | 13.1% | 11.0% |
| W0a_only_L23N981 | 10.0% | 8.1% |
| F1_lora | 78.8% | 76.5% |
