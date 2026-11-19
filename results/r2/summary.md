
### Qwen2.5-1.5B-Instruct

Weight reading: **strong**; candidates L23/N981 (z = -9.2), L22/N1821 (z = -7.9). Sign step: L23/N981 fires +35.8 on Danish/Norwegian answers and -0.4 on Swedish answers → delete; L22/N1821 fires +1.0 on Danish/Norwegian answers and -19.3 on Swedish answers → keep (44 Danish/Norwegian and 62 Swedish dev answers).

| Condition | T1: Swedish answers (320 new-template prompts) | Δ vs B0 [95% CI] | T1 answers in da+no / en | MKQA: Swedish answers (300 human-translated questions) | Δ vs B0 [95% CI] | MKQA answers in da+no / en | Danish→Danish | English→English (MKQA / T1) | PPL sv / da | Grammar pairs: synthetic / cleaned / native |
|---|---|---|---|---|---|---|---|---|---|---|
| B0 Unedited | 96.6% | – | 11 / 0 | 50.0% | – | 10 / 139 | 47.3% | 99.3% / 100.0% | 52.5 / 99.3 | 73.0% / 73.5% / 75.0% |
| R Random neuron(s), 2 seeds | 96.6% | +0.0 | 11 / 0 | 49.7% | -0.3 | 10 / 140 | – | – | – | – |
| W0a Delete L23/N981 (pre-registered) | 98.4% | +1.9 [-0.9, +6.6] | 5 / 0 | 51.0% | +1.0 [-0.7, +2.7] | 4 / 142 | 16.0% | 99.3% / 100.0% | 51.7 / 170.6 | 73.4% / 74.0% / 76.0% |
| W0 Delete both neurons | 93.1% | -3.4 [-11.6, +3.1] | 21 / 0 | 47.3% | -2.7 [-5.7, +0.3] | 7 / 150 | 15.3% | 99.3% / 100.0% | 56.8 / 173.6 | 72.6% / 73.1% / 75.0% |
| S2 Steering, best of 54 | 99.4% | +2.8 [-0.6, +8.1] | 2 / 0 | 59.7% | +9.7 [+6.0, +13.7] | 3 / 117 | 38.7% | 99.3% / 100.0% | 51.4 / 135.4 | 71.3% / 72.2% / 77.5% |
| P1 Prompt instruction | 99.7% | +3.1 [+0.0, +8.7] | 1 / 0 | 96.3% | +46.3 [+40.3, +52.0] | 4 / 6 | 13.3% | 90.7% / 72.5% | 52.5 / 99.3 | 73.0% / 73.5% / 75.0% |

Steering search: 54 configurations, 6760 dev prompts; dev-val Swedish rate unedited 86.2%, selected: {'sv_rate': 1.0, 'sv_langs': {'sv': 80}, 'en_rate': 1.0, 'degenerate': 0.025, 'recipe': 'B_language', 'layer': 6, 'strength': 1}.

### Qwen2.5-0.5B-Instruct

Weight reading: **weak**; candidates L21/N952 (z = -5.1), L10/N2976 (z = -4.7). Sign step: L21/N952 fires +17.8 on Danish/Norwegian answers and -2.1 on Swedish answers → delete; L10/N2976 fires -0.1 on Danish/Norwegian answers and -0.1 on Swedish answers → keep (47 Danish/Norwegian and 72 Swedish dev answers).

| Condition | T1: Swedish answers (320 new-template prompts) | Δ vs B0 [95% CI] | T1 answers in da+no / en | MKQA: Swedish answers (300 human-translated questions) | Δ vs B0 [95% CI] | MKQA answers in da+no / en | Danish→Danish | English→English (MKQA / T1) | PPL sv / da | Grammar pairs: synthetic / cleaned / native |
|---|---|---|---|---|---|---|---|---|---|---|
| B0 Unedited | 84.1% | – | 36 / 11 | 69.3% | – | 15 / 76 | 51.3% | 100.0% / 100.0% | 136.1 / 234.2 | 53.6% / 54.3% / 70.0% |
| R Random neuron(s), 2 seeds | 84.1% | +0.0 | 36 / 11 | 70.3% | +1.0 | 12 / 76 | – | – | – | – |
| Recipe edit | 92.2% | +8.1 [+1.3, +15.9] | 12 / 10 | 72.7% | +3.3 [+0.0, +6.7] | 8 / 73 | 40.7% | 100.0% / 100.0% | 133.8 / 277.8 | 51.9% / 52.9% / 72.0% |
| P1 Prompt instruction | 96.2% | +12.2 [+4.1, +21.6] | 11 / 0 | 90.7% | +21.3 [+15.7, +27.0] | 13 / 13 | – | – | – | – |

### EuroLLM-1.7B-Instruct

Weight reading: **strong**; candidates L17/N1921 (z = -16.4), L19/N2597 (z = -14.2), L19/N1706 (z = -14.2), L16/N4202 (z = -7.1). Sign step: L17/N1921 fires +0.5 on Danish/Norwegian answers and -1.6 on Swedish answers → keep; L19/N2597 fires +0.8 on Danish/Norwegian answers and -17.2 on Swedish answers → keep; L19/N1706 fires +19.2 on Danish/Norwegian answers and -0.6 on Swedish answers → delete; L16/N4202 fires +1.4 on Danish/Norwegian answers and -0.3 on Swedish answers → delete (56 Danish/Norwegian and 96 Swedish dev answers).

| Condition | T1: Swedish answers (320 new-template prompts) | Δ vs B0 [95% CI] | T1 answers in da+no / en | MKQA: Swedish answers (300 human-translated questions) | Δ vs B0 [95% CI] | MKQA answers in da+no / en | Danish→Danish | English→English (MKQA / T1) | PPL sv / da | Grammar pairs: synthetic / cleaned / native |
|---|---|---|---|---|---|---|---|---|---|---|
| B0 Unedited | 99.4% | – | 2 / 0 | 88.3% | – | 7 / 16 | 81.3% | 99.3% / 100.0% | 23.8 / 48.8 | 98.3% / 98.2% / 92.5% |
| R Random neuron(s), 2 seeds | 99.4% | +0.0 | 2 / 0 | 88.7% | +0.3 | 7 / 15 | – | – | – | – |
| Recipe edit | 99.7% | +0.3 [+0.0, +1.9] | 1 / 0 | 88.7% | +0.3 [-1.0, +1.7] | 8 / 15 | 21.3% | 99.3% / 100.0% | 23.5 / 121.1 | 98.3% / 98.2% / 92.5% |
| P1 Prompt instruction | 99.7% | +0.3 [+0.0, +1.9] | 1 / 0 | 94.7% | +6.3 [+2.3, +10.3] | 2 / 11 | – | – | – | – |

### Qwen3-1.7B

Weight reading: **strong**; candidates L24/N2796 (z = -15.1), L21/N4923 (z = -11.0), L27/N3153 (z = -9.4), L27/N3191 (z = -7.1), L23/N5637 (z = -6.3), L27/N110 (z = -6.2), L25/N2164 (z = -6.1). Sign step: L24/N2796 fires +308.0 on Danish/Norwegian answers and +0.8 on Swedish answers → delete; L21/N4923 fires +0.6 on Danish/Norwegian answers and -96.4 on Swedish answers → keep; L27/N3153 fires -0.2 on Danish/Norwegian answers and -185.7 on Swedish answers → keep; L27/N3191 fires +0.2 on Danish/Norwegian answers and -2.5 on Swedish answers → keep; L23/N5637 fires +59.2 on Danish/Norwegian answers and -0.4 on Swedish answers → delete; L27/N110 fires -4.5 on Danish/Norwegian answers and -21.1 on Swedish answers → keep; L25/N2164 fires -34.0 on Danish/Norwegian answers and -0.1 on Swedish answers → keep (60 Danish/Norwegian and 90 Swedish dev answers).

| Condition | T1: Swedish answers (320 new-template prompts) | Δ vs B0 [95% CI] | T1 answers in da+no / en | MKQA: Swedish answers (300 human-translated questions) | Δ vs B0 [95% CI] | MKQA answers in da+no / en | Danish→Danish | English→English (MKQA / T1) | PPL sv / da | Grammar pairs: synthetic / cleaned / native |
|---|---|---|---|---|---|---|---|---|---|---|
| B0 Unedited | 96.6% | – | 9 / 1 | 86.7% | – | 7 / 30 | 82.0% | 100.0% / 100.0% | 53.7 / 98.5 | 69.6% / 70.4% / 81.0% |
| R Random neuron(s), 2 seeds | 96.6% | +0.0 | 9 / 1 | 86.7% | +0.0 | 7 / 30 | – | – | – | – |
| Recipe edit | 99.4% | +2.8 [+0.0, +7.5] | 0 / 1 | 88.3% | +1.7 [+0.3, +3.3] | 1 / 31 | 35.3% | 100.0% / 100.0% | 53.1 / 149.3 | 70.5% / 70.9% / 81.0% |
| P1 Prompt instruction | 99.4% | +2.8 [-0.6, +8.4] | 2 / 0 | 97.0% | +10.3 [+6.3, +14.3] | 2 / 7 | – | – | – | – |
