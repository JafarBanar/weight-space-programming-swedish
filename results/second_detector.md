# Second language identifier: agreement with the primary detector

Answers re-labelled: **27,230**. Same label from both identifiers: **94.8%**.

Most common disagreements (primary → second): da->no (281), sv->no (236), no->sv (187), no->da (152), sv->da (103), da->sv (91), sv->en (57), en->sv (35), da->en (28), es->sq (18), no->en (17), nl->en (14).

Right-language rate of 77 round-2 and round-3 test sets under each identifier: mean absolute difference 1.4 points, largest 10.0 points.

| File | Set | Target | n | Primary | Second |
|---|---|---|---|---|---|
| r2/eurollm-1.7b/test_B0.json | T1_sv | sv | 320 | 99.4% | 99.1% |
| r2/eurollm-1.7b/test_B0.json | MK_sv | sv | 300 | 88.3% | 86.7% |
| r2/eurollm-1.7b/test_B0.json | MK_da | da | 150 | 81.3% | 78.0% |
| r2/eurollm-1.7b/test_B0.json | MK_en | en | 150 | 99.3% | 99.3% |
| r2/eurollm-1.7b/test_B0.json | T1_en | en | 160 | 100.0% | 98.8% |
| r2/eurollm-1.7b/test_P1.json | T1_sv | sv | 320 | 99.7% | 99.1% |
| r2/eurollm-1.7b/test_P1.json | MK_sv | sv | 300 | 94.7% | 91.0% |
| r2/eurollm-1.7b/test_RECIPE.json | T1_sv | sv | 320 | 99.7% | 99.7% |
| r2/eurollm-1.7b/test_RECIPE.json | MK_sv | sv | 300 | 88.7% | 86.3% |
| r2/eurollm-1.7b/test_RECIPE.json | MK_da | da | 150 | 21.3% | 12.7% |
| r2/eurollm-1.7b/test_RECIPE.json | MK_en | en | 150 | 99.3% | 99.3% |
| r2/eurollm-1.7b/test_RECIPE.json | T1_en | en | 160 | 100.0% | 98.8% |
| r2/eurollm-1.7b/test_R_seed0.json | T1_sv | sv | 320 | 99.4% | 99.1% |
| r2/eurollm-1.7b/test_R_seed0.json | MK_sv | sv | 300 | 88.7% | 86.7% |
| r2/eurollm-1.7b/test_R_seed1.json | T1_sv | sv | 320 | 99.4% | 99.1% |
| r2/eurollm-1.7b/test_R_seed1.json | MK_sv | sv | 300 | 88.7% | 87.0% |
| r2/qwen2.5-0.5b/test_B0.json | T1_sv | sv | 320 | 84.1% | 82.2% |
| r2/qwen2.5-0.5b/test_B0.json | MK_sv | sv | 300 | 69.3% | 69.3% |
| r2/qwen2.5-0.5b/test_B0.json | MK_da | da | 150 | 51.3% | 48.0% |
| r2/qwen2.5-0.5b/test_B0.json | MK_en | en | 150 | 100.0% | 100.0% |
| r2/qwen2.5-0.5b/test_B0.json | T1_en | en | 160 | 100.0% | 100.0% |
| r2/qwen2.5-0.5b/test_P1.json | T1_sv | sv | 320 | 96.2% | 98.8% |
| r2/qwen2.5-0.5b/test_P1.json | MK_sv | sv | 300 | 90.7% | 90.0% |
| r2/qwen2.5-0.5b/test_RECIPE.json | T1_sv | sv | 320 | 92.2% | 93.8% |
| r2/qwen2.5-0.5b/test_RECIPE.json | MK_sv | sv | 300 | 72.7% | 72.7% |
| r2/qwen2.5-0.5b/test_RECIPE.json | MK_da | da | 150 | 40.7% | 38.7% |
| r2/qwen2.5-0.5b/test_RECIPE.json | MK_en | en | 150 | 100.0% | 100.0% |
| r2/qwen2.5-0.5b/test_RECIPE.json | T1_en | en | 160 | 100.0% | 100.0% |
| r2/qwen2.5-0.5b/test_R_seed0.json | T1_sv | sv | 320 | 84.1% | 82.2% |
| r2/qwen2.5-0.5b/test_R_seed0.json | MK_sv | sv | 300 | 70.3% | 69.3% |
| r2/qwen2.5-0.5b/test_R_seed1.json | T1_sv | sv | 320 | 84.1% | 82.2% |
| r2/qwen2.5-0.5b/test_R_seed1.json | MK_sv | sv | 300 | 70.3% | 69.3% |
| r2/qwen2.5-1.5b/test_B0.json | T1_sv | sv | 320 | 96.6% | 95.9% |
| r2/qwen2.5-1.5b/test_B0.json | MK_sv | sv | 300 | 50.0% | 50.7% |
| r2/qwen2.5-1.5b/test_B0.json | MK_da | da | 150 | 47.3% | 44.0% |
| r2/qwen2.5-1.5b/test_B0.json | MK_en | en | 150 | 99.3% | 100.0% |
| r2/qwen2.5-1.5b/test_B0.json | T1_en | en | 160 | 100.0% | 99.4% |
| r2/qwen2.5-1.5b/test_P1.json | T1_sv | sv | 320 | 99.7% | 98.1% |
| r2/qwen2.5-1.5b/test_P1.json | MK_sv | sv | 300 | 96.3% | 94.3% |
| r2/qwen2.5-1.5b/test_P1.json | MK_da | da | 150 | 13.3% | 6.0% |
| r2/qwen2.5-1.5b/test_P1.json | MK_en | en | 150 | 90.7% | 90.7% |
| r2/qwen2.5-1.5b/test_P1.json | T1_en | en | 160 | 72.5% | 72.5% |
| r2/qwen2.5-1.5b/test_R_seed0.json | T1_sv | sv | 320 | 96.6% | 96.2% |
| r2/qwen2.5-1.5b/test_R_seed0.json | MK_sv | sv | 300 | 49.7% | 50.3% |
| r2/qwen2.5-1.5b/test_R_seed1.json | T1_sv | sv | 320 | 96.6% | 95.9% |
| r2/qwen2.5-1.5b/test_R_seed1.json | MK_sv | sv | 300 | 49.7% | 50.3% |
| r2/qwen2.5-1.5b/test_S2_steering.json | T1_sv | sv | 320 | 99.4% | 99.1% |
| r2/qwen2.5-1.5b/test_S2_steering.json | MK_sv | sv | 300 | 59.7% | 60.7% |
| r2/qwen2.5-1.5b/test_S2_steering.json | MK_da | da | 150 | 38.7% | 32.0% |
| r2/qwen2.5-1.5b/test_S2_steering.json | MK_en | en | 150 | 99.3% | 100.0% |
| r2/qwen2.5-1.5b/test_S2_steering.json | T1_en | en | 160 | 100.0% | 99.4% |
| r2/qwen2.5-1.5b/test_W0.json | T1_sv | sv | 320 | 93.1% | 94.4% |
| r2/qwen2.5-1.5b/test_W0.json | MK_sv | sv | 300 | 47.3% | 46.3% |
| r2/qwen2.5-1.5b/test_W0.json | MK_da | da | 150 | 15.3% | 10.7% |
| r2/qwen2.5-1.5b/test_W0.json | MK_en | en | 150 | 99.3% | 100.0% |
| r2/qwen2.5-1.5b/test_W0.json | T1_en | en | 160 | 100.0% | 99.4% |
| r2/qwen2.5-1.5b/test_W0a.json | T1_sv | sv | 320 | 98.4% | 99.1% |
| r2/qwen2.5-1.5b/test_W0a.json | MK_sv | sv | 300 | 51.0% | 50.7% |
| r2/qwen2.5-1.5b/test_W0a.json | MK_da | da | 150 | 16.0% | 10.0% |
| r2/qwen2.5-1.5b/test_W0a.json | MK_en | en | 150 | 99.3% | 100.0% |
| r2/qwen2.5-1.5b/test_W0a.json | T1_en | en | 160 | 100.0% | 99.4% |
| r2/qwen3-1.7b/test_B0.json | T1_sv | sv | 320 | 96.6% | 95.9% |
| r2/qwen3-1.7b/test_B0.json | MK_sv | sv | 300 | 86.7% | 86.7% |
| r2/qwen3-1.7b/test_B0.json | MK_da | da | 150 | 82.0% | 80.7% |
| r2/qwen3-1.7b/test_B0.json | MK_en | en | 150 | 100.0% | 100.0% |
| r2/qwen3-1.7b/test_B0.json | T1_en | en | 160 | 100.0% | 100.0% |
| r2/qwen3-1.7b/test_P1.json | T1_sv | sv | 320 | 99.4% | 99.4% |
| r2/qwen3-1.7b/test_P1.json | MK_sv | sv | 300 | 97.0% | 95.0% |
| r2/qwen3-1.7b/test_RECIPE.json | T1_sv | sv | 320 | 99.4% | 98.8% |
| r2/qwen3-1.7b/test_RECIPE.json | MK_sv | sv | 300 | 88.3% | 88.7% |
| r2/qwen3-1.7b/test_RECIPE.json | MK_da | da | 150 | 35.3% | 25.3% |
| r2/qwen3-1.7b/test_RECIPE.json | MK_en | en | 150 | 100.0% | 100.0% |
| r2/qwen3-1.7b/test_RECIPE.json | T1_en | en | 160 | 100.0% | 100.0% |
| r2/qwen3-1.7b/test_R_seed0.json | T1_sv | sv | 320 | 96.6% | 95.9% |
| r2/qwen3-1.7b/test_R_seed0.json | MK_sv | sv | 300 | 86.7% | 86.3% |
| r2/qwen3-1.7b/test_R_seed1.json | T1_sv | sv | 320 | 96.6% | 95.9% |
| r2/qwen3-1.7b/test_R_seed1.json | MK_sv | sv | 300 | 86.7% | 86.7% |
