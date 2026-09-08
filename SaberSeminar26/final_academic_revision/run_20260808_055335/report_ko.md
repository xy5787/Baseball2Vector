# Baseball2Vector 최종 학술 수정 보고서

## A. 작업공간 및 원본 보존

별도 작업 폴더는 `final_academic_revision/run_20260808_055335`이다. 수정한 TeX는 `manuscript_preview/baseball2vector_en_revision.tex`, 컴파일 결과는 같은 폴더의 `baseball2vector_en_revision.pdf`이다.

권위 있는 원본 `paper/baseball2vector_en.tex`는 읽기만 했고 이동·포맷·수정하지 않았다. 작업 전후 SHA-256은 모두 `a873b4beb3b9acbcdd44c5f13d859b8c2a326241234ef49794fe20ff65e9f00d`로 동일하다. 차이는 `manuscript_preview/original_vs_revision.diff`에 비파괴 방식으로 저장했다.

분석 입력은 `data/processed/v3_results.csv`(SHA-256 `70bea280ec725ac78376e94edf41fae6fd036fe8b7a0b71270869eb1cfdf0873`)와 stable-ID가 붙은 격리 사본 `data_intermediate/player_seasons_with_ids.csv`(SHA-256 `0c810b9ea6aab754cd389ae7c5ab401a139bb251c6490a84b6e4b71d6cd07963`)이다. scaling 구현은 `src/baseball2vec/tools.py`와 `src/baseball2vec/baselines.py`를 읽어 추적했다. 로컬에 constituent raw-statistic cache가 없어 새 수집은 수행하지 않았다.

## B. 기존 결과 재현

다음 값은 재현되었다.

| 항목 | 재현값 |
|---|---:|
| area–WAR Pearson r | 0.8061 |
| league-mean MAE | 22.2743 |
| carry-forward MAE | 22.3492 |
| forecast-safe B2V MAE | 18.0267 |
| forecast-safe B2V R² | 0.2799 |
| forecast-safe B2V Spearman ρ | 0.5079 |
| 같은 선수 인접 시즌 거리 중앙값 | 0.7136 |

차이가 난 값은 오류를 숨기지 않고 교정했다.

- archived pooled 20–80 B2V의 MAE 18.0225는 재현되지만, 예측의 주 분석을 input-season standardized variant로 바꾸면 18.0267이다.
- scalar-matched 중앙값은 약 1.001에서 1.0041로 바뀌었다. 부동소수점상 거의 같은 gap을 12자리에서 반올림한 뒤 stable ID와 source row로 결정하는 명시적 tie-break를 적용했기 때문이다.
- random 중앙값은 약 1.197에서 1.1884로 바뀌었다. random 비교를 정확히 2,080개의 matching 성공 focal row와 같은 pair-count basis로 재구성했기 때문이다.

세부 기록은 `results/reproduction_check.json`에 있다.

## C. 보정된 scalar baseline 결과

학습은 2021→2022, 2022→2023, 2023→2024의 1,056개 전이, 최종 테스트는 2024→2025의 동일한 358명이다. Ridge는 stable player ID 기준 5-fold GroupKFold에서 training MAE로 alpha를 고르고, imputation과 scaling은 각 training fold 및 최종 training set에서만 적합했다. 표의 구간은 held-out player ID 2,000회 bootstrap 95% CI이다.

| 모델 | 선택 α | MAE | R² | Spearman ρ |
|---|---:|---:|---:|---:|
| 학습 target 평균 | — | 22.274 [20.511, 24.003] | -0.001 [-0.016, -0.000] | — |
| wRC+ carry-forward | — | 22.349 [20.514, 24.386] | -0.102 [-0.298, 0.078] | 0.449 [0.355, 0.535] |
| calibrated wRC+ Ridge | 10 | 19.220 [17.618, 20.892] | 0.206 [0.119, 0.290] | 0.449 [0.355, 0.535] |
| wRC+ + age + PA Ridge | 100 | 19.182 [17.576, 20.756] | 0.220 [0.139, 0.294] | 0.463 [0.378, 0.541] |
| B2V Ridge | 0.01 | 18.027 [16.514, 19.681] | 0.280 [0.179, 0.375] | 0.508 [0.423, 0.589] |
| B2V + age + PA Ridge | 10 | 18.028 [16.528, 19.634] | 0.288 [0.192, 0.381] | 0.522 [0.438, 0.600] |
| wRC+ + B2V + age + PA Ridge | 10 | 18.057 [16.561, 19.654] | 0.286 [0.190, 0.378] | 0.520 [0.435, 0.600] |

MAE paired difference는 `baseline MAE - candidate MAE`로 정의하여 양수가 candidate 개선을 뜻한다.

- B2V 대 carry-forward: +4.322 [2.947, 5.838]. B2V가 더 낫다.
- B2V 대 calibrated wRC+: +1.193 [0.403, 2.012]. 이 테스트에서는 통계적으로 구별되는 개선이다. R² 개선은 +0.074 [0.022, 0.122], ρ 개선은 +0.059 [0.002, 0.120]이다.
- B2V-context 대 scalar-context: +1.154 [0.404, 1.880]. 현재 wRC+가 포함된 비교 기준보다 B2V profile이 추가 정보를 제공한다.
- combined 대 scalar-context: +1.125 [0.304, 1.929]. 결합 모델도 scalar-context보다 낫다.
- combined 대 B2V-context: -0.029 [-0.145, 0.086]. B2V-context 이후 current wRC+의 추가 이득은 명확하지 않다.
- scalar-context 대 calibrated wRC+: +0.038 [-0.496, 0.543], B2V-context 대 B2V-only: -0.001 [-0.376, 0.368]. age와 PA의 명확한 추가 MAE 개선은 관찰되지 않았다.

따라서 연구 질문에 대한 답은 “그렇다, 다만 제한적으로”이다. B2V는 보정된 current wRC+보다 예측 정보가 더 있었고 scalar-context 이후에도 증분 정보가 있었다. 그러나 calibrated scalar 대비 MAE 개선 1.193점은 그 모델 오차의 약 6.2%이므로 실용적 크기는 modest하다. 단일 최종 연도와 qualified returner만을 대상으로 한 결과이며 raw inputs 대비 우월성은 검증하지 못했다.

## D. Temporal scaling 점검

기존 score는 2021–2025 pooled min/max로 같은 affine map을 적용했다.

`x_20_80 = 20 + 60 × (x - pooled_min_2021_2025) / (pooled_max_2021_2025 - pooled_min_2021_2025)`

`MinMaxScaler(feature_range=(20,80))`의 기본 `clip=False`를 사용했고 별도 clipping은 없다. 따라서 archived 2024 score의 offset과 scale에는 2025 extreme이 들어간다. 이후 예측 pipeline에는 training-only StandardScaler가 한 번 더 있다.

pre-min-max score가 없어 각 입력 시즌 코호트 안에서 해당 시즌의 평균과 모집단 표준편차(ddof=0)로 5개 차원을 표준화한 variant를 주 분석으로 사용했다. 이는 2025 extreme에 대한 의존성을 제거한다.

| B2V 모델 | archived MAE / R² / ρ | forecast-safe MAE / R² / ρ | 최대 개인 예측 차이 |
|---|---|---|---:|
| B2V | 18.0225 / 0.2804 / 0.5077 | 18.0267 / 0.2799 / 0.5079 | 0.7939 |
| B2V + age + PA | 18.0184 / 0.2889 / 0.5216 | 18.0279 / 0.2885 / 0.5222 | 0.6595 |
| combined | 18.0508 / 0.2869 / 0.5207 | 18.0569 / 0.2863 / 0.5197 | 0.7507 |

집계 영향은 매우 작지만 두 방법은 정확히 같지 않다. 따라서 archived 결과에는 “leakage-free”를 쓰지 않았고, input-season standardized 결과만 forecast-safe prospective evaluation로 기술했다.

## E. Matching 및 안정성 점검

Matching 규칙은 다음과 같이 고정했다.

- 같은 시즌, 다른 stable player ID만 후보로 사용
- `|ΔwRC+| ≤ 5`와 `|ΔWAR| ≤ 0.2`를 동시에 충족
- 해당 시즌 wRC+ 및 WAR sample SD로 나눈 2차원 Euclidean scalar gap 최소화
- gap을 12자리 반올림한 값, stable ID, source row 순서로 tie-break
- B2V 거리는 후보 선택에 사용하지 않음
- comparison 재사용 허용
- random pair는 matching에 성공한 동일 focal 2,080개를 기준으로 같은 시즌·같은 PA 구간에서 seed 42로 한 개씩 선택
- profile distance는 5개 차원을 각 pooled sample SD로 표준화한 뒤 RMS
- 중앙값과 차이의 CI는 focal stable player ID 단위 2,000회 bootstrap

2,286 focal player-season 중 2,080개가 matching되었고 206개는 joint-threshold 후보가 없었다. unique focal player는 802명, unique comparison은 709명이다. comparison 516명(고유 comparison의 72.8%)이 재사용되었으며, 재사용 comparison을 포함한 row는 1,887개(90.7%)다.

| 비교군 | pair 수 | 중앙값 | IQR | 95% CI |
|---|---:|---:|---:|---:|
| 같은 선수 인접 시즌 | 1,414 | 0.714 | [0.541, 0.904] | [0.696, 0.726] |
| scalar-matched 다른 선수 | 2,080 | 1.004 | [0.746, 1.289] | [0.980, 1.025] |
| 같은 focal의 같은 시즌·PA random | 2,080 | 1.188 | [0.892, 1.532] | [1.161, 1.210] |

scalar-matched minus same-player는 0.291 [0.264, 0.317], random minus scalar-matched는 0.184 [0.154, 0.211]이다. 즉 scalar matching이 profile 이질성을 일부 줄이지만 대부분 제거하지는 않는다. CI는 구축된 matching 표본에 조건부이며 독립 pair 모집단 실험을 뜻하지 않는다.

PA 안정성은 `min(PA_t, PA_t+1)`로 구간을 정했다. 아래는 pair 수와 Pearson r [95% player-cluster CI]이다.

| 차원 | 전체 | 100–249 PA | 250–499 PA | ≥500 PA |
|---|---|---|---|---|
| Contact | 1414; 0.75 [0.72, 0.78] | 507; 0.63 [0.57, 0.68] | 567; 0.76 [0.71, 0.80] | 340; 0.78 [0.71, 0.84] |
| Power | 1414; 0.75 [0.71, 0.78] | 507; 0.58 [0.51, 0.64] | 567; 0.75 [0.69, 0.80] | 340; 0.80 [0.74, 0.85] |
| Plate Discipline | 1414; 0.77 [0.74, 0.80] | 507; 0.72 [0.67, 0.76] | 567; 0.78 [0.74, 0.82] | 340; 0.81 [0.73, 0.86] |
| Defense | 1414; 0.59 [0.53, 0.63] | 507; 0.46 [0.37, 0.54] | 567; 0.61 [0.52, 0.68] | 340; 0.63 [0.55, 0.70] |
| Speed | 1414; 0.64 [0.60, 0.67] | 507; 0.51 [0.44, 0.58] | 567; 0.65 [0.60, 0.71] | 340; 0.69 [0.61, 0.75] |

모든 차원이 PA가 많을수록 대체로 안정적이고 Defense가 일관되게 가장 낮다. Spearman 민감도 결과는 `results/stability_by_pa.csv`에 함께 저장했다.

## F. 그림 수정

- Figure 1: 기존 ordinary row-level `p < 0.001`을 제거했다. plot에는 `r=0.806`과 player-clustered 95% CI `[0.779,0.830]`만 넣었고 caption과 JSON이 일치한다.
- Figure 2: 기존 caption이 말한 right panel이 없던 문제를 고쳤다. 왼쪽은 세 거리 분포, 오른쪽은 검증된 Isaac Paredes–Salvador Perez 2024 radar example이다. wRC+ 116/117, WAR 3.3/3.3, PA 641/652, 거리 1.797을 저장 pair row에서 직접 표시했다. 예시는 illustrative로만 해석했다.
- Figure 3: ROI 및 random-forest importance를 제거했다. concurrent Δdimension 대 ΔwRC+ linear coefficient와 player-clustered CI, zero line만 표시했으며 caption은 descriptive partial association임을 명시한다.

세 PDF figure를 개별 확인했고, 최종 8쪽 PDF를 모두 120 dpi PNG로 렌더링해 page-by-page로 점검했다. Figure 2의 두 panel과 caption이 일치하며, Figure 3에는 ROI 표현이 없다. Table 1의 53pt overflow는 estimate/CI 두 줄 셀로 수정했다. 최종 log에는 undefined reference/citation과 overfull/underfull box가 없다. Tectonic/XeTeX의 Times font-shape fallback 경고만 남으며 배치 오류는 아니다.

## G. 논문 주장 변화

### 강하게 주장 가능

- 공개된 B2V output은 서로 다른 선수가 비슷한 wRC+와 WAR에 도달하는 profile 차이를 보존한다.
- 이 데이터와 고정된 test set에서 B2V는 naive carry-forward와 calibrated current-wRC+보다 낮은 next-season wRC+ MAE를 보인다.
- 저 PA 시즌의 차원 score는 더 불안정하며 Defense 안정성이 가장 낮다.

### 제한적으로 주장 가능

- B2V는 current wRC+ 이후에도 modest한 증분 예측 정보를 제공한다. 근거는 scalar-context 대비 B2V-context 및 combined의 paired CI가 0을 넘는다는 점이다.
- Z-score area의 WAR 상관은 내부 일관성 점검으로는 쓸 수 있으나 shared input 때문에 외부 타당도 증거는 아니다.
- matching 결과는 scalar가 버리는 profile 구조를 보여주지만 재사용을 허용한 구성 표본에 조건부다.

### 현재 결과로 주장할 수 없음

- 전문 scout grade와의 수치적 동등성 또는 외부 construct validity
- raw constituent statistics보다 우월함
- causal development effect, 최적 훈련 target, counterfactual 효과 또는 ROI
- Defense 차원의 타당한 잠재구성
- 모든 MLB 선수나 미래 연도에 대한 보편적 예측 우월성
- JointVAE latent의 안정성 또는 해석 가능성

검증할 원문 scouting report가 없는 Baseball America Table 5와 관련 본문·주장·참조는 수정본에서 모두 제거된 상태를 테스트로 확인했다.

## H. 남은 한계

- constituent raw-statistic 데이터가 없어 raw-statistic Ridge를 비교하지 못했다.
- Def에는 Fld와 positional adjustment가 포함되므로 Def와 Fld의 평균은 fielding을 중복하고 position·opportunity·playing time을 섞는다.
- pre-shrinkage feature가 없어 shrinkage 강도 ablation을 수행하지 못했다.
- constituent feature가 없어 leave-one-statistic-out ablation을 수행하지 못했다.
- JointVAE artifact와 반복 seed 결과가 없어 multi-seed latent stability를 평가하지 못했다.
- 독립적으로 검증 가능한 scouting/reference dataset이 없어 external construct validity가 확립되지 않았다.
- final test가 2024→2025 한 연도뿐이라 era 변화와 장기 일반화를 평가하지 못한다.
- 2025년에 다시 100 PA 이상 기록한 선수만 test에 남아 survivorship와 role-selection bias가 있다.
- bootstrap CI는 고정된 modeling 및 matching 절차에 조건부이며 전체 분석 선택 불확실성을 포함하지 않는다.

## I. 결과물과 재현 명령

주요 결과물:

- 수정 TeX: `manuscript_preview/baseball2vector_en_revision.tex`
- preview PDF: `manuscript_preview/baseball2vector_en_revision.pdf`
- 원본 대비 diff: `manuscript_preview/original_vs_revision.diff`
- prediction 결과: `results/calibrated_prediction.csv`
- prediction bootstrap: `results/calibrated_prediction_bootstrap.csv`
- paired 차이: `results/paired_model_differences.csv`
- scaling audit: `results/scaling_audit.json`
- matching pair: `results/matched_profile_distances.csv`
- matching flow: `results/matching_sample_flow.csv`
- stability: `results/stability_by_pa.csv`
- 재현 점검: `results/reproduction_check.json`
- figures: `figures/figure1_alignment_revised.pdf`, `figure2_profile_diversity_revised.pdf`, `figure3_delta_association_revised.pdf`
- LaTeX tables: `tables/prospective_prediction_revised.tex`, `tables/stability_by_pa.tex`
- 테스트 log: `logs/pytest.log`
- 환경 기록: `logs/environment_used.txt`

저장소 루트에서 전체 분석 재현:

```bash
./final_academic_revision/run_20260808_055335/run_all.sh
```

PDF 재현:

```bash
cd final_academic_revision/run_20260808_055335/manuscript_preview
../.conda-tectonic/bin/tectonic baseball2vector_en_revision.tex --keep-logs --keep-intermediates
../.conda-tectonic/bin/pdftoppm -png -r 120 baseball2vector_en_revision.pdf page_renders/page
```

실제 환경은 Python 3.12.3, NumPy 2.5.1, pandas 3.0.5, SciPy 1.18.0, scikit-learn 1.9.0, Matplotlib 3.11.1, Tectonic 0.17.0, Poppler 26.07.0이다. 분석 seed는 42, bootstrap은 2,000회다. 최종 원본 SHA-256 재확인 결과 원본 TeX는 변경되지 않았다.
