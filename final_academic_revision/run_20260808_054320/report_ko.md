# Baseball2Vector 최종 학술 수정 보고서

## 핵심 과학적 결론

Baseball2Vector는 고정된 2024→2025 검증 표본에서 단순 전년도 wRC+ 이월뿐 아니라, 평균 회귀를 학습한 보정 wRC+ Ridge보다도 더 정확했다. Forecast-safe B2V Ridge의 MAE는 18.027, 보정 wRC+ Ridge는 19.220이었고, 같은 선수를 함께 재표집한 MAE 개선량은 1.193점(95% 신뢰구간 0.403–2.012)이었다. 따라서 이번 표본에서는 다차원 프로필이 보정된 현재 공격가치보다 추가적인 차년도 예측 정보를 가진다고 제한적으로 주장할 수 있다. 다만 개선 폭은 약 1.2 wRC+로 실용적 크기가 크다고 단정하기 어렵고, 최종연도 한 번의 검증이라는 한계가 있다.

현재 wRC+, B2V, 나이, PA를 합친 모형은 scalar-context 모형보다 MAE가 1.125점 개선되었다(0.304–1.929). 이는 현재 wRC+를 포함한 뒤에도 B2V가 보완 정보를 준다는 증거다. 반대로 결합모형은 B2V-context보다 개선되지 않았다(-0.029, -0.145–0.086). 나이와 PA도 scalar 또는 B2V에 명확한 추가 MAE 이득을 주지 않았다.

## A. 작업공간 및 원본 보존

- 별도 작업 폴더: `/home/i2slab4/baseball-strategy-agents/baseball2vec/final_academic_revision/run_20260808_054320`
- 수정한 TeX 복사본: `manuscript_preview/baseball2vector_en_revision.tex`
- 복사 직후 원본 보존본: `manuscript_preview/baseball2vector_en_source_copy.tex`
- 권위 원본: `/home/i2slab4/baseball-strategy-agents/baseball2vec/paper/baseball2vector_en.tex`
- 작업 시작과 종료 시 권위 원본 SHA-256: `a873b4beb3b9acbcdd44c5f13d859b8c2a326241234ef49794fe20ff65e9f00d`
- 복사 직후 원본 보존본 SHA-256도 동일하다.
- 비파괴 비교 파일: `manuscript_preview/original_vs_revision.diff`

권위 원본 TeX는 수정, 이동, 재포맷, stage, commit하지 않았다. 다만 첫 테스트 실행에서 `uv`가 상위 프로젝트의 기존 `.venv`를 Python 3.12 환경으로 재생성했다. 이후 테스트는 `--isolated`로 실행했으며, 이 부수효과는 원고·데이터·Git 추적 파일에는 영향을 주지 않았지만 투명성을 위해 기록한다. 기존 Git 작업트리는 처음부터 수정·미추적 파일이 있었으며 이를 보존했다. 이번 작업에서 생성한 연구 산출물은 모두 위 타임스탬프 폴더 안에 두었다.

사용한 주 입력은 `data/processed/v3_results.csv`와 기존 검증에서 생성된 안정-ID 자료 `academic_validation_experiments/data_intermediate/player_seasons_with_ids.csv`이다. 후자를 작업공간의 `data_intermediate/player_seasons_with_ids.csv`로 복사하여 사용했다. 각각의 해시는 `results/input_provenance.json`에 기록했다. 원시 FanGraphs 구성 통계 캐시와 pre-min-max 차원 점수는 없었다.

## B. 기존 결과 재현

기존 prospective 결과는 archived pooled 20–80 벡터로 보고된 반올림 자릿수까지 재현됐다.

| 항목 | 기존 근삿값 | 재현값 | 판정 |
|---|---:|---:|---|
| League mean MAE | 22.27 | 22.274 | 재현 |
| Carry-forward MAE | 22.35 | 22.349 | 재현 |
| B2V Ridge MAE | 18.02 | 18.023 | 재현 |
| B2V Ridge R² | 0.280 | 0.280 | 재현 |
| B2V Ridge Spearman | 0.508 | 0.508 | 재현 |
| 동일 선수 연속시즌 거리 중앙값 | 0.714 | 0.714 | 재현 |
| scalar-matched 거리 중앙값 | 1.001 | 1.001 | 재현 |
| 무작위 거리 중앙값 | 1.197 | 1.188 | 수정 후 차이 |

무작위 거리 차이는 오류 은폐나 seed 조정 때문이 아니다. 기존 값은 2,286개 전체 안정-ID focal row를 무작위 집단에 사용했지만, 최종 규칙은 요청대로 scalar-matched와 동일한 2,080개 성공 focal row를 사용한다. 이 pair-count 교정으로 중앙값이 1.197에서 1.188로 바뀌었다. 상세 판정은 `results/reproduction_check.json`에 저장했다.

## C. 보정된 scalar baseline 결과

학습은 2021→2022, 2022→2023, 2023→2024의 1,056개 전이이고, 최종 검증은 2024→2025의 358개 전이다. 모든 모형이 같은 358명에게 평가됐다. 대괄호는 안정 선수 ID를 단위로 한 2,000회 bootstrap 95% 신뢰구간이다.

| 모형 | MAE | R² | Spearman |
|---|---:|---:|---:|
| 학습 target 평균 | 22.27 [20.51, 24.00] | -0.001 [-0.016, -0.000] | 정의되지 않음 |
| 전년도 wRC+ 이월 | 22.35 [20.51, 24.39] | -0.102 [-0.298, 0.078] | 0.449 [0.355, 0.535] |
| 보정 wRC+ Ridge | 19.22 [17.62, 20.89] | 0.206 [0.119, 0.290] | 0.449 [0.355, 0.535] |
| wRC+ + 나이 + PA Ridge | 19.18 [17.58, 20.76] | 0.220 [0.139, 0.294] | 0.463 [0.378, 0.541] |
| B2V Ridge | 18.03 [16.51, 19.68] | 0.280 [0.179, 0.375] | 0.508 [0.423, 0.589] |
| B2V + 나이 + PA Ridge | 18.03 [16.53, 19.63] | 0.288 [0.192, 0.381] | 0.522 [0.438, 0.600] |
| wRC+ + B2V + 나이 + PA Ridge | 18.06 [16.56, 19.65] | 0.286 [0.190, 0.378] | 0.520 [0.435, 0.600] |

MAE의 paired 차이는 `baseline MAE − candidate MAE`로 정의해 양수가 candidate 개선을 뜻한다.

| Candidate | Baseline | MAE 개선량 [95% 신뢰구간] | 해석 |
|---|---|---:|---|
| B2V | carry-forward | 4.322 [2.947, 5.838] | 명확한 개선 |
| B2V | 보정 wRC+ | 1.193 [0.403, 2.012] | 통계적으로 구별되지만 크기는 제한적 |
| scalar + 나이 + PA | 보정 scalar | 0.038 [-0.496, 0.543] | 명확하지 않음 |
| B2V + 나이 + PA | B2V | -0.001 [-0.376, 0.368] | 명확하지 않음 |
| B2V + 나이 + PA | scalar + 나이 + PA | 1.154 [0.404, 1.880] | 명확한 개선 |
| 결합모형 | scalar + 나이 + PA | 1.125 [0.304, 1.929] | B2V의 보완 정보 |
| 결합모형 | B2V + 나이 + PA | -0.029 [-0.145, 0.086] | current wRC+의 추가 이득 없음 |

- B2V가 carry-forward보다 나은가: 그렇다.
- B2V가 calibrated wRC+보다 나은가: 이번 holdout에서는 그렇다.
- B2V가 current wRC+ 이후에도 추가 정보를 제공하는가: 결합모형과 scalar-context 비교에서는 그렇다.
- 나이와 PA의 추가 효과가 있는가: MAE 기준 명확한 이득은 없다.
- 차이가 통계적으로 구별되는가: B2V 대 보정 scalar, B2V-context 대 scalar-context, 결합 대 scalar-context는 paired CI가 0을 제외한다. 결합 대 B2V-context와 나이·PA 추가 비교는 0을 포함한다.
- 실용적 크기: 보정 scalar 대비 약 1.2 wRC+ MAE 감소이므로 유용할 가능성은 있으나 큰 개선으로 과장해서는 안 된다.

Ridge penalty는 `[0.01, 0.1, 1, 10, 100, 1000]`의 공통 grid에서 학습기간 내부 5-fold 선수-ID GroupKFold와 MAE로 골랐다. Imputer와 StandardScaler는 매 fold와 최종 학습자료에만 적합했다. 선택 alpha는 보정 scalar 10, scalar-context 100, B2V 0.01, B2V-context 10, 결합 10이다. 모든 누수·ID·공통 표본 검사는 `experiments/tests/`에 있으며 최종 12개 검사가 통과했다.

## D. Temporal scaling 점검

구현 추적 결과는 다음과 같다.

- `src/baseball2vec/tools.py`의 `scale_to_2080`은 `MinMaxScaler(feature_range=(20, 80))`를 쓴다.
- 2021–2025 전체 차원 점수에 한 번 적합한 같은 affine 변환을 모든 시즌에 적용한다.
- 식은 `20 + 60 × (x − pooled min)/(pooled max − pooled min)`이다.
- `clip` 인자를 쓰지 않아 기본값 `False`이고 별도 clipping도 없다.
- 원시 통계는 먼저 입력 통계별로 시즌 내 표준화되지만, 최종 20–80 min-max extrema에는 2025가 포함된다.
- 예측 Ridge 앞에는 별도의 training-only StandardScaler가 있다.

따라서 archived 2024 predictor의 표시 스케일에는 2025 extrema가 사용됐다. 단, 영향은 차원별 공통 offset과 scale에 한정되며 clipping은 없다. Pre-min-max 점수가 없으므로 입력 시즌의 선수 cohort만으로 각 archived 차원을 다시 표준화하는 forecast-safe 변형을 만들고 이를 주 분석으로 사용했다.

| 벡터 모형 | Archived MAE | Forecast-safe MAE | 최대 개인 예측 절대차 |
|---|---:|---:|---:|
| B2V | 18.0226 | 18.0267 | 0.794 |
| B2V + 나이 + PA | 18.0184 | 18.0279 | 0.659 |
| 결합 | 18.0508 | 18.0569 | 0.751 |

집계 성능은 거의 같지만 개인 예측은 완전히 같지 않으므로 수치적 동등성을 주장하지 않았다. 수정 원고의 주 prospective 결과는 forecast-safe 변형이다. Archived globally scaled 결과에는 “leakage-free”라는 표현을 쓰지 않았고, forecast-safe 분석은 outcome-season feature가 없는 prospective 평가로 기술했다. 세부 JSON은 `results/scaling_audit.json`이다.

## E. Matching 및 안정성 점검

### Matching 규칙과 표본

- 기준: 같은 시즌, 다른 안정 선수 ID, `|ΔwRC+| ≤ 5`, `|ΔWAR| ≤ 0.2`.
- 모든 행은 원 데이터 설계상 PA 100 이상이다.
- 순위식: 시즌 내 wRC+와 WAR 표준편차로 나눈 두 차이의 Euclidean norm.
- tie-break: 안정 선수 ID, 그다음 원본 행 순서.
- B2V 거리는 후보 선택에 사용하지 않았다.
- comparison player 재사용을 허용했다.
- random은 성공적으로 match된 같은 2,080 focal row 각각에 대해 같은 시즌·같은 PA 구간에서 seed 42로 한 명을 뽑았다.
- bootstrap 단위는 focal 안정 선수 ID이다. 신뢰구간은 구성된 matching 절차에 조건부이며 독립 population 실험이 아니다.

| 흐름 | 수 |
|---|---:|
| 안정-ID focal player-season | 2,286 |
| 성공 joint-threshold match | 2,080 |
| 후보가 없어 실패 | 206 |
| 고유 focal 선수 | 802 |
| 고유 comparison 선수 | 707 |
| 두 번 이상 재사용된 comparison 선수 | 520/707, 73.6% |
| 재사용 comparison이 포함된 match row | 1,893/2,080, 91.0% |

| 거리 집단 | Pair 수 | 중앙값 | IQR | 95% 신뢰구간 |
|---|---:|---:|---:|---:|
| 동일 선수 연속시즌 | 1,414 | 0.714 | [0.541, 0.904] | [0.696, 0.726] |
| scalar-matched 다른 선수 | 2,080 | 1.001 | [0.743, 1.289] | [0.979, 1.025] |
| 무작위 같은 시즌·PA 구간 | 2,080 | 1.188 | [0.892, 1.532] | [1.161, 1.210] |

Scalar-matched−동일 선수 중앙값 차이는 0.288 [0.262, 0.317], 무작위−scalar-matched는 0.187 [0.154, 0.213]이다.

### PA 구간별 안정성

각 셀은 `Pearson r [95% 선수-cluster 신뢰구간]; adjacent pair 수`이다. PA 구간은 두 시즌 중 작은 PA로 정했다.

| 차원 | 전체 | 100–249 PA | 250–499 PA | 500 PA 이상 |
|---|---|---|---|---|
| Contact | 0.75 [0.72, 0.78]; 1,414 | 0.63 [0.57, 0.68]; 507 | 0.76 [0.71, 0.80]; 567 | 0.78 [0.71, 0.84]; 340 |
| Power | 0.75 [0.71, 0.78]; 1,414 | 0.58 [0.51, 0.64]; 507 | 0.75 [0.69, 0.80]; 567 | 0.80 [0.74, 0.85]; 340 |
| Plate Discipline | 0.77 [0.74, 0.80]; 1,414 | 0.72 [0.67, 0.76]; 507 | 0.78 [0.74, 0.82]; 567 | 0.81 [0.73, 0.86]; 340 |
| Defense | 0.59 [0.53, 0.63]; 1,414 | 0.46 [0.37, 0.54]; 507 | 0.61 [0.52, 0.68]; 567 | 0.63 [0.55, 0.70]; 340 |
| Speed | 0.64 [0.60, 0.67]; 1,414 | 0.51 [0.44, 0.58]; 507 | 0.65 [0.60, 0.71]; 567 | 0.69 [0.61, 0.75]; 340 |

모든 차원에서 500 PA 이상 구간이 100–249 PA보다 안정적이다. 이는 표본 기회 증가에 따른 reliability와 일치하지만 큰 역할을 계속 받은 선수의 선택 효과도 포함할 수 있다. Spearman 민감도 분석도 `results/stability_by_pa.csv`에 함께 저장했고 질적 순서는 같다.

## F. 그림 수정

### Figure 1

- 기존 오류: 일반 row-level `p < 0.001`이 선수-cluster uncertainty 강조와 맞지 않았다.
- 수정: 일반 p-value를 제거하고 `r = 0.81; player-clustered 95% CI [0.78, 0.83]`만 범례에 표시했다.
- caption 일치: 실제 그림과 caption 모두 2,286개 안정-ID player-season, WAR, clustered interval을 가리킨다.
- 시각 점검: 축·범례·점·회귀선이 PDF에서 읽히며 clipping이 없다.

### Figure 2

- 기존 오류: caption은 오른쪽 예시 panel을 설명했지만 실제 그림에는 거리분포만 있었다.
- 수정: 왼쪽에 세 거리분포, 오른쪽에 규칙으로 고른 Isaac Paredes–Salvador Perez 2024 radar를 배치했다. 검증값은 각각 641/652 PA, 116/117 wRC+, 3.3/3.3 WAR, 거리 1.797이다.
- caption 일치: left/right panel, pair 수, 선수·시즌·scalar·거리, illustrative 성격을 모두 실제 그림과 맞췄다.
- 시각 점검: 5개 차원 순서는 Contact, Power, Plate Discipline, Defense, Speed로 통일했고 radar label과 legend가 잘리지 않는다.

### Figure 3

- 기존 오류: 수익률을 뜻하는 표현과 linear coefficient caption, random-forest importance panel이 서로 맞지 않았다.
- 수정: 제목이 `Concurrent Dimension-Change Associations with ΔwRC+`인 단일 coefficient plot으로 교체했다. 안정 선수-ID bootstrap 신뢰구간과 0 기준선을 넣었다. 그림과 caption에는 수익률 표현이 없다.
- caption 일치: 실제 panel은 multivariable linear coefficient와 clustered interval만 보여준다.
- 시각 점검: 5개 차원, 오차막대, 0선, 축 제목이 읽히고 잘리지 않는다.

세 PNG를 원본 해상도로 직접 확인했다. 수정 PDF 6쪽 전부를 `manuscript_preview/rendered_pages/page_01.png`부터 `page_06.png`까지 렌더링해 점검했다. Table 1·2는 페이지 폭 안에 있고, Figure 2의 두 panel이 모두 보이며, Figure 3에는 금지된 수익률 문구가 없다. 컴파일 로그에는 undefined reference/citation, overfull box, 오류가 없다. 남은 것은 문단 정렬에서 발생한 underfull 경고뿐이며 내용 손실은 없다.

## G. 논문 주장 변화

### 강하게 주장 가능

- 이번 고정 2024→2025 holdout에서 B2V가 carry-forward보다 낮은 MAE를 보였다.
- 이번 holdout에서 B2V가 calibrated current wRC+보다 낮은 MAE를 보였고 paired CI가 0을 제외했다.
- 같은 season의 wRC+와 WAR를 엄격히 맞춘 다른 선수 프로필은 동일 선수의 연속시즌보다 더 멀고, 무작위 pair보다는 가깝다는 표본 내 순서가 확인됐다.
- PA가 큰 구간에서 모든 차원의 adjacent-season 안정성이 더 높았다.

### 제한적으로 주장 가능

- B2V는 current wRC+를 포함한 뒤에도 보완적인 차년도 정보를 제공한다. 근거는 결합모형 대 scalar-context 비교지만, 한 번의 최종연도 검증에 한정된다.
- 보정 scalar 대비 약 1.2 wRC+ MAE 개선은 통계적으로 구별되나 실용적 크기는 제한적이다.
- 벡터는 해석 가능한 profile interface로 유용하다. 다만 외부 scouting construct가 아니라 공개 시즌 통계의 재표현이다.

### 현재 결과로 주장할 수 없음

- 인과적 player-development 효과, 최적 훈련 목표, causal exchange rate.
- scouting-grade validity 또는 professional 20–80 grade와의 동등성.
- unavailable raw constituent statistics보다 우월하다는 주장.
- Defense 차원 구성이 검증됐다는 주장.
- 모든 시대·리그·선수에게 적용되는 보편적 forecasting 우월성.
- JointVAE의 seed 안정성이나 learned representation의 일반적 우월성.

Abstract는 area correlation이 아니라 calibrated prospective 비교를 중심으로 다시 썼고, 결론도 같은 근거에 맞췄다. 신뢰할 수 있는 원문을 직접 확인한 관련 연구 4편을 추가했다. 검증 메타데이터는 `references/verified_references.md`에 있다.

## H. 남은 한계

- **Raw-statistic baseline:** 구성 원시 통계가 없어 동일 표본 raw-stat Ridge를 실행하지 못했다.
- **Defense construction:** Def와 Fld 평균은 fielding 정보를 반복하고 positional adjustment를 섞으며 playing-time 의존성을 남긴다. 대안 구성은 미해결이다.
- **Shrinkage ablation:** 원시 pre-shrinkage 통계가 없어 none/current/2×를 문자 그대로 재계산하지 못했다.
- **Leave-one-statistic-out:** 최종 차원 점수만 있어 구성 통계 제거 실험을 할 수 없다.
- **JointVAE seed stability:** 한 seed 결과뿐이며 multi-seed 재학습을 하지 않았다.
- **External construct validity:** 작은 기존 scouting pilot은 외부 타당도를 확립하지 못한다.
- **Single final-year test:** 최종 검증은 2024→2025 한 번뿐이다.
- **Survivorship bias:** 양 시즌에 각각 100 PA 이상인 선수가 대상이므로 건강, 역할, 기회, roster retention에 조건부다.
- **Matching dependence:** comparison 재사용이 많아 pair 행은 독립이 아니고, CI는 구성된 matching 절차에 조건부다.
- **미리보기 서식:** 저장소에 `cvpr.sty`와 시스템 LaTeX가 없어 표준 article 서식과 작업공간 내 Tectonic 0.16.9로 미리보기를 만들었다. 제출 시 공식 venue style에서 한 번 더 컴파일해야 한다.

## I. 결과물과 재현 명령

주요 결과물은 다음과 같다.

- 수정 TeX: `manuscript_preview/baseball2vector_en_revision.tex`
- 미리보기 PDF: `manuscript_preview/baseball2vector_en_revision.pdf`
- 원본 비교: `manuscript_preview/original_vs_revision.diff`
- 보정 예측: `results/calibrated_prediction.csv`
- bootstrap 분포: `results/calibrated_prediction_bootstrap.csv`
- paired 비교: `results/paired_model_differences.csv`
- scaling 감사: `results/scaling_audit.json`
- matching row와 요약: `results/matched_profile_distances.csv`, `results/matched_profile_distance_summary.csv`, `results/matching_sample_flow.csv`
- 안정성: `results/stability_by_pa.csv`
- 기존값 재현: `results/reproduction_check.json`
- 입력 provenance: `results/input_provenance.json`
- Figure 1–3: `figures/figure1_alignment_revised.pdf`, `figures/figure2_profile_diversity_revised.pdf`, `figures/figure3_delta_association_revised.pdf`
- LaTeX 표: `tables/prospective_prediction_revised.tex`, `tables/stability_by_pa.tex`, `tables/paired_differences_revised.tex`
- 테스트 로그: `logs/tests.log`
- 컴파일 로그: `logs/latex_compile.log`
- 전 페이지 렌더: `manuscript_preview/rendered_pages/`

작업 폴더에서 전체 재현 명령은 다음과 같다.

```bash
bash experiments/scripts/run_all.sh
```

개별 명령은 다음과 같다.

```bash
python experiments/scripts/run_calibrated_baselines.py
python experiments/scripts/audit_temporal_scaling.py
python experiments/scripts/run_matching_analysis.py
python experiments/scripts/run_stability_analysis.py
python experiments/scripts/generate_revised_figures.py
python experiments/scripts/generate_tables_and_audit.py
uv run --isolated --python 3.12 --with pytest --with pandas --with numpy --with scipy --with scikit-learn python -m pytest -q experiments/tests
(cd manuscript_preview && ../tools/tectonic baseball2vector_en_revision.tex --keep-logs --keep-intermediates)
python experiments/scripts/render_pdf_pages.py
```

마지막으로 권위 원본 `paper/baseball2vector_en.tex`의 해시가 작업 시작 때와 같은 것을 재확인했다. 원본 TeX는 변경되지 않았다.
