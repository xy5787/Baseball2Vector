# SD10Pipeline 최종 결과 (mean=50, SD=10 스케일 단독 적용)

`ScalingRevision/`에서 old(MinMax) vs new(SD10)를 비교 검증한 데이터를 바탕으로, 이번엔
**새 스케일만 적용한 "실전" 파이프라인**을 처음부터 끝까지(02→04) 별도 폴더에서 완전히
새로 돌렸다. VAE도 다시 학습(seed=42, 4000 epoch, 코드 변경 없음).

## 1. Pentagon Area vs WAR/wRC+/OPS

| Method   | Scale        |    WAR |   wRC+ |    OPS |
|----------|--------------|-------:|-------:|-------:|
| ZScore   | MinMax (archived) | 0.8058 | 0.6479 | 0.6256 |
| ZScore   | **SD10 (이번)**    | 0.7979 | 0.5946 | 0.5729 |
| PCA      | MinMax (archived) | 0.7934 | 0.6336 | 0.6098 |
| PCA      | **SD10 (이번)**    | 0.7872 | 0.5804 | 0.5577 |
| JointVAE | MinMax (archived) | 0.7391 | 0.5064 | 0.4861 |
| JointVAE | **SD10 (이번)**    | 0.7423 | 0.5022 | 0.4815 |

`ScalingRevision`의 결론과 동일: 소폭 하락, 우려할 수준 아님 (자세한 이유는 그쪽 리포트 참고).

## 2. ROI 분석 — 가장 중요한 발견

**LR 계수 순위(Power > Contact > Discipline > Defense > Speed)는 old/new 동일** —
"어떤 tool이 wRC+에 가장 큰 영향을 주는가"라는 전역적 결론은 안 바뀌었다.

| 계수 (ΔwRC+ LR) | old (MinMax) | new (SD10) |
|---|---:|---:|
| ΔPower | +3.150 | +2.480 |
| ΔContact | +2.166 | +1.650 |
| ΔDiscipline | +0.704 | +0.534 |
| ΔSpeed | -0.137 | -0.076 |
| ΔDefense | +0.043 | +0.025 |

그러나 **선수 개인별 "지금 어떤 tool을 키워야 하나" 추천(RF10 counterfactual ROI)의
집계 분포는 완전히 뒤집혔다:**

| Best_Tool | old (MinMax) | new (SD10) |
|---|---:|---:|
| Contact | 213 (61.2%) | 98 (28.2%) |
| Power | 135 (38.8%) | 250 (71.8%) |

전역 선형 순위는 그대로인데 개인별 비선형(RF10, 현재 수준×개선폭 상호작용) 추천이 뒤집힌
이유는, 스케일 교체로 선수마다 tool별 "현재 위치"가 서로 다른 폭으로 이동했기 때문 —
RF10 모델이 각 선수의 새 현재 위치에서 한계 개선 효과를 다시 계산한 결과다.

- Byron Buxton Contact ROI: old +14.02 → new +10.87
- 모델 적합도(CV R²) 자체는 old/new 거의 동일 (ΔwRC+ LR: 0.7068→0.7043, RF10: 0.6773→0.6781) —
  즉 예측력은 안 변했는데 "무엇을 추천하는가"만 바뀜.

## 3. 정성적 예시 선수 (실제 재학습 결과, JointVAE 등급)

| 선수 (시즌) | Tool | Old | New | 비고 |
|---|---|---:|---:|---|
| **Bobby Witt Jr. (2024)** | 전체 5-tool | — | — | Pentagon Area 6921→11463 (+66%), 5개 tool 전부 +12~20점 상승. 가장 극적인 "전체 프로필이 저평가돼있던" 사례. |
| Bobby Witt Jr. (2024) | Power | 60.0 | 80.0 | |
| Bobby Witt Jr. (2024) | Speed | 50.7 | 70.4 | |
| **Luis Arraez (2021)** | Contact | 62.6 | **80.0** | 전형적 "진짜 80 컨택" 스토리 — 배팅 타이틀 상비군, 직관적으로 설득력 있는 예시 |
| **Aaron Judge (2021)** | Power | 58.3 | **79.9** | 62홈런 시즌(2022) 이전이지만 이미 실질적으로 3SD급 파워였음을 새 스케일이 포착 |
| **Shohei Ohtani (2024)** | Power | 80.0 | 80.0 | 대조 사례 — 이미 두 스케일 모두에서 만점 (40/40+ 시즌) |
| Shohei Ohtani (2024) | Discipline | 63.0 | 80.0 | Power와 달리 Discipline은 저평가돼 있었음 |
| **Aaron Judge (2024)** | Defense | 35.1 | 34.9 | 대조 사례 — 모든 게 오르진 않는다는 뉘앙스 (소폭 하락) |

**세미나 발표용 추천 조합**: Witt Jr.(전체 상승, 가장 극적) + Arraez(직관적 단일 tool 스토리)
+ Ohtani(대조 — 이미 맞던 것과 저평가됐던 것이 공존) 3인이 서로 다른 각도의 스토리를 보여줘서
같이 쓰면 좋을 것 같다.

## 4. "평균급" 선수들 — 비슷한 WAR, 다른 spike point

WAR가 비슷해도(1.6~2.3 범위) pentagon의 어느 축이 튀어나오는지는 선수마다 완전히 다르다는
걸 보여주는 5명. Speed/Contact/Power/Defense/Discipline 각각 하나씩 spike하는 선수를 골랐다
(JointVAE 등급, old=MinMax archived, new=SD10):

| 선수 (시즌) | WAR | wRC+ | Spike tool | Old | New |
|---|---:|---:|---|---:|---:|
| **J.D. Martinez (2023)** | 2.34 | 133 | Power | 57.6 | 76.6 |
| **Michael A. Taylor (2021)** | 1.96 | 75 | Defense | 66.5 | **80.0** |
| **Luis Arraez (2021)** | 1.59 | 105 | Contact | 62.6 | **80.0** |
| **Kyle Schwarber (2022)** | 2.21 | 129 | Discipline | 59.0 | 75.6 |
| **Elly De La Cruz (2023)** | 1.88 | 85 | Speed | 62.6 | **80.0** |

포인트:
- Martinez와 Taylor는 WAR가 거의 같지만(2.34 vs 1.96) **가치의 출처가 정반대**다 — Martinez는
  타격(wRC+ 133)만으로, Taylor는 수비(wRC+ 75, 리그 평균 이하 타격)만으로 비슷한 WAR를 만든다.
  Pentagon을 보면 이게 한눈에 보이지만 WAR 숫자 하나만 보면 완전히 안 보인다.
- new 스케일에서 spike가 old보다 훨씬 더 뚜렷해진다 (Taylor Defense 66.5→80.0, Arraez Contact
  62.6→80.0, De La Cruz Speed 62.6→80.0 — 셋 다 old에서는 "높은 편" 정도였다가 new에서 명확한
  **80(만점)**으로 찍힌다). 반대로 Schwarber는 Defense가 old 24.3→new 20.0(바닥)으로 떨어져서
  "3TO 슬러거, 수비는 약점"이라는 이야기도 더 선명해진다.
- radar 그림: `outputs/figures/average_players_new/`(새 스케일), `average_players_old/`(비교용
  구 스케일) — 5명 전부.

## 산출물

- `data/processed/v3_results.csv` — 최종 선수별 결과 (2309행), 새 스케일 단독 적용.
- `outputs/figures/` — v2/v3 correlation heatmap, scatter grid, covariance heatmap,
  radar × 4 (Judge/Soto/Witt Jr./Ohtani, 2024시즌), uncertainty plot,
  roi_coefficients, roi_partial_dependence, roi_delta_scatter, roi_individual_best_tool.
- `outputs/tables/` — `correlations.csv`, `v3_summary.txt`, `lr_results.csv`,
  `individual_roi.csv`, `delta_dataset.csv`.
