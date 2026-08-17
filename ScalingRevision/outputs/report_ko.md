# 20-80 Scale 재조정: MinMax → Mean 50 / SD 10

## 배경

기존 `src/baseball2vec/tools.py`의 `scale_to_2080()`은 `sklearn.MinMaxScaler(20, 80)`을 사용해
관측된 표본의 최솟값→20, 최댓값→80으로 선형 스트레칭하는 방식이었다. 이는 Fangraphs가 설명하는
20-80 스카우팅 스케일의 정의(평균 50, **10점 = 1 표준편차**, 20~80 = ±3SD ≈ 99.7%,
[출처](https://blogs.fangraphs.com/scouting-explained-the-20-80-scouting-scale/))와 다르며,
이상치(단일 선수의 극단값) 한둘이 전체 스케일의 끝점을 정의해버리는 문제가 있었다.

`ScalingRevision/scaling.py`에 다음 함수를 새로 추가했다.

```
grade = 50 + 10 * (x - mean(x)) / std(x)   # then clip to [20, 80]
```

## 데이터 확보 경위

1. **자동 fetch 시도 → 차단.** `baseball2vector` conda env(python 3.12, `requirements.txt`
   버전 그대로: torch 2.11.0 등)를 새로 만들어 `pybaseball.batting_stats()` 재학습을 시도했으나
   `fangraphs.com/leaders-legacy.aspx`가 이 환경을 Cloudflare 챌린지로 403 차단했다. Playwright
   헤드리스 브라우저로도 동일하게 막혔다 (실제 브라우저 핑거프린팅 챌린지, 단순 User-Agent 체크가
   아님). 우회 시도는 하지 않았다. 웹서치로 확인한 결과 이는 pybaseball 자체의 알려진/미해결
   이슈([jldbc/pybaseball#479](https://github.com/jldbc/pybaseball/issues/479))였다.
2. **Baseball Savant(Statcast)는 차단되지 않음을 확인**했으나 (arm-strength/poptime 리더보드 등
   정상 접근), 재현에 필요한 컬럼(`xBA`,`HardHit%`,`Barrel%`,`maxEV`,`EV`)은 이미 FanGraphs
   쪽에도 있어 이번 재현엔 불필요했다.
3. **사용자가 FanGraphs 리더보드에서 수동 export**(`data/raw/fangraphs-leaderboards (1).csv`,
   PA 제한 없음, 2021-2026, 465컬럼)를 제공. 검증 결과:
   - PA≥100으로 필터링하면 정확히 **2309행** — 아카이브된 `v3_results_clean.csv`와 행 수 일치.
   - `NameASCII` 기준 매칭 시 99.0%(2285/2309) 정확히 겹침. 나머지는 전부 이름 표기 차이
     (`A.J. Pollock` vs `AJ Pollock` 등)였고 실제 데이터 누락은 없었다.
   - 동명이인 1건 발견 (`Max Muncy`, 2025 — LAD 고참 선수 vs ATH 선수). 아카이브본은
     `UniqueName`에 `(ATH)`를 붙여 구분해뒀던 것과 동일한 방식으로 `ScalingRevision/build_raw_from_fangraphs_export.py`에서 처리.
   - WAR/wRC+/OPS 값 자체는 완전히 동일하진 않지만 아주 작고 일관된 차이만 있었다
     (WAR 차이 중앙값 0.026, 95.5%가 0.05 이내) — v3 아카이브 이후 FanGraphs가 수비 지표 등을
     소급 수정하면서 생긴 정상적인 드리프트로 판단.
4. 이 데이터를 `data/raw/batting_stats_2021_2025.csv`(파이프라인이 원래 기대하는 캐시 경로)로
   저장하고, **`02_baseline_revision.py` / `03_joint_vae_revision.py`를 실제로 실행 — 진짜
   VAE 재학습**을 완료했다 (`baseball2vector` env, seed=42, 4000 epoch, `src/baseball2vec/joint_vae.py`
   그대로 사용, 코드 변경 없음).

## 결과: Pentagon Area vs WAR / wRC+ / OPS (Pearson r) — 실제 재학습 기준

| Method   | Scale              |    WAR |   wRC+ |    OPS |
|----------|---------------------|-------:|-------:|-------:|
| ZScore   | MinMax (old)         | 0.8058 | 0.6479 | 0.6256 |
| ZScore   | Mean50/SD10 (new)    | 0.7979 | 0.5946 | 0.5729 |
| PCA      | MinMax (old)         | 0.7934 | 0.6336 | 0.6098 |
| PCA      | Mean50/SD10 (new)    | 0.7872 | 0.5804 | 0.5577 |
| JointVAE | MinMax (old)         | 0.7526 | 0.5350 | 0.5163 |
| JointVAE | Mean50/SD10 (new)    | 0.7423 | 0.5022 | 0.4815 |

(참고: 재학습 전, `v3_results_clean.csv`를 재표준화만 해서 얻은 이전 추정치는 ZScore/PCA는
소수점 셋째 자리까지 일치했고, JointVAE만 아카이브된 모델(0.7391/0.5064/0.4861, old)과 이번에
새로 학습된 모델(0.7526/0.5350/0.5163, old) 사이에 차이가 있다 — 같은 seed=42라도 VAE는
torch 버전/난수 스트림에 재현성이 완전히 보장되지 않는 유일한 컴포넌트이기 때문. ZScore/PCA는
결정론적이라 거의 완벽히 일치한다.)

세 방법 모두, 세 성과지표 모두에서 **new(SD10) 스케일의 상관관계가 old(MinMax)보다 소폭 낮다.**
WAR는 거의 변화가 없고(-0.008~-0.010), wRC+/OPS는 조금 더 크게 하락한다(-0.02~-0.05).
이 패턴은 재학습 전/후 동일하게 나타나 — CSV 재사용 방식의 수학적 불변성 논리가 실제로도
성립함을 확인했다.

## 왜 상관관계가 떨어지는가

1. **Clipping으로 인한 정보 손실.** MinMax는 정의상 min→20, max→80이라 클리핑이 절대 발생하지
   않는다. 반면 mean=50/SD=10 방식은 ±3SD를 벗어나는 관측치를 20/80으로 눌러버린다. 실제 클리핑
   비율(`outputs/tables/clip_fractions_retrained.csv`):

   | Method   | Contact | Power | Speed | Defense | Discipline |
   |----------|--------:|------:|------:|--------:|-----------:|
   | ZScore   |   0.39% | 0.61% | 1.04% |   1.30% |      0.35% |
   | PCA      |   0.35% | 0.65% | 1.17% |   1.30% |      0.30% |
   | JointVAE |   0.43% | 1.08% | 1.34% |   1.47% |      0.69% |

   완전한 정규분포라면 ±3SD 바깥은 0.3% 정도여야 하는데, 실제로는 Speed/Defense/Power에서 이보다
   두세 배 높다 (선수 능력치 분포는 정규분포보다 꼬리가 두껍다). 이 구간에 몰린 최상위/최하위
   선수들이 서로 다른 실제 값에도 불구하고 같은 20 또는 80으로 뭉개지면서, 성과지표와의 상관관계
   중 "극단값이 기여하는 부분"이 줄어든다.

2. **MinMax는 표본 극단값에 맞춰 스케일이 정의된다는 점 자체가 상관관계를 부풀릴 수 있다.**
   MinMax에서는 표본 내 정확히 한 명(최댓값)이 그 tool의 80점을 정의한다. 이 스케일링은 통계적
   의미(표준편차) 없이 "이번 표본에서 가장 튀는 사람 vs 가장 못한 사람" 사이를 20~80에 욱여넣는
   것이므로, 표본 구성에 따라 상관관계가 흔들릴 수 있는 다소 임의적인 기준이다.

즉 상관관계 하락은 "새 스케일이 더 나쁘다"는 뜻이 아니라, **기존 MinMax 상관관계 일부가
표본 극단값에 의해 부풀려져 있었다는 뜻에 가깝다.** 하락 폭 자체도 작다 (WAR 기준 1%p 내외,
wRC+/OPS 기준 2~5%p).

## 결론 및 권고

- Fangraphs 정의(평균 50, 1SD=10점)를 따르는 `scale_to_2080_sd()` 채택을 권장한다. 이는
  (a) 실제 스카우팅 스케일의 통계적 의미("70=대략 +2SD")를 보존하고, (b) 단일 이상치가 전체
  선수 집단의 스케일을 재정의하는 MinMax의 취약점을 없앤다.
- 상관관계 하락은 예상된 트레이드오프이며 우려할 수준이 아니다 — 오히려 기존 수치가 다소
  낙관적이었을 가능성을 시사한다.
- 이번 결과는 **실제 raw 데이터로 VAE를 처음부터 재학습해서 얻은 것**이라, CSV 재사용 방식의
  "수학적으로 동일하다"는 주장을 실측으로도 확인했다 (ZScore/PCA는 소수점까지 일치, JointVAE는
  재학습 특성상 근소한 차이).
- 논문 본문에 반영하려면 `src/baseball2vec/tools.py`의 `scale_to_2080`을 이 방식으로 교체하고
  `data/raw/batting_stats_2021_2025.csv`(이번에 확보된 raw 데이터)로 원래의
  `scripts/02_baseline_comparison.py` / `03_joint_vae.py` / `run_all.py`를 다시 돌려 메인
  파이프라인의 `outputs/`, `data/processed/v3_results.csv`를 갱신하면 된다.

## 산출물

- `data/raw/batting_stats_2021_2025.csv` — 이번에 확보/검증된 raw 데이터 (메인 파이프라인과
  공유하는 표준 캐시 경로. 필요시 원본 `scripts/01~04`도 이걸로 그대로 재실행 가능).
- `outputs/data/v3_results_revision.csv` — 재학습된 모델 기준 선수별 old/new grade, Area, σ.
- `outputs/tables/correlation_comparison_retrained.csv`, `clip_fractions_retrained.csv`,
  `v3_summary_old.txt`, `v3_summary_new.txt` — 위 표들의 원본 데이터.
- `outputs/figures/old/`, `outputs/figures/new/` — correlation heatmap, scatter grid,
  **covariance heatmap(이번에 재학습해서 새로 생성)**, radar × 4명(Judge/Soto/Witt Jr./Ohtani),
  uncertainty plot.
- (이전 CSV-재사용 단계 산출물도 `outputs/rescaled_results.csv` 등으로 남아있음 — 위 수치와
  비교용으로 유지.)
