# FanGraphs 수동 Export 시 필요한 컬럼 체크리스트

PA 하한 없이(모든 타자) 2021~2025 **시즌별로 각각** export (5개 CSV). 리더보드에서
아래 컬럼이 전부 보이도록 stat group 탭을 조합하거나 "Customize columns"(컬럼 추가)에서
직접 검색해서 추가하면 됩니다. 정확한 탭 배치는 사이트 UI 버전에 따라 달라질 수 있어서,
이름으로 하나씩 검색해 추가하는 쪽이 제일 안전합니다.

## 식별자 / 필수

| 컬럼 | 비고 |
|---|---|
| `Name` | 선수명 |
| `Team` | 있으면 좋음 (파이프라인 필수는 아님) |
| `PA` | Plate Appearances — 필터링용, 하한 없이 전부 |
| `WAR` | |
| `wRC+` | |
| `OPS` | |

## Contact 그룹

| 컬럼 |
|---|
| `Contact%` |
| `K%` |
| `AVG` |
| `xBA` |
| `SwStr%` |

## Power 그룹

| 컬럼 |
|---|
| `ISO` |
| `SLG` |
| `HardHit%` |
| `Barrel%` |
| `maxEV` |
| `EV` |
| `HR/FB` |

## Speed 그룹

| 컬럼 |
|---|
| `Spd` |
| `BsR` |
| `UBR` |
| `wSB` |

## Defense 그룹

| 컬럼 |
|---|
| `Def` |
| `Fld` |

## Discipline 그룹

| 컬럼 |
|---|
| `BB%` |
| `O-Swing%` |
| `Swing%` |
| `BB/K` |

## 주의사항

- `maxEV`는 Statcast 계열 탭에 있음 (일반 Standard/Advanced 탭에는 없을 수 있음).
- `xBA`도 Statcast 계열. FanGraphs 리더보드에서 Statcast 데이터가 있는 연도만 존재
  (2021~2025는 문제없음).
- `Def`와 `Fld`는 서로 다른 값(총 수비 가치 vs 수비 실점 기여분)이니 둘 다 필요.
- 컬럼명 대소문자/기호(%, /, +)까지 위 표와 최대한 똑같이 맞춰주시면 병합 스크립트가
  바로 매칭됩니다. 다르면 제가 병합할 때 리네임 처리하겠습니다.
- PA 하한(qual)은 걸지 마시고 전부 받으시면 됩니다 — 원래 파이프라인은 qual=100으로
  걸렀지만(`data.py`의 `load_raw(qual=100)`), 그 필터는 병합 후 제가 후처리로 적용할 수
  있습니다.

## 병합 방법

5개 CSV(연도당 1개)를 받으면, 아래 형태로 합쳐서
`data/raw/batting_stats_2021_2025.csv`에 저장해드리겠습니다 (원래 `fetch_raw()`가 만드는
형태와 동일 — 연도별 concat + `Season` 컬럼 추가):

```python
import pandas as pd
frames = []
for year, path in [(2021, "fg_2021.csv"), (2022, "fg_2022.csv"), ...]:
    d = pd.read_csv(path)
    d["Season"] = year
    frames.append(d)
raw = pd.concat(frames).reset_index(drop=True)
raw.to_csv("data/raw/batting_stats_2021_2025.csv", index=False)
```
