# Baseball2Vector 최종 학술 수정 재현 패키지

이 디렉터리는 원본 원고와 기존 작업물을 건드리지 않고 생성한 격리 실행본이다.

## 핵심 결론

고정된 2024→2025 테스트 358명에서 forecast-safe B2V Ridge는 MAE 18.027로 calibrated current-wRC+ Ridge의 19.220보다 1.193점 낮았다. 동일 테스트 선수에 대한 paired bootstrap 95% CI는 [0.403, 2.012]이므로 이 표본에서는 차이가 통계적으로 구별된다. 개선 폭은 scalar baseline MAE의 약 6.2%로 크지 않다. Combined model도 scalar-context보다 개선되지만, B2V-context에 current wRC+를 더한 이득은 명확하지 않다.

## 주요 경로

- 수정 원고: `manuscript_preview/baseball2vector_en_revision.tex`
- 컴파일 PDF: `manuscript_preview/baseball2vector_en_revision.pdf`
- 원본 대비 diff: `manuscript_preview/original_vs_revision.diff`
- 한국어 보고서: `report_ko.md`
- 통합 실행: `run_all.sh`
- 분석 코드: `experiments/scripts/`
- 검증 테스트: `experiments/tests/`
- 기계 판독 결과: `results/`
- LaTeX 표: `tables/`
- 수정 그림: `figures/`
- 실행 로그: `logs/`

## 입력과 보존

- 원본 원고: `../../paper/baseball2vector_en.tex`
- 원본 원고 SHA-256: `a873b4beb3b9acbcdd44c5f13d859b8c2a326241234ef49794fe20ff65e9f00d`
- 처리 데이터: `../../data/processed/v3_results.csv`
- 처리 데이터 SHA-256: `70bea280ec725ac78376e94edf41fae6fd036fe8b7a0b71270869eb1cfdf0873`
- stable-ID 분석 입력: `data_intermediate/player_seasons_with_ids.csv`
- stable-ID 입력 SHA-256: `0c810b9ea6aab754cd389ae7c5ab401a139bb251c6490a84b6e4b71d6cd07963`

원본 TeX는 읽기만 했으며 완료 시점에도 위 SHA-256이 유지되었다.

## 재현 명령

저장소 루트에서:

```bash
./final_academic_revision/run_20260808_055335/run_all.sh
```

이 명령은 순서대로 calibrated baseline, temporal scaling audit, deterministic matching, PA-stratified stability, figure/table 생성, 15개 테스트를 실행한다. seed는 42, bootstrap은 2,000회, Ridge alpha grid는 `0.01, 0.1, 1, 10, 100, 1000`이다.

PDF 재컴파일:

```bash
cd final_academic_revision/run_20260808_055335/manuscript_preview
../.conda-tectonic/bin/tectonic baseball2vector_en_revision.tex --keep-logs --keep-intermediates
../.conda-tectonic/bin/pdftoppm -png -r 120 baseball2vector_en_revision.pdf page_renders/page
```

공식 `cvpr.sty`는 CVPR author-kit 커밋 `291758547e923160eb4d37079b7b9f0dfce82355`에서 복사했다. 실행 환경은 `logs/environment_used.txt`에 기록했다.

## 모델 및 검증 규칙

- 학습: 2021→2022, 2022→2023, 2023→2024 전이 1,056개
- 테스트: 2024→2025 전이 358개
- 모든 모델의 테스트 stable player ID가 동일
- inner CV: stable player ID 기준 5-fold GroupKFold, training MAE로 alpha 선택
- pipeline: training-only median imputation, training-only StandardScaler, Ridge
- 예측용 B2V: 입력 시즌 코호트 내 표준화한 forecast-safe feature
- 불확실성: held-out player bootstrap 2,000회
- matching: 같은 시즌, 다른 stable ID, |ΔwRC+|≤5, |ΔWAR|≤0.2
- matching tie-break: 12자리 반올림 scalar gap, stable ID, source row
- 안정성 PA 구간: min(PA_t, PA_t+1)

## 환경

재실행 환경은 Python 3.12.3, NumPy 2.5.1, pandas 3.0.5, SciPy 1.18.0, scikit-learn 1.9.0, Matplotlib 3.11.1이다. PDF는 Tectonic 0.17.0으로 생성하고 Poppler 26.07.0으로 렌더링했다.

## 주의

원시 constituent statistics와 pre-min-max dimension score가 없으므로 raw-statistic Ridge, shrinkage ablation, leave-one-statistic-out, Defense 재구성, JointVAE multi-seed 분석은 수행하지 않았다. 이 부재는 음의 결과가 아니라 미해결 검증 범위다.
