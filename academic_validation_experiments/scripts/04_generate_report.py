"""Generate Stage A tables, preview material, summary JSON, and Korean report."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import matplotlib
import numpy
import pandas
import scipy
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validation_utils import EXP_ROOT, REPO_ROOT, ensure_dirs, load_config, sha256_file  # noqa: E402


def metric_cell(frame: pandas.DataFrame, model: str, metric: str, digits: int = 3) -> str:
    row = frame[frame["model"].eq(model) & frame["metric"].eq(metric)].iloc[0]
    if pandas.isna(row["estimate"]):
        return "—"
    return f"{row['estimate']:.{digits}f} [{row['ci_low']:.{digits}f}, {row['ci_high']:.{digits}f}]"


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def tex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("+", r"+")


def main() -> None:
    ensure_dirs()
    cfg = load_config()
    results = EXP_ROOT / "results"
    pred = pandas.read_csv(results / "prospective_prediction.csv")
    pairwise = pandas.read_csv(results / "prospective_prediction_pairwise.csv")
    stability = pandas.read_csv(results / "representation_stability.csv")
    match_summary = pandas.read_csv(results / "matched_profile_distance_summary.csv")
    match_diff = pandas.read_csv(results / "matched_profile_distance_differences.csv")
    match_sens = pandas.read_csv(results / "matched_profile_sensitivity.csv")
    illustrative = pandas.read_csv(results / "illustrative_matched_pair.csv").iloc[0]
    sample_flow = pandas.read_csv(results / "sample_flow.csv")
    reproduction = json.loads((results / "headline_reproduction.json").read_text())
    provenance = json.loads((results / "input_provenance.json").read_text())

    wrc = pred[pred["target"].eq("wRC+_t1") & pred["status"].eq("ok")]
    war600 = pred[pred["target"].eq("WAR600_t1") & pred["status"].eq("ok")]
    model_labels = {
        "league_mean": "League mean",
        "carry_forward": "Carry-forward",
        "b2v_ridge": "B2V Ridge",
        "b2v_age_pa_ridge": "B2V + age + PA Ridge",
    }
    prediction_table_rows = []
    for model, label in model_labels.items():
        prediction_table_rows.append(
            {
                "model": label,
                "mae": metric_cell(wrc, model, "mae"),
                "r2": metric_cell(wrc, model, "r2"),
                "spearman": metric_cell(wrc, model, "spearman"),
            }
        )
    prediction_table_rows.append({"model": "Raw stats + age + PA Ridge", "mae": "not run", "r2": "not run", "spearman": "not run"})
    prediction_table = pandas.DataFrame(prediction_table_rows)
    prediction_table.to_csv(EXP_ROOT / "tables" / "prospective_prediction_table.csv", index=False)

    stability_table = stability[stability["metric"].eq("spearman")].copy()
    stability_table["formatted"] = stability_table.apply(
        lambda r: f"{r.estimate:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]", axis=1
    )
    stability_pivot = stability_table.pivot(index="dimension", columns="subset", values="formatted").reset_index()
    stability_pivot = stability_pivot[["dimension", "overall", "100-249", "250-499", "500+"]]
    stability_pivot.to_csv(EXP_ROOT / "tables" / "stability_table.csv", index=False)
    match_summary.to_csv(EXP_ROOT / "tables" / "matched_profile_distance_table.csv", index=False)

    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Prospective prediction of 2025 wRC+ from 2024 information only. Brackets are player-bootstrap 95\% confidence intervals ($n_{train}=1{,}056$, $n_{test}=358$).}",
        r"\label{tab:prospective}",
        r"\scriptsize",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Model & MAE & $R^2$ & Spearman $\rho$ \\",
        r"\midrule",
    ]
    for row in prediction_table_rows:
        tex_lines.append(
            f"{tex_escape(row['model'])} & {row['mae']} & {row['r2']} & {row['spearman']} \\\\"
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    tex_lines.extend(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Year-to-year Spearman stability of the Z-score dimensions. Brackets are player-clustered bootstrap 95\% confidence intervals.}",
            r"\label{tab:stability}",
            r"\scriptsize",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Dimension & Overall & 100--249 PA & 250--499 PA & $\geq$500 PA \\",
            r"\midrule",
        ]
    )
    for row in stability_pivot.itertuples(index=False):
        tex_lines.append(f"{row.dimension} & {row.overall} & {getattr(row, '_2')} & {getattr(row, '_3')} & {getattr(row, '_4')}" + r" \\")
    tex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (EXP_ROOT / "manuscript_preview" / "proposed_tables.tex").write_text("\n".join(tex_lines))

    captions = (
        "\\caption{Standardized five-dimensional profile distances for same-player adjacent-season "
        "pairs, scalar-matched different-player pairs, and random same-season PA-stratified pairs. "
        "Scalar matching uses only $|\\Delta\\mathrm{wRC+}|\\leq5$ and "
        "$|\\Delta\\mathrm{WAR}|\\leq0.2$; Baseball2Vector distance is not used for matching.}\n"
    )
    (EXP_ROOT / "manuscript_preview" / "proposed_figure_captions.tex").write_text(captions)
    preview = r"""\documentclass{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\begin{document}
\section*{Baseball2Vector Stage A validation preview}
\input{proposed_tables.tex}
\begin{figure}[t]
\centering
\includegraphics[width=0.9\linewidth]{../figures/matched_profile_distances.png}
\input{proposed_figure_captions.tex}
\end{figure}
\end{document}
"""
    (EXP_ROOT / "manuscript_preview" / "validation_preview.tex").write_text(preview)

    proposed_diff = r"""--- paper/baseball2vector_en.tex (authoritative; unchanged in Stage A)
+++ academic_validation_experiments/manuscript_preview/proposed integration
@@ Title
-Baseball2Vector: An Interpretable Five-Tool Representation of MLB Position Players
+Baseball2Vector: An Interpretable Five-Dimensional Skill Representation of MLB Position Players
@@ Transition terminology
-the closest available approximation to a genuinely prospective forecast
+a temporally held-out evaluation of a contemporaneous transition association; it is not a forecast because season-(t+1) dimension changes enter the predictors
@@ Proposed new subsection: Prospective and Robustness Validation
+Predict 2025 wRC+ from 2024-only information after training on 2021->2024 transitions.
+Report n_train=1,056 and n_test=358; include league-mean, carry-forward, B2V Ridge,
+B2V+age+PA Ridge, and an explicit unavailable raw-stat row until the raw export is restored.
+Report dimension stability by PA stratum and scalar-matched profile-distance distributions.
@@ Data identity
-Resolve identical names using performance context.
+Use stable FanGraphs/MLBAM IDs. Exclude unresolved collisions from longitudinal analyses.
@@ Shrinkage sensitivity
-Claim that stronger shrinkage changes correlations by at most 0.003.
+Remove this claim until literal no/current/2x-prior recomputation is possible from raw statistics.
"""
    (EXP_ROOT / "manuscript_preview" / "proposed_changes.diff").write_text(proposed_diff)

    environment = "\n".join(
        [
            f"python={platform.python_version()}",
            f"executable={sys.executable}",
            f"platform={platform.platform()}",
            f"pandas={pandas.__version__}",
            f"numpy={numpy.__version__}",
            f"scikit-learn={sklearn.__version__}",
            f"scipy={scipy.__version__}",
            f"matplotlib={matplotlib.__version__}",
            f"source_sha256={provenance['source_sha256']}",
            f"manuscript_sha256={sha256_file(REPO_ROOT / cfg['authoritative_manuscript'])}",
            f"seed={cfg['seed']}",
            f"bootstrap_replicates={cfg['bootstrap_replicates']}",
        ]
    )
    (EXP_ROOT / "logs" / "environment_used.txt").write_text(environment + "\n")

    def pvalue(target: str, candidate: str, baseline: str, metric: str, field: str) -> float:
        return float(pairwise[
            pairwise["target"].eq(target)
            & pairwise["candidate"].eq(candidate)
            & pairwise["baseline"].eq(baseline)
            & pairwise["metric"].eq(metric)
        ].iloc[0][field])

    matched_map = match_summary.set_index("pair_type")
    diff_map = {(r.left, r.right): r for r in match_diff.itertuples(index=False)}
    summary_json = {
        "headline_reproduction": reproduction,
        "identity": {
            "matched_player_seasons": int(sample_flow.set_index("stage").loc["stable_id_matched_player_seasons", "n_rows"]),
            "matched_players": int(sample_flow.set_index("stage").loc["stable_id_matched_player_seasons", "n_players"]),
            "excluded_player_seasons": int(sample_flow.set_index("stage").loc["excluded_ambiguous_or_unmatched", "n_rows"]),
        },
        "experiments": {
            "prospective_prediction": "partial_success_raw_stat_baseline_unavailable",
            "temporal_stability": "success",
            "shrinkage_sensitivity": "not_run_missing_raw",
            "leave_one_stat_out": "not_run_missing_raw",
            "defense_variants": "conceptual_audit_only_missing_raw",
            "matched_profile_diversity": "success",
            "jointvae_five_seed": "skipped_missing_raw_and_core_incomplete",
        },
        "primary_wrc_prediction": {
            model: {
                metric: float(wrc[wrc["model"].eq(model) & wrc["metric"].eq(metric)].iloc[0]["estimate"])
                if not pandas.isna(wrc[wrc["model"].eq(model) & wrc["metric"].eq(metric)].iloc[0]["estimate"])
                else None
                for metric in ["mae", "r2", "spearman"]
            }
            for model in model_labels
        },
        "profile_distance_medians": {key: float(row["median"]) for key, row in matched_map.iterrows()},
        "illustrative_pair": {
            "focal": illustrative["focal_name"],
            "comparison": illustrative["comparison_name"],
            "season": int(illustrative["focal_season"]),
            "wrc": [float(illustrative["focal_wrc"]), float(illustrative["comparison_wrc"])],
            "war": [float(illustrative["focal_war"]), float(illustrative["comparison_war"])],
            "distance": float(illustrative["profile_distance"]),
        },
    }
    (results / "additional_experiments_summary.json").write_text(json.dumps(summary_json, indent=2, ensure_ascii=False))

    pred_md_rows = []
    for row in prediction_table_rows:
        pred_md_rows.append([row["model"], row["mae"], row["r2"], row["spearman"]])
    pred_md = md_table(["모델", "MAE", "R²", "Spearman ρ"], pred_md_rows)
    stability_md = md_table(
        ["차원", "전체", "100–249 PA", "250–499 PA", "500+ PA"],
        [[r.dimension, r.overall, getattr(r, "_2"), getattr(r, "_3"), getattr(r, "_4")] for r in stability_pivot.itertuples(index=False)],
    )
    match_md = md_table(
        ["비교군", "pair 수", "중앙값", "IQR", "95% CI"],
        [
            [
                idx,
                int(row["n_pairs"]),
                f"{row['median']:.3f}",
                f"[{row['q1']:.3f}, {row['q3']:.3f}]",
                f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]",
            ]
            for idx, row in matched_map.iterrows()
        ],
    )

    report = f"""# Baseball2Vector 학술 검증 실험 보고서 (Stage A)

> 결론부터 말하면, 2024 정보만으로 2025 wRC+를 예측한 Baseball2Vector Ridge는 league mean과 carry-forward보다 명확히 나았지만(R² 0.28–0.29), age와 PA를 추가한 이득은 거의 없었습니다. 유사한 wRC+/WAR를 가진 선수들의 벡터 거리는 동일 선수의 연속 시즌보다 컸고 무작위 pair보다는 작아, ‘같은 가치, 다른 프로필’ 주장을 지지했습니다. 반면 원시 구성 통계량이 저장소에 없어 raw-stat Ridge, 실제 shrinkage 비교, leave-one-stat-out, Defense 변형은 실행할 수 없었습니다. 따라서 현 단계에서 표현의 원시 통계 대비 우월성이나 shrinkage/Defense의 강건성을 주장해서는 안 됩니다.

## 12.1 실행 개요

- 생성 폴더: `{EXP_ROOT}`
- 권위 원고: `{REPO_ROOT / cfg['authoritative_manuscript']}` — Stage A 시작/종료 SHA-256 `{provenance['manuscript_sha256_stage_a_start']}`로 동일함을 테스트했습니다.
- 원본 입력: `{REPO_ROOT / cfg['source_data']}` (SHA-256 `{provenance['source_sha256']}`)
- 기간/선정: 2021–2025, 시즌별 PA ≥100. 시즌별 행 수는 463/469/461/455/461, 총 2,309 player-season입니다.
- 원본 파일에는 826개 고유 표시명이 있으나 `Max Muncy (2025)`가 두 행이라 `UniqueName`은 2,308개뿐입니다. Chadwick Register의 FanGraphs/MLBAM ID로 보수적으로 매핑해 2,286행·818명을 확보했고, ID를 유일하게 결정할 수 없는 23행은 모든 종단 분석에서 제외했습니다. Max Muncy 2025 두 행은 성적이나 포지션으로 추정하지 않고 모두 제외했습니다.
- 안정-ID 연속 시즌 pair는 1,414개(567명)입니다. 기존 이름 기반 코드는 1,427개를 만들므로 종단 표본 수가 13개 줄었습니다.
- 기존 핵심 수치 재현: Z-score area–WAR Pearson r = {reproduction['zscore_area_war_r']:.6f}, 이름 기반 ΔwRC+ 선형회귀 pooled 5-fold R² = {reproduction['name_keyed_delta_wrc_lr_cv_r2']:.6f}. 논문 반올림값 0.81과 0.71을 재현했습니다.
- 성공: 진짜 차년도 예측(B2V 및 두 단순 baseline), PA별 안정성, scalar-matched profile diversity.
- 부분 성공: Defense는 정의 감사와 현재 저장 점수의 안정성만 가능했습니다.
- 실패/미실행: raw-stat Ridge, no/current/2× shrinkage의 문자 그대로의 비교, 22개 구성 통계량 leave-one-out, Def/Fld/innings-normalized 실증 비교. 이유는 `data/raw/batting_stats_2021_2025.csv`가 없고 처리 파일에는 원시 통계가 없기 때문입니다.
- 선택 실험: JointVAE 5-seed 재학습은 원시 입력 부재 및 핵심 ablation 미완료 때문에 생략했습니다. 외부 scouting pilot 확장도 새 대규모 외부 수집 금지 조건에 따라 생략했습니다.

## 12.2 실험 1: 차년도 예측

학습은 2021→2022, 2022→2023, 2023→2024의 1,056개 안정-ID 전이, 최종 테스트는 2024→2025의 358명입니다. 모든 예측 변수는 시즌 t에서만 가져왔습니다. Ridge alpha는 학습 자료 내부에서 선수-ID `GroupKFold` 5-fold와 MAE로 선택했고, imputation/standardization도 각 학습 fold에서만 적합했습니다. 시즌 t 벡터를 그 시즌 전체 리그 분포로 계산하는 것은 시즌 종료 시 이용 가능하다고 가정했습니다. 저장된 전기간 min–max 변환은 고정 affine 변환이며 Ridge 앞의 train-only 표준화로 상쇄되지만, 향후 재생성 시에는 원시 점수와 변환 파라미터를 명시적으로 보존하는 편이 낫습니다.

### 2025 wRC+ 예측

{pred_md}

괄호는 테스트 선수 단위 paired bootstrap 2,000회의 95% CI입니다. League-mean의 Spearman은 상수 예측이라 정의되지 않습니다.

- **League mean보다 나았는가?** 예. B2V-only의 MAE는 22.27→18.02로 4.25 감소(95% CI [2.88, 5.63]), R²는 -0.001→0.280입니다. B2V+age+PA도 MAE 4.26 감소([2.92, 5.61]), R² 0.289입니다.
- **Carry-forward보다 나았는가?** 예. B2V+age+PA의 MAE 개선은 4.33([2.89, 5.87]), R² 개선은 0.391([0.261, 0.540]), Spearman 개선은 0.073([0.011, 0.138])입니다.
- **Raw-stat Ridge보다 나았는가?** 답할 수 없습니다. 구성 원시 통계가 없어서 동일 표본의 raw-stat baseline을 실행하지 못했습니다. 이는 핵심 미해결 비교입니다.
- **age+PA가 의미 있게 추가 개선했는가?** 아니라고 보는 편이 타당합니다. B2V-only 대비 MAE 개선은 0.005([−0.371, 0.370]), R² 개선은 0.009([−0.013, 0.031])로 사실상 0입니다.
- **통계적·실용적 의미:** 단순 baseline 대비 MAE가 약 19% 줄고 rank correlation이 약 0.52인 것은 실용적으로 의미 있는 예측 신호입니다. 그러나 R² 0.29는 다음 시즌 변동의 대부분이 여전히 설명되지 않음을 뜻합니다.
- **WAR600 보조 결과:** B2V+age+PA는 MAE 1.553 [1.420, 1.686], R² 0.223 [0.134, 0.305], Spearman 0.439 [0.350, 0.529]였습니다. Carry-forward의 R²는 −0.140이었습니다. 이 보조 결과도 raw-stat 비교가 없습니다.
- **선택·생존 편향:** t와 t+1 모두 PA≥100인 선수만 포함하므로 은퇴, 심각한 부상, 마이너 강등, 벤치화, 짧은 콜업을 체계적으로 제외합니다. 결과는 “다음 시즌에도 100 PA 이상 기록한 선수”에 조건부이며 전체 MLB 선수 집단의 예측 성능이 아닙니다.

## 12.3 실험 2: 표현 안정성과 ablation

차원별 year-to-year Spearman은 다음과 같습니다.

{stability_md}

- 전체에서는 Discipline 0.760, Contact 0.734, Power 0.730이 가장 안정적이고, Speed 0.626, Defense 0.568이 낮았습니다.
- 저PA에서는 모든 차원이 약해졌습니다. 특히 Defense 0.463, Speed 0.533, Power 0.572였고, 500+ PA에서는 각각 0.661, 0.703, 0.789였습니다. 표본량이 안정성에 실질적으로 관련됩니다.
- 전체 raw 20–80 cosine 중앙값은 0.9955 [0.9952, 0.9958]이지만, 모든 좌표에 큰 양의 절편이 있어 기하학적으로 과대평가됩니다. 표준화 Euclidean 거리 중앙값 0.714 [0.696, 0.726]와 차원별 Spearman을 주 해석 대상으로 권합니다.
- **Shrinkage:** 현재 점수에서 저PA 차원 Spearman 평균은 {stability[(stability['subset'].eq('100-249')) & (stability['metric'].eq('spearman'))]['estimate'].mean():.3f}입니다. 하지만 no shrinkage와 2× prior는 원시 통계가 없어 재계산하지 못했습니다. 기존 revision의 composite-score 추가 수축 proxy는 실제 pseudo-PA 변경이 아니므로, 원고의 “상관 변화 ≤0.003” 문장은 삭제하거나 원시 자료 복구 뒤 다시 검증해야 합니다.
- **Leave-one-stat-out:** Contact 5개, Power 7개, Speed 4개, Defense 2개, Discipline 4개 총 22개 모두 실행 불가입니다. 특정 통계 의존성에 관한 결론을 내릴 수 없습니다.
- **Defense 감사:** FanGraphs에서 `Def = Fld + positional adjustment`입니다. 현재 정의가 z(Def)와 z(Fld)를 평균하므로 Fld 신호가 두 번 들어가고 포지션 조정도 섞입니다. 둘 다 누적 run 값이라 출장 기회 의존성도 있습니다. 현재 Defense의 안정성은 0.568이지만, 이는 구성의 타당성을 입증하지 않습니다. 원시 자료가 복구되면 최소한 `Fld only`, `Def only`, `Fld/defensive innings`를 비교해야 합니다. 개념적으로는 중복과 포지션 조정을 피하는 innings-normalized Fld가 “skill”에 더 가깝지만, 현 결과만으로 최종 권장 정의를 확정할 수 없습니다. 정의 근거: [FanGraphs Def](https://library.fangraphs.com/defense/def/), [Introducing FanGraphs Stats: Offense and Defense](https://blogs.fangraphs.com/introducing-fangraphs-stats-offense-and-defense/).

## 12.4 실험 3: 유사 가치 선수의 프로필 다양성

매칭은 같은 시즌·다른 안정 ID에서 `|ΔwRC+|≤5`, `|ΔWAR|≤0.2`를 만족하는 후보 중 시즌 내 표준편차로 표준화한 두 scalar 차이가 최소인 선수를 선택했습니다. 벡터 거리는 매칭에 사용하지 않았습니다. 모든 focal player-season에 대한 방향성 nearest-neighbor이며 comparison 재사용을 허용했습니다.

{match_md}

거리 중앙값 차이는 scalar-matched−same-player = {diff_map[('scalar_matched_intersection', 'same_player_adjacent')].median_difference:.3f} [{diff_map[('scalar_matched_intersection', 'same_player_adjacent')].ci_low:.3f}, {diff_map[('scalar_matched_intersection', 'same_player_adjacent')].ci_high:.3f}], random−scalar-matched = {diff_map[('random_same_season_pa_stratum', 'scalar_matched_intersection')].median_difference:.3f} [{diff_map[('random_same_season_pa_stratum', 'scalar_matched_intersection')].ci_low:.3f}, {diff_map[('random_same_season_pa_stratum', 'scalar_matched_intersection')].ci_high:.3f}]입니다. 따라서 사전 가설 `same-player < scalar-matched < random`은 지지됐습니다. wRC+-only 2,274 pair의 중앙값은 {match_sens.set_index('pair_type').loc['matched_wrc_only', 'median']:.3f}, WAR-only 2,262 pair는 {match_sens.set_index('pair_type').loc['matched_war_only', 'median']:.3f}으로 결론 방향이 유지됐습니다.

사전 규칙(양쪽 400 PA 이상, 유효 매칭, 거리 90백분위에 가장 가까움)으로 고른 설명용 사례는 **{illustrative['focal_name']}–{illustrative['comparison_name']} ({int(illustrative['focal_season'])})**입니다. PA {int(illustrative['focal_pa'])}/{int(illustrative['comparison_pa'])}, wRC+ {int(illustrative['focal_wrc'])}/{int(illustrative['comparison_wrc'])}, WAR {illustrative['focal_war']:.1f}/{illustrative['comparison_war']:.1f}, 표준화 거리 {illustrative['profile_distance']:.3f}입니다. 이 사례는 그림 설명용이며 일반화나 인과 추론의 증거가 아닙니다. 기존 Witt–Rutschman 예시는 가치가 맞지 않으므로, 이 pair로 교체하는 것이 논문의 주장과 일치합니다.

## 12.5 논문에 반영할 수 있는 결론

### 강하게 주장 가능

- 조건부 생존 표본에서 시즌-t Baseball2Vector가 league mean과 carry-forward보다 다음 시즌 wRC+ 예측 오차를 줄였다.
- 차원 안정성은 PA에 따라 달라지고 Defense/Speed가 Contact/Power/Discipline보다 낮다.
- 비슷한 wRC+/WAR의 다른 선수들은 동일 선수 연속 시즌보다 더 다양한 5차원 프로필을 보이며, 무작위 pair보다는 가깝다.

### 제한적으로 주장 가능

- Baseball2Vector는 다음 시즌 성과에 관한 신호를 보유한다(R²≈0.29). 다만 raw-stat baseline 부재와 생존 편향을 함께 명시해야 합니다.
- Alex Verdugo–Luis Robert Jr. pair는 profile diversity의 설명용 사례로 사용할 수 있습니다.

### 추가 검증 필요

- Baseball2Vector가 원시 통계를 압축하면서 예측력을 유지하거나 개선하는지.
- 현재 shrinkage가 저PA 안정성을 개선하는지, 2× shrinkage가 고PA 변이를 과도하게 압축하는지.
- 어떤 구성 통계가 차원을 지배하는지와 Defense 대체 정의의 강건성.
- JointVAE seed 안정성과 2025 데이터의 정확한 추출 시점.

### 현재 결과로 주장하면 안 됨

- Δv 회귀를 prospective forecast라고 부르는 것. predictor에 시즌 t+1 정보가 들어가므로 contemporaneous transition association입니다.
- 합성 +5 변화의 인과 효과, 개발 처방, ROI.
- B2V가 raw-stat 모델보다 우수하다는 주장.
- 현재 shrinkage가 검증됐거나 현재 Defense 정의가 타당하다는 주장.

## 12.6 논문 수정 제안

| 섹션 | 현재 문제 | 제안 수정 | 근거 | 방식 | 길이 영향 |
| --- | --- | --- | --- | --- | --- |
| Title | “Five-Tool”이 canonical Hit–Power–Run–Field–Throw를 암시 | “Five-Dimensional Skill Representation”으로 변경 | taxonomy 불일치 | 교체 | 0줄 |
| Abstract | 동시 Δ 회귀 0.71이 중심이고 진짜 예측 근거가 없음 | 2024→2025 wRC+ B2V+age+PA R² 0.289, MAE 18.02와 생존 조건을 추가; Δ 회귀는 association으로 축소 | 실험 1 | 교체+추가 | +2–3문장 |
| Introduction | contribution에 prospective evidence 없음 | “t-only prospective prediction against mean/carry baselines” 추가, raw-stat 미완료 명시 | 실험 1 | 1 bullet 교체 | +2줄 |
| Methods/Data | 이름+성적으로 Max Muncy를 해결 | Chadwick stable ID, 23행 보수적 제외, 종단 1,414 pair로 수정 | ID 감사 | 교체 | +4–5줄 |
| Methods | forward-time Δv holdout을 forecast에 가깝다고 표현 | 모든 Δv 분석을 contemporaneous transition association으로 정의; 별도 t→t+1 prospective protocol 추가 | 누수 개념 구분/실험 1 | 교체+추가 | +0.4쪽 |
| Results | true forecast 및 baseline 부재 | compact prospective table 추가, age+PA null과 raw-stat 미실행 포함 | 실험 1 | 추가 | 표 1개, +0.2쪽 |
| Results | 안정성/PA 효과 없음 | PA별 차원 Spearman을 compact summary로 추가 | 실험 2 | 추가 | 표 또는 4–5문장 |
| Results/Qualitative | Witt–Rutschman이 scalar matched가 아님 | Verdugo–Robert 2024로 교체하고 selection rule 명시 | 실험 3 | 그림/문단 교체 | 거의 동일 |
| Results | profile diversity가 정량 검증되지 않음 | 세 거리 분포 figure와 중앙값 차이 CI 추가 | 실험 3 | 추가 | 그림 1개, +0.2쪽 |
| Discussion | Defense 중복 문제 미해결 | Def=Fld+Pos이므로 현재 평균이 Fld를 중복함을 명시 | Defense 감사 | 추가 | +3문장 |
| Limitations | raw-stat/ID/provenance 문제가 약함 | raw-stat baseline·ablation 미완료, 23행 ID 제외, 2025 추출일 부재, survivorship 추가 | 전 실험 | 교체 | +5–7줄 |
| Appendix | 세부 bootstrap/매칭/PA 표 부족 | 전체 안정성, paired CI, sensitivity, exclusion list를 부록으로 이동 | 전 실험 | 추가 | +1–2쪽 |

메인 텍스트는 약 1쪽 증가, main table 최대 2개와 figure 1개로 제한할 수 있습니다. 상세 ablation 상태표는 appendix에 두는 편이 좋습니다.

## 12.7 연구의 예상 평가 변화

- **해결된 약점:** Δv association과 진짜 forecasting의 구분, 최종연도 holdout, 단순 baseline, 선수-ID 보수적 처리, PA별 안정성, scalar-matched diversity가 추가됐습니다.
- **남은 약점:** 가장 중요한 raw-stat baseline, 실제 shrinkage/leave-one-out, Defense 변형, JointVAE seed 안정성이 여전히 없습니다. 외부 construct validation도 6명 pilot 수준입니다.
- **완성도 변화:** 예측·안정성·프로필 다양성 주장은 이전보다 훨씬 검증 가능해졌지만, 입력 구성의 강건성은 검증되지 않았습니다. 따라서 “전반적 검증 완료”가 아니라 “핵심 주장 일부가 out-of-time 및 robustness evidence를 얻음” 정도로 평가해야 합니다.
- **가장 강해진 contribution:** 단일 scalar와 달리 유사 가치 선수의 프로필 차이를 정량적으로 보존하면서도, t-only 벡터가 제한적이지만 유의한 다음 시즌 예측 신호를 갖는다는 점입니다.
- **과장 금지:** 원시 통계보다 우월한 표현, scouting grade 대체, causal development effect, ROI, Defense construct validity는 주장하면 안 됩니다.

## 12.8 실행 및 재현 방법

### 환경 재생성

```bash
cd {REPO_ROOT}
python3.12 -m venv academic_validation_experiments/.venv
source academic_validation_experiments/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt pytest==9.0.3 requests
```

### 각 실험

```bash
python academic_validation_experiments/scripts/00_audit_and_ids.py
python academic_validation_experiments/scripts/01_prospective_prediction.py
python academic_validation_experiments/scripts/02_stability_ablation.py
python academic_validation_experiments/scripts/03_profile_diversity.py
```

### 표·그림·보고서 재생성

```bash
python academic_validation_experiments/scripts/04_generate_report.py
```

### 테스트

```bash
python -m pytest academic_validation_experiments/tests -q
```

### 전체 실행 및 로그 저장

```bash
bash academic_validation_experiments/run_stage_a.sh
```

### manuscript preview 빌드

```bash
cd academic_validation_experiments/manuscript_preview
latexmk -pdf -interaction=nonstopmode -halt-on-error validation_preview.tex
```

현재 실행 환경에는 `latexmk`와 `pdflatex`가 없어 preview PDF 컴파일은 생략했습니다. 소스와 표/그림 경로는 생성했으며, TeX 도구가 있는 환경에서 위 명령으로 빌드할 수 있습니다. 원문 통합 및 CVPR 스타일에서의 시각 검사는 Stage B 승인 후 수행해야 합니다.

## 승인 전 상태

권위 원고와 기존 paper figure/table/result 파일은 수정하지 않았습니다. 원시 FanGraphs 캐시가 제공되면 실패한 네 분석(raw-stat Ridge, literal shrinkage, leave-one-out, Defense variants)을 같은 격리 폴더에서 이어서 실행해야 합니다. 그렇지 않으면 현재 확인된 결과만 제한적으로 통합할 수 있습니다.
"""
    (EXP_ROOT / "report_ko.md").write_text(report)
    print(f"Wrote {EXP_ROOT / 'report_ko.md'}")


if __name__ == "__main__":
    main()
