"""Deterministic, script-free HTML views of strict comparison artifacts."""

from __future__ import annotations

import html
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from about_llm.evaluation.comparison_artifact import EvaluationComparisonArtifact

EVALUATION_COMPARISON_HTML_SCOPE = (
    "此页面只渲染已经严格加载的 comparison artifact; 未重新打开 "
    "cases、answers、results 或 run manifests, 未重新评分或重跑统计, 也未认证来源。"
)


def render_evaluation_comparison_html(
    artifact: EvaluationComparisonArtifact,
) -> str:
    """Render one self-contained, deterministic and HTML-escaped report."""

    content = artifact.content
    quality = _mapping(content["quality"], "quality")
    bootstrap = _mapping(content["bootstrap"], "bootstrap")
    gate = _mapping(content["gate_configuration"], "gate_configuration")
    bindings = _mapping(content["run_bindings"], "run_bindings")
    baseline_binding = _mapping(bindings["baseline"], "run_bindings.baseline")
    candidate_binding = _mapping(bindings["candidate"], "run_bindings.candidate")
    reasons = _sequence(content["reasons"], "reasons")
    protected = _mapping(content["protected_slices"], "protected_slices")
    decision_class = "pass" if artifact.passed else "fail"
    decision_text = "通过" if artifact.passed else "阻断"
    reason_items = (
        "\n".join(f"<li>{_h(reason)}</li>" for reason in reasons)
        if reasons
        else "<li>无; 所有已记录门禁条件均通过。</li>"
    )
    protected_rows = "\n".join(
        _result_row(slice_name, _mapping(result, f"protected_slices.{slice_name}"))
        for slice_name, result in protected.items()
    )
    if not protected_rows:
        protected_rows = (
            '<tr><td colspan="8">未配置 protected slice。</td></tr>'
        )
    metric_revisions = _mapping(
        bindings["metric_revisions"], "run_bindings.metric_revisions"
    )
    revision_rows = "\n".join(
        f"<tr><td><code>{_h(name)}</code></td><td><code>{_h(revision)}</code></td></tr>"
        for name, revision in metric_revisions.items()
    )
    cluster_details = _cluster_details(bootstrap, quality)
    report = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Content-Security-Policy"
        content="default-src 'none'; style-src 'unsafe-inline'; img-src data:;
                 base-uri 'none'; form-action 'none'">
  <meta name="about-llm-verification-scope" content="artifact_only_render">
  <title>LLM 评测发布门禁报告</title>
  <style>
    :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
    body {{ max-width: 1080px; margin: 0 auto; padding: 2rem; line-height: 1.55; }}
    header, section {{ margin-block: 1.5rem; }}
    .badge {{ display: inline-block; padding: .25rem .7rem; border: 2px solid currentColor;
              border-radius: 999px; font-weight: 700; }}
    .pass {{ color: #087830; }} .fail {{ color: #b42318; }}
    table {{ width: 100%; border-collapse: collapse; margin-block: .75rem; }}
    caption {{ text-align: left; font-weight: 700; margin-bottom: .4rem; }}
    th, td {{ border: 1px solid #8888; padding: .5rem; text-align: left; vertical-align: top; }}
    th {{ background: #8882; }}
    code {{ overflow-wrap: anywhere; }}
    .scope {{ border-left: .35rem solid #a15c00; padding: .8rem 1rem; background: #a15c0014; }}
    .muted {{ opacity: .78; }}
  </style>
</head>
<body data-comparison-fingerprint="{_h(artifact.comparison_fingerprint)}">
  <header>
    <h1>LLM 评测发布门禁报告</h1>
    <p><span class="badge {decision_class}">判定: {decision_text}</span></p>
    <p>Comparison identity: <code>{_h(artifact.comparison_fingerprint)}</code></p>
    <p class="muted">Schema: <code>{_h(content["comparison_version"])}</code>;
       case 数: {_number(content["case_count"])}</p>
  </header>

  <section aria-labelledby="systems">
    <h2 id="systems">比较对象</h2>
    <table>
      <thead><tr><th>角色</th><th>system_id</th><th>run manifest identity</th></tr></thead>
      <tbody>
        <tr><td>Baseline</td><td><code>{_h(baseline_binding["system_id"])}</code></td>
            <td><code>{_h(baseline_binding["manifest_fingerprint"])}</code></td></tr>
        <tr><td>Candidate</td><td><code>{_h(candidate_binding["system_id"])}</code></td>
            <td><code>{_h(candidate_binding["manifest_fingerprint"])}</code></td></tr>
      </tbody>
    </table>
    <p>Cases identity: <code>{_h(bindings["cases_fingerprint"])}</code></p>
  </section>

  <section aria-labelledby="overall">
    <h2 id="overall">总体结果</h2>
    <table>
      <thead><tr><th>质量指标</th><th>Baseline</th><th>Candidate</th>
        <th>均值差</th><th>区间下界</th><th>区间上界</th><th>改善重采样比例</th></tr></thead>
      <tbody>{_result_row(content["quality_metric"], quality, overall=True)}</tbody>
    </table>
    <ul>
      <li>Safety metric: <code>{_h(content["safety_metric"])}</code>;
          difference: {_number(content["safety_difference"])}</li>
      <li>Baseline mean latency: {_number(content["baseline_mean_latency_seconds"])} s</li>
      <li>Candidate mean latency: {_number(content["candidate_mean_latency_seconds"])} s</li>
    </ul>
    {cluster_details}
  </section>

  <section aria-labelledby="slices">
    <h2 id="slices">Protected slices</h2>
    <table>
      <thead><tr><th>Slice</th><th>Baseline</th><th>Candidate</th><th>均值差</th>
        <th>区间下界</th><th>区间上界</th><th>改善比例</th><th>重采样信息</th></tr></thead>
      <tbody>{protected_rows}</tbody>
    </table>
  </section>

  <section aria-labelledby="gate">
    <h2 id="gate">门禁配置与原因</h2>
    <table>
      <tbody>
        <tr><th>minimum_quality_difference</th><td>{_number(gate["minimum_quality_difference"])}</td></tr>
        <tr><th>maximum_safety_regression</th><td>{_number(gate["maximum_safety_regression"])}</td></tr>
        <tr><th>maximum_latency_increase_fraction</th><td>{_number(gate["maximum_latency_increase_fraction"])}</td></tr>
        <tr><th>maximum_slice_regression</th><td>{_number(gate["maximum_slice_regression"])}</td></tr>
        <tr><th>protected_slices</th><td><code>{_json_text(gate["protected_slices"])}</code></td></tr>
      </tbody>
    </table>
    <h3>失败原因</h3>
    <ul>{reason_items}</ul>
  </section>

  <section aria-labelledby="bootstrap">
    <h2 id="bootstrap">统计配置</h2>
    <table>
      <tbody>
        <tr><th>unit</th><td><code>{_h(bootstrap["unit"])}</code></td></tr>
        <tr><th>confidence</th><td>{_number(bootstrap["confidence"])}</td></tr>
        <tr><th>requested samples</th><td>{_number(bootstrap["samples"])}</td></tr>
        <tr><th>seed</th><td>{_number(bootstrap["seed"])}</td></tr>
        <tr><th>cluster metadata key</th>
            <td><code>{_h(bootstrap["cluster_metadata_key"])}</code></td></tr>
        <tr><th>cluster weighting</th>
            <td><code>{_h(bootstrap["cluster_weighting"])}</code></td></tr>
        <tr><th>exact max clusters</th>
            <td>{_optional_number(bootstrap["exact_max_clusters"])}</td></tr>
      </tbody>
    </table>
    <table>
      <caption>Metric revisions</caption>
      <thead><tr><th>Metric</th><th>Revision</th></tr></thead>
      <tbody>{revision_rows}</tbody>
    </table>
  </section>

  <section class="scope" aria-labelledby="scope">
    <h2 id="scope">证据范围</h2>
    <p><strong>Render scope:</strong> {_h(EVALUATION_COMPARISON_HTML_SCOPE)}</p>
    <p><strong>Artifact boundary:</strong> {_h(content["evidence_boundary"])}</p>
  </section>
</body>
</html>
"""
    return report


def write_evaluation_comparison_html(
    path: Path, artifact: EvaluationComparisonArtifact
) -> None:
    """Write the derived report as UTF-8 with a stable trailing newline."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_evaluation_comparison_html(artifact),
        encoding="utf-8",
        newline="\n",
    )


def _result_row(
    label: Any, result: Mapping[str, Any], *, overall: bool = False
) -> str:
    baseline_key = (
        "baseline_estimand" if "baseline_estimand" in result else "baseline_mean"
    )
    candidate_key = (
        "candidate_estimand" if "candidate_estimand" in result else "candidate_mean"
    )
    details = _result_details(result)
    cells = [
        f"<td><code>{_h(label)}</code></td>",
        f"<td>{_number(result[baseline_key])}</td>",
        f"<td>{_number(result[candidate_key])}</td>",
        f"<td>{_number(result['mean_difference'])}</td>",
        f"<td>{_number(result['confidence_low'])}</td>",
        f"<td>{_number(result['confidence_high'])}</td>",
        f"<td>{_number(result['probability_of_improvement'])}</td>",
    ]
    if not overall:
        cells.append(f"<td>{details}</td>")
    return "<tr>" + "".join(cells) + "</tr>"


def _result_details(result: Mapping[str, Any]) -> str:
    if "cluster_count" not in result:
        return "case bootstrap"
    return (
        f"clusters={_number(result['cluster_count'])}; "
        f"sizes=<code>{_json_text(result['cluster_sizes'])}</code>; "
        f"weighting=<code>{_h(result['cluster_weighting'])}</code>; "
        f"method=<code>{_h(result['method'])}</code>; "
        f"resamples={_number(result['resamples_evaluated'])}"
    )


def _cluster_details(
    bootstrap: Mapping[str, Any], quality: Mapping[str, Any]
) -> str:
    if bootstrap["unit"] != "cluster":
        return "<p>Resampling unit: case。</p>"
    return (
        "<p>Cluster result: "
        + _result_details(quality)
        + f"; quantile=<code>{_h(quality['quantile_method'])}</code>; "
        + f"effective seed={_optional_number(quality['seed'])}。</p>"
    )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return cast(Mapping[str, Any], value)


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return cast(Sequence[Any], value)


def _number(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("report numeric value must be an int or float")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("report numeric value must be finite")
    return html.escape(json.dumps(value, allow_nan=False), quote=True)


def _optional_number(value: Any) -> str:
    return "<code>null</code>" if value is None else _number(value)


def _json_text(value: Any) -> str:
    thawed = _thaw(value)
    return _h(json.dumps(thawed, ensure_ascii=False, allow_nan=False, sort_keys=True))


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _h(value: Any) -> str:
    if value is None:
        text = "null"
    elif isinstance(value, str):
        text = value
    else:
        text = str(value)
    return html.escape(text, quote=True)
