from __future__ import annotations

from html import escape
from pathlib import Path

from .training import multi_seed_validation, training_evidence


def model_report_html() -> str:
    primary = training_evidence(42)
    seeds = multi_seed_validation()
    validation = primary["validation"]
    final_test = primary["final_test"]
    coefficients = "".join(
        "<tr>"
        f"<td>{escape(str(row['feature']))}</td>"
        f"<td>{row['standardised_coefficient']:.4f}</td>"
        "</tr>"
        for row in primary["top_coefficients"]
    )
    seed_rows = "".join(
        "<tr>"
        f"<td>{row['seed']}</td>"
        f"<td>{row['validation']['improvement_pct']:.1f}%</td>"
        f"<td>{row['final_test']['improvement_pct']:.1f}%</td>"
        f"<td>{escape(str(row['accepted']))}</td>"
        "</tr>"
        for row in seeds
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Predictive Business Analytics</title></head>
<body>
<h1>Predictive Business Analytics</h1>
<p><strong>Validation improvement:</strong> {validation['improvement_pct']:.1f}%</p>
<p><strong>Final-test improvement:</strong> {final_test['improvement_pct']:.1f}%</p>
<p><strong>Accepted:</strong> {escape(str(primary['accepted']))}</p>
<h2>Model coefficients</h2>
<table><thead><tr><th>Feature</th><th>Standardised coefficient</th></tr></thead><tbody>{coefficients}</tbody></table>
<h2>Multi-seed robustness</h2>
<table><thead><tr>
<th>Seed</th><th>Validation improvement</th>
<th>Final-test improvement</th><th>Accepted</th>
</tr></thead><tbody>{seed_rows}</tbody></table>
<p><small>Synthetic forecasting benchmark. Results are not production accuracy claims.</small></p>
</body></html>"""


def write_model_report(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(model_report_html(), encoding="utf-8")
    return output
