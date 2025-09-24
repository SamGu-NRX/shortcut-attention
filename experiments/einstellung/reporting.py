"""HTML/Markdown reporting helpers for Einstellung experiments."""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Dict

import pandas as pd

_TEMPLATE_HEADER = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>{title}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; }}
h1, h2 {{ color: #222; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background: #f5f5f5; }}
.bad {{ color: #c0392b; }}
.good {{ color: #27ae60; }}
.small {{ color: #555; font-size: 0.9em; }}
</style>
</head>
<body>
"""


def write_single_run_report(
    *,
    result: Dict[str, Any],
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Render a compact HTML summary for a single run."""
    output_dir.mkdir(parents=True, exist_ok=True)

    title = f"Einstellung Run – {result['strategy']} / {result['backbone']} (seed={result['seed']})"
    final_top1 = result.get("final_top1")
    final_top5 = result.get("final_top5")
    top1_delta = result.get("top1_delta")
    top5_delta = result.get("top5_delta")

    summary = summary_df.copy()
    summary["top1_pct"] = (summary["top1"] * 100).round(2)
    summary["top5_pct"] = (summary["top5"] * 100).round(2)

    html_parts = [
        _TEMPLATE_HEADER.format(title=title),
        f"<h1>{title}</h1>",
        f"<p class='small'>Generated: {_dt.datetime.now().isoformat(timespec='seconds')}</p>",
        "<h2>Final Metrics</h2>",
        "<table>",
        "<tr><th>Metric</th><th>Value</th></tr>",
        f"<tr><td>Top-1</td><td>{final_top1:.4f} ({(final_top1 or 0)*100:.2f}%)</td></tr>" if final_top1 is not None else "",
        f"<tr><td>Top-5</td><td>{final_top5:.4f} ({(final_top5 or 0)*100:.2f}%)</td></tr>" if final_top5 is not None else "",
        f"<tr><td>Δ Top-1 vs {result['reference_top1']:.2f}</td><td>{top1_delta:.4f}</td></tr>" if top1_delta is not None else "",
        f"<tr><td>Δ Top-5 vs {result['reference_top5']:.2f}</td><td>{top5_delta:.4f}</td></tr>" if top5_delta is not None else "",
        f"<tr><td>Performance Deficit</td><td>{result.get('performance_deficit')}</td></tr>",
        f"<tr><td>Shortcut Reliance</td><td>{result.get('shortcut_feature_reliance')}</td></tr>",
        f"<tr><td>Adaptation Delay</td><td>{result.get('adaptation_delay')}</td></tr>",
        "</table>",
        "<h2>Subset Summary</h2>",
        summary.to_html(index=False, float_format=lambda x: f"{x:.4f}"),
        "</body></html>",
    ]

    html = "\n".join(part for part in html_parts if part)
    output_path = output_dir / "report.html"
    output_path.write_text(html)
    return output_path


def write_comparative_report(
    *,
    comparative_table: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Generate an aggregated comparative HTML report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    title = "Comparative Einstellung Summary"

    html_parts = [
        _TEMPLATE_HEADER.format(title=title),
        f"<h1>{title}</h1>",
        f"<p class='small'>Generated: {_dt.datetime.now().isoformat(timespec='seconds')}</p>",
        "<h2>Topline Metrics (T2_shortcut_normal)</h2>",
        comparative_table.to_html(index=False, float_format=lambda x: f"{x:.4f}"),
        "<h2>Full Summary</h2>",
        summary_df.to_html(index=False, float_format=lambda x: f"{x:.4f}"),
        "</body></html>",
    ]

    html = "\n".join(html_parts)
    output_path = output_dir / "comparative_report.html"
    output_path.write_text(html)
    return output_path
