import os
import json
import pandas as pd
from typing import Dict, Any, List

def generate_route_svg(
    route: str,
    route_type: str,
    route_df: pd.DataFrame,
    note_windows: List[Dict[str, Any]],
    genuine_note_ids: set = None,
    width: int = 950,
    height: int = 340
) -> str:
    """
    Renders a self-contained, publication-grade SVG chart for a single shipping route.
    Plots weekly cost per tonne-km, 8-week baseline, peer baseline,
    shaded note windows, and colored anomaly flags (green=justified, red=unexplained).
    Requires ZERO network connections or external libraries.
    """
    margin = {"top": 30, "right": 40, "bottom": 50, "left": 65}
    chart_w = width - margin["left"] - margin["right"]
    chart_h = height - margin["top"] - margin["bottom"]

    sorted_df = route_df.sort_values(by="week_of").reset_index(drop=True)
    n_points = len(sorted_df)
    if n_points == 0:
        return f"<svg width='{width}' height='{height}'><text x='20' y='40'>No data for {route}</text></svg>"

    # Compute bounds
    all_vals = []
    for col in ["cost_per_tonne_km", "own_baseline", "peer_baseline"]:
        if col in sorted_df.columns:
            valid_vals = sorted_df[col].dropna().tolist()
            all_vals.extend(valid_vals)

    min_val = max(0.0, min(all_vals) * 0.88 if all_vals else 0.0)
    max_val = max(all_vals) * 1.12 if all_vals else 5.0
    val_range = max_val - min_val if max_val > min_val else 1.0

    def get_x(idx: int) -> float:
        if n_points <= 1:
            return margin["left"] + chart_w / 2
        return margin["left"] + (idx / (n_points - 1)) * chart_w

    def get_y(val: float) -> float:
        norm = (val - min_val) / val_range
        return margin["top"] + chart_h - (norm * chart_h)

    # SVG Elements
    svg_parts = []
    svg_parts.append(f"<svg viewBox='0 0 {width} {height}' class='route-chart' xmlns='http://www.w3.org/2000/svg'>")

    # Background & Grid
    svg_parts.append(f"<rect x='{margin['left']}' y='{margin['top']}' width='{chart_w}' height='{chart_h}' fill='#ffffff' stroke='#e2e8f0' stroke-width='1'/>")

    # Shaded Note Windows
    for nw in note_windows:
        # nw: {'note_id': str, 'start': str, 'end': str, 'text': str, 'applies_to': str}
        # Find index range overlapping this note
        start_dt = pd.to_datetime(nw['start'])
        end_dt = pd.to_datetime(nw['end'])
        
        # Check if note applies to this route
        if nw['applies_to'] != route and nw['applies_to'] != "All Routes":
            continue

        matching_indices = [
            i for i, row in sorted_df.iterrows()
            if (pd.to_datetime(row['week_of']) <= end_dt) and (pd.to_datetime(row['week_of']) + pd.Timedelta(days=6) >= start_dt)
        ]

        if matching_indices:
            idx_start = max(0, min(matching_indices))
            idx_end = min(n_points - 1, max(matching_indices))
            x_start = get_x(idx_start) - (chart_w / n_points / 2)
            x_end = get_x(idx_end) + (chart_w / n_points / 2)
            band_w = max(x_end - x_start, 12.0)

            # Different color for genuine vs decoy note windows
            is_genuine = nw['note_id'] in (genuine_note_ids or set())
            fill_col = "rgba(59, 130, 246, 0.12)" if is_genuine else "rgba(148, 163, 184, 0.12)"
            stroke_col = "rgba(59, 130, 246, 0.35)" if is_genuine else "rgba(148, 163, 184, 0.35)"

            svg_parts.append(
                f"<rect x='{x_start:.1f}' y='{margin['top']}' width='{band_w:.1f}' height='{chart_h}' "
                f"fill='{fill_col}' stroke='{stroke_col}' stroke-width='1' stroke-dasharray='2,2'>"
                f"<title>{nw['note_id']}: {nw['text']} ({nw['start']} to {nw['end']})</title></rect>"
            )
            svg_parts.append(
                f"<text x='{x_start + 4:.1f}' y='{margin['top'] + 14}' font-size='10' font-weight='600' "
                f"fill='#475569'>{nw['note_id']}</text>"
            )

    # Y-axis Grid & Ticks (5 steps)
    for step in range(5):
        tick_val = min_val + (step / 4.0) * val_range
        y_pos = get_y(tick_val)
        svg_parts.append(f"<line x1='{margin['left']}' y1='{y_pos:.1f}' x2='{margin['left'] + chart_w}' y2='{y_pos:.1f}' stroke='#f1f5f9' stroke-width='1'/>")
        svg_parts.append(f"<text x='{margin['left'] - 8}' y='{y_pos + 4:.1f}' text-anchor='end' font-size='11' fill='#64748b'>₹{tick_val:.2f}</text>")

    # X-axis Ticks (quarterly intervals)
    for i in range(0, n_points, 13):
        x_pos = get_x(i)
        date_str = str(sorted_df.iloc[i]['week_of'])
        short_date = date_str[5:10] if len(date_str) >= 10 else date_str
        year_str = date_str[:4]
        svg_parts.append(f"<line x1='{x_pos:.1f}' y1='{margin['top'] + chart_h}' x2='{x_pos:.1f}' y2='{margin['top'] + chart_h + 5}' stroke='#cbd5e1' stroke-width='1'/>")
        svg_parts.append(f"<text x='{x_pos:.1f}' y='{margin['top'] + chart_h + 18}' text-anchor='middle' font-size='10' fill='#64748b'>{short_date}</text>")
        svg_parts.append(f"<text x='{x_pos:.1f}' y='{margin['top'] + chart_h + 30}' text-anchor='middle' font-size='9' fill='#94a3b8'>{year_str}</text>")

    # Path 1: Peer Baseline (dotted purple)
    peer_pts = []
    for i, row in sorted_df.iterrows():
        v = row.get('peer_baseline')
        if pd.notna(v):
            peer_pts.append(f"{get_x(i):.1f},{get_y(float(v)):.1f}")
    if peer_pts:
        svg_parts.append(f"<polyline points='{' '.join(peer_pts)}' fill='none' stroke='#8b5cf6' stroke-width='1.8' stroke-dasharray='3,3'/>")

    # Path 2: Own Trailing 8-Week Baseline (dashed slate)
    own_pts = []
    for i, row in sorted_df.iterrows():
        v = row.get('own_baseline')
        if pd.notna(v):
            own_pts.append(f"{get_x(i):.1f},{get_y(float(v)):.1f}")
    if own_pts:
        svg_parts.append(f"<polyline points='{' '.join(own_pts)}' fill='none' stroke='#64748b' stroke-width='2' stroke-dasharray='5,4'/>")

    # Path 3: Weekly Cost per tonne-km (solid blue)
    cost_pts = []
    for i, row in sorted_df.iterrows():
        v = row.get('cost_per_tonne_km')
        if pd.notna(v):
            cost_pts.append(f"{get_x(i):.1f},{get_y(float(v)):.1f}")
    if cost_pts:
        svg_parts.append(f"<polyline points='{' '.join(cost_pts)}' fill='none' stroke='#0284c7' stroke-width='2.2'/>")

    # Flagged Markers
    for i, row in sorted_df.iterrows():
        flag_val = str(row.get('flagged', 'No'))
        cptk = float(row.get('cost_per_tonne_km', 0.0))
        cx = get_x(i)
        cy = get_y(cptk)
        week_str = str(row.get('week_of'))

        if "justified" in flag_val.lower():
            # Green marker for Justified
            nid = str(row.get('matched_note_id', ''))
            svg_parts.append(
                f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='6' fill='#10b981' stroke='#ffffff' stroke-width='2'>"
                f"<title>Week: {week_str}\nCost: ₹{cptk:.2f}/tonne-km\nVerdict: Justified (Note {nid})\n{row.get('reason','')}</title>"
                f"</circle>"
            )
            svg_parts.append(
                f"<text x='{cx:.1f}' y='{cy - 9:.1f}' text-anchor='middle' font-size='9' font-weight='700' fill='#059669'>{nid}</text>"
            )
        elif flag_val == "Yes":
            # Red/Orange marker for Unjustified
            own_dev = row.get('vs_own_history', '')
            peer_dev = row.get('vs_similar_routes', '')
            svg_parts.append(
                f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='6' fill='#ef4444' stroke='#ffffff' stroke-width='2'>"
                f"<title>Week: {week_str}\nCost: ₹{cptk:.2f}/tonne-km\nVerdict: Unexplained Spike (Yes)\nvs Own: {own_dev}\nvs Peer: {peer_dev}\n{row.get('reason','')}</title>"
                f"</circle>"
            )
            svg_parts.append(
                f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='10' fill='none' stroke='#ef4444' stroke-width='1.5' stroke-dasharray='2,2'/>"
            )

    # Route Header Title inside SVG
    svg_parts.append(f"<text x='{margin['left']}' y='20' font-size='14' font-weight='700' fill='#0f172a'>{route} <tspan font-size='11' font-weight='500' fill='#64748b'>({route_type} Haul)</tspan></text>")

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def generate_html_report(
    all_weeks_df: pd.DataFrame,
    notes_df: pd.DataFrame,
    output_html_path: str = "report.html",
    fuel_window_weeks: int = 2
) -> str:
    """
    Builds a complete, standalone, self-contained HTML audit dashboard.
    Renders responsive SVG charts for all corridors, executive metrics,
    note summaries, and anomaly tables. Fully functional offline.
    """
    # 1. Parse note windows using production-grade temporal parser
    from src.temporal_parser import TemporalWindowParser
    tw_parser = TemporalWindowParser(fuel_window_weeks=fuel_window_weeks)
    note_windows = []
    for _, r in notes_df.iterrows():
        nid = r['note_id']
        note_txt = r['note']
        d_val = pd.to_datetime(r['date'])
        start_dt, end_dt, wtype = tw_parser.parse_window(d_val, note_txt)
        note_windows.append({
            "note_id": nid,
            "start": start_dt.strftime('%Y-%m-%d'),
            "end": end_dt.strftime('%Y-%m-%d'),
            "applies_to": r['applies_to'],
            "text": note_txt
        })

    # 2. Executive KPI Counts
    total_route_weeks = len(all_weeks_df)
    flagged_rows = all_weeks_df[all_weeks_df['flagged'].isin(['Yes', 'No (justified)'])].copy()
    total_flagged = len(flagged_rows)
    justified_count = len(flagged_rows[flagged_rows['flagged'] == 'No (justified)'])
    unjustified_count = len(flagged_rows[flagged_rows['flagged'] == 'Yes'])

    # Notes that actually justify a row (derived from matched_note_id, not a hardcoded list)
    genuine_note_ids = set(
        str(v) for v in flagged_rows[flagged_rows['flagged'] == 'No (justified)']['matched_note_id'].dropna().tolist()
        if str(v).strip() and str(v).lower() != 'nan'
    )

    # 3. Build Charts
    routes = sorted(all_weeks_df['route'].unique())
    charts_html = []
    
    for r in routes:
        sub_df = all_weeks_df[all_weeks_df['route'] == r].copy()
        rtype = sub_df.iloc[0].get('route_type', 'Corridor')
        svg_code = generate_route_svg(r, rtype, sub_df, note_windows, genuine_note_ids)
        
        r_flagged = len(sub_df[sub_df['flagged'] == 'Yes'])
        r_justified = len(sub_df[sub_df['flagged'] == 'No (justified)'])
        
        card = f"""
        <div class="chart-card">
            <div class="card-header">
                <div>
                    <h3 class="route-title">{r}</h3>
                    <span class="route-type-badge">{rtype} Haul</span>
                </div>
                <div class="route-badges">
                    <span class="badge badge-unjustified">{r_flagged} Unexplained</span>
                    <span class="badge badge-justified">{r_justified} Justified</span>
                </div>
            </div>
            <div class="chart-container">
                {svg_code}
            </div>
        </div>
        """
        charts_html.append(card)

    # 4. Table of Flagged Rows
    flagged_sorted = flagged_rows.sort_values(by=['week_of', 'route'])
    table_rows = []
    for _, row in flagged_sorted.iterrows():
        is_just = "justified" in str(row['flagged']).lower()
        badge_cls = "badge-justified" if is_just else "badge-unjustified"
        status_label = "Justified" if is_just else "Yes"
        nid = str(row.get('matched_note_id', ''))
        nid_str = f"<strong>{nid}</strong>" if nid and nid != 'nan' else "<span style='color:#94a3b8;'>—</span>"
        
        n_shipments = int(row.get('total_shipments', 0))
        if n_shipments > 0:
            if n_shipments <= 3:
                vol_badge = f"<span style='color:#b45309; font-weight:600;'>{n_shipments} <span style='font-size:11px; color:#92400e;'>(Thin)</span></span>"
            else:
                vol_badge = f"<span style='color:#0f766e; font-weight:600;'>{n_shipments} <span style='font-size:11px; color:#115e59;'>(High)</span></span>"
        else:
            vol_badge = "<span style='color:#94a3b8;'>—</span>"

        table_rows.append(f"""
        <tr>
            <td style="font-weight: 600;">{row['route']}</td>
            <td><code>{row['week_of']}</code></td>
            <td style="text-align: right; font-weight: 600;">₹{float(row['cost_per_tonne_km']):.2f}</td>
            <td style="text-align: right;">{row['vs_own_history']}</td>
            <td style="text-align: right;">{row['vs_similar_routes']}</td>
            <td><span class="badge {badge_cls}">{status_label}</span></td>
            <td style="text-align: center;">{nid_str}</td>
            <td style="text-align: center;">{vol_badge}</td>
            <td style="font-size: 13px; color: #334155;">{row['reason']}</td>
        </tr>
        """)

    # 5. Assemble HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Freight Tiger — Shipping Cost Anomaly Detection Report</title>
    <style>
        :root {{
            --bg: #f8fafc;
            --surface: #ffffff;
            --border: #e2e8f0;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --primary: #0284c7;
            --success: #10b981;
            --danger: #ef4444;
            --purple: #8b5cf6;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text-main);
            line-height: 1.5;
            padding: 24px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: white;
            padding: 32px;
            border-radius: 12px;
            margin-bottom: 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }}
        .header h1 {{ font-size: 26px; font-weight: 800; margin-bottom: 8px; letter-spacing: -0.5px; }}
        .header p {{ color: #94a3b8; font-size: 14px; max-width: 800px; }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 28px;
        }}
        .kpi-card {{
            background: var(--surface);
            padding: 20px;
            border-radius: 10px;
            border: 1px solid var(--border);
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .kpi-label {{ font-size: 12px; font-weight: 600; text-transform: uppercase; color: var(--text-muted); letter-spacing: 0.5px; }}
        .kpi-val {{ font-size: 28px; font-weight: 800; margin-top: 4px; color: var(--text-main); }}
        .kpi-sub {{ font-size: 12px; color: var(--text-muted); margin-top: 2px; }}
        
        .legend-box {{
            background: var(--surface);
            padding: 16px 20px;
            border-radius: 8px;
            border: 1px solid var(--border);
            margin-bottom: 24px;
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 20px;
            font-size: 13px;
        }}
        .legend-item {{ display: flex; align-items: center; gap: 8px; }}
        .legend-line {{ width: 24px; height: 3px; border-radius: 2px; }}
        .legend-circle {{ width: 12px; height: 12px; border-radius: 50%; }}
        .legend-rect {{ width: 16px; height: 12px; border-radius: 2px; border: 1px dashed rgba(59, 130, 246, 0.5); background: rgba(59, 130, 246, 0.15); }}

        .chart-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }}
        .route-title {{ font-size: 18px; font-weight: 700; color: var(--text-main); display: inline-block; }}
        .route-type-badge {{
            font-size: 11px;
            font-weight: 600;
            background: #f1f5f9;
            color: #475569;
            padding: 3px 8px;
            border-radius: 4px;
            margin-left: 8px;
        }}
        .chart-container {{ width: 100%; overflow-x: auto; }}
        .route-chart {{ width: 100%; height: auto; display: block; }}

        .badge {{
            display: inline-block;
            font-size: 11px;
            font-weight: 700;
            padding: 3px 8px;
            border-radius: 9999px;
            letter-spacing: 0.3px;
        }}
        .badge-justified {{ background: #dcfce7; color: #166534; }}
        .badge-unjustified {{ background: #fee2e2; color: #991b1b; }}

        .table-section {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px;
            margin-top: 32px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .table-section h2 {{ font-size: 20px; font-weight: 800; margin-bottom: 16px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            text-align: left;
        }}
        th {{
            background: #f8fafc;
            padding: 12px 14px;
            font-weight: 700;
            color: #475569;
            border-bottom: 2px solid var(--border);
        }}
        td {{
            padding: 12px 14px;
            border-bottom: 1px solid var(--border);
            vertical-align: middle;
        }}
        tr:hover td {{ background: #f8fafc; }}
        code {{
            background: #f1f5f9;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 12px;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            color: var(--text-muted);
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>Freight Tiger — Cost Anomaly & Disruption Audit Report</h1>
            <p>Automated corridor-level anomaly detection, rolling baseline modeling, and deterministic context-note justification audit for freight shipping operations.</p>
        </header>

        <section class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Total Route-Weeks</div>
                <div class="kpi-val">{total_route_weeks}</div>
                <div class="kpi-sub">7 corridors across 2024–2025</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Anomalies Flagged</div>
                <div class="kpi-val">{total_flagged}</div>
                <div class="kpi-sub">{total_flagged / total_route_weeks * 100:.1f}% flag rate</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Justified by Notes</div>
                <div class="kpi-val" style="color: #10b981;">{justified_count}</div>
                <div class="kpi-sub">Verified against context notes</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Unexplained Spikes</div>
                <div class="kpi-val" style="color: #ef4444;">{unjustified_count}</div>
                <div class="kpi-sub">Actionable for operational review</div>
            </div>
        </section>

        <div class="legend-box">
            <strong style="color: #334155;">Legend:</strong>
            <div class="legend-item">
                <div class="legend-line" style="background: #0284c7;"></div>
                <span>Weekly Unit Cost (₹/tonne-km)</span>
            </div>
            <div class="legend-item">
                <div class="legend-line" style="background: #64748b; border-top: 2px dashed #64748b; height: 0;"></div>
                <span>Trailing 8-Week Own Baseline</span>
            </div>
            <div class="legend-item">
                <div class="legend-line" style="background: #8b5cf6; border-top: 2px dotted #8b5cf6; height: 0;"></div>
                <span>Peer Baseline (Same Haul Type)</span>
            </div>
            <div class="legend-item">
                <div class="legend-circle" style="background: #10b981;"></div>
                <span>Justified Spike (Cited Note)</span>
            </div>
            <div class="legend-item">
                <div class="legend-circle" style="background: #ef4444;"></div>
                <span>Unexplained Anomaly (Flagged Yes)</span>
            </div>
            <div class="legend-item">
                <div class="legend-rect"></div>
                <span>Note Disruption Window (Shaded)</span>
            </div>
        </div>

        <section class="charts-section">
            {"".join(charts_html)}
        </section>

        <section class="table-section">
            <h2>Audited Anomalies Register ({total_flagged} Weeks)</h2>
            <div style="overflow-x: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>Route</th>
                            <th>Week Of</th>
                            <th style="text-align: right;">Unit Cost</th>
                            <th style="text-align: right;">vs Own Baseline</th>
                            <th style="text-align: right;">vs Peer Baseline</th>
                            <th>Verdict</th>
                            <th style="text-align: center;">Cited Note</th>
                            <th style="text-align: center;">Shipments</th>
                            <th>Explanation / Reason</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(table_rows)}
                    </tbody>
                </table>
            </div>
        </section>

        <footer class="footer">
            <p>Generated by Freight Tiger Shipping Cost Anomaly Detection Pipeline &bull; 100% Deterministic &bull; Self-Contained Offline Report</p>
        </footer>
    </div>
</body>
</html>
"""
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_html_path
