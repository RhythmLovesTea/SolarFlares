from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .ladder import STATE_COLORS, InstabilityLadderState


def build_ladder_gauge(state, fai: float, nri_sigma: float) -> go.Figure:
    import math

    color = STATE_COLORS[state]

    # ── Segment definitions (5 states mapped to 0-4) ──────────────────────────
    segments = [
        (0, 1, "#4caf7d", "Quiet"),
        (1, 2, "#e8c96a", "Pre-Heat"),
        (2, 3, "#e89c5a", "Therm. Instab."),
        (3, 4, "#e05a5a", "Ignition"),
        (4, 5, "#c64545", "Critical"),
    ]
    # Map state value (0-4) to arc; full arc spans 210° from 210° to -30° (clockwise)
    total_range = 5
    start_deg = 210   # leftmost (Quiet start)
    sweep_deg = 240   # total sweep

    def val_to_deg(v):
        return start_deg - (v / total_range) * sweep_deg

    def arc_path(v_start, v_end, r_inner=0.55, r_outer=0.85):
        """SVG-like path for a donut arc segment in polar-ish coords."""
        a0 = math.radians(val_to_deg(v_start))
        a1 = math.radians(val_to_deg(v_end))
        # Outer arc points
        x0o, y0o = r_outer * math.cos(a0), r_outer * math.sin(a0)
        x1o, y1o = r_outer * math.cos(a1), r_outer * math.sin(a1)
        # Inner arc points
        x0i, y0i = r_inner * math.cos(a0), r_inner * math.sin(a0)
        x1i, y1i = r_inner * math.cos(a1), r_inner * math.sin(a1)
        large = 1 if abs(v_end - v_start) / total_range * sweep_deg > 180 else 0
        path = (
            f"M {x0o:.4f},{y0o:.4f} "
            f"A {r_outer},{r_outer} 0 {large},0 {x1o:.4f},{y1o:.4f} "
            f"L {x1i:.4f},{y1i:.4f} "
            f"A {r_inner},{r_inner} 0 {large},1 {x0i:.4f},{y0i:.4f} Z"
        )
        return path

    fig = go.Figure()

    # ── Draw arc segments ─────────────────────────────────────────────────────
    for v0, v1, seg_color, label in segments:
        fig.add_shape(
            type="path",
            path=arc_path(v0, v1),
            fillcolor=seg_color,
            opacity=0.18,
            line=dict(width=0),
            xref="paper", yref="paper",
            x0=0, y0=0, x1=1, y1=1,
        )

    # ── Draw active segment (brighter fill) ───────────────────────────────────
    v0, v1 = state.value, state.value + 1
    fig.add_shape(
        type="path",
        path=arc_path(v0, v1),
        fillcolor=color,
        opacity=0.72,
        line=dict(width=0),
        xref="paper", yref="paper",
        x0=0, y0=0, x1=1, y1=1,
    )

    # ── Needle ────────────────────────────────────────────────────────────────
    needle_val = state.value + 0.5
    needle_angle = math.radians(val_to_deg(needle_val))
    nx = 0.70 * math.cos(needle_angle)
    ny = 0.70 * math.sin(needle_angle)
    # Convert from [-1,1] space to [0,1] paper coords (centre at 0.5, 0.45)
    cx, cy = 0.5, 0.45
    scale = 0.38
    fig.add_shape(
        type="line",
        x0=cx, y0=cy,
        x1=cx + nx * scale, y1=cy + ny * scale * 0.9,
        line=dict(color=color, width=3),
        xref="paper", yref="paper",
    )
    fig.add_shape(
        type="circle",
        x0=cx - 0.018, y0=cy - 0.018 * 0.9,
        x1=cx + 0.018, y1=cy + 0.018 * 0.9,
        fillcolor=color, line=dict(width=0),
        xref="paper", yref="paper",
    )

    # ── Segment boundary ticks + labels ───────────────────────────────────────
    label_r = 0.96
    tick_labels = ["Quiet", "Pre-H.", "Therm.", "Ign.", "Crit.", ""]
    for i, lbl in enumerate(tick_labels):
        angle = math.radians(val_to_deg(i))
        lx = cx + math.cos(angle) * label_r * scale
        ly = cy + math.sin(angle) * label_r * scale * 0.9
        # small tick mark
        t0x = cx + math.cos(angle) * 0.86 * scale
        t0y = cy + math.sin(angle) * 0.86 * scale * 0.9
        t1x = cx + math.cos(angle) * 0.93 * scale
        t1y = cy + math.sin(angle) * 0.93 * scale * 0.9
        fig.add_shape(
            type="line", x0=t0x, y0=t0y, x1=t1x, y1=t1y,
            line=dict(color="#b0a898", width=1.5),
            xref="paper", yref="paper",
        )
        if lbl:
            fig.add_annotation(
                x=lx, y=ly, text=lbl, showarrow=False,
                font=dict(size=9.5, color="#59554e", family="Inter, sans-serif"),
                xref="paper", yref="paper",
            )

    # ── Centre annotations ────────────────────────────────────────────────────
    state_labels = {
        "QUIET": "Quiet", "PREHEATING": "Pre-Heating",
        "THERMAL_INSTABILITY": "Thermal Instability",
        "IGNITION": "Ignition", "CRITICAL": "Critical",
    }
    fig.add_annotation(
        x=0.5, y=0.55,
        text="<b>Instability Ladder</b>",
        showarrow=False,
        font=dict(size=13, color="#6c6a64", family="'Cormorant Garamond', serif"),
        xref="paper", yref="paper",
    )
    fig.add_annotation(
        x=0.5, y=0.32,
        text=f"<b style='color:{color}'>{state_labels.get(state.name, state.name)}</b>",
        showarrow=False,
        font=dict(size=15, color=color, family="'Cormorant Garamond', serif"),
        xref="paper", yref="paper",
    )
    fig.add_annotation(
        x=0.5, y=0.16,
        text=f"FAI {fai:.2f}  |  NRI {nri_sigma:.1f}σ",
        showarrow=False,
        font=dict(size=11, color="#6c6a64", family="Inter, sans-serif"),
        xref="paper", yref="paper",
    )

    fig.update_layout(
        height=310,
        margin=dict(l=10, r=10, t=18, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        showlegend=False,
    )
    return fig




def build_lightcurve(
    df: pd.DataFrame,
    events: pd.DataFrame,
    fai_col: str = "fai",
    nri_col: str = "nri_sigma",
    sxr_col: str = "sxr_b",
    hxr_col: str = "hxr_proxy",
    source_label: str = "GOES/Fermi proxy",
) -> go.Figure:
    real_counts = source_label == "real_aditya_l1"
    soft_title = "SoLEXS (real): Level-1 counts/sec" if real_counts else "SoLEXS proxy: GOES XRS-B"
    hard_title = "HEL1OS (real): CZT full-band counts/sec" if real_counts else "HEL1OS proxy: Fermi GBM / XRS-A derivative"
    soft_name = "SoLEXS SDD2" if real_counts else "GOES XRS-B"
    hard_name = "HEL1OS CZT full band" if real_counts else "HXR proxy"
    if len(df) > 50_000:
        step = max(1, len(df) // 50_000)
        df = pd.concat([df.iloc[::step], df.tail(1)]).sort_index()
        df = df[~df.index.duplicated(keep="first")]
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
        subplot_titles=(soft_title, hard_title),
    )
    fig.add_trace(go.Scatter(x=df.index, y=df[sxr_col], name=soft_name, line=dict(color="#3d7abf", width=1.5)), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=df.index, y=df[fai_col], name="FAI", line=dict(color="#cc785c", dash="dash", width=1.5)),
        row=1,
        col=1,
        secondary_y=True,
    )
    if not real_counts:
        for threshold, label, color in [(1e-6, "C", "#a9583e"), (1e-5, "M", "#cc785c"), (1e-4, "X", "#c64545")]:
            fig.add_hline(
                y=threshold,
                line_dash="dot",
                line_color=color,
                annotation_text=label,
                annotation_position="left",
                row=1,
                col=1,
            )

    fig.add_trace(go.Scatter(x=df.index, y=df[hxr_col], name=hard_name, line=dict(color="#5db8a6", width=1.5)), row=2, col=1)
    fig.add_trace(
        go.Scatter(x=df.index, y=df[nri_col], name="NRI sigma", line=dict(color="#e8a55a", dash="dash", width=1.5)),
        row=2,
        col=1,
        secondary_y=True,
    )
    for sigma in [1, 2, 3]:
        fig.add_hline(
            y=sigma,
            line_dash="dash",
            line_color="#e8a55a",
            annotation_text=f"{sigma}σ",
            annotation_position="right",
            row=2,
            col=1,
            secondary_y=True,
        )

    if "ladder_state" in df:
        previous_state = None
        start = None
        state_intervals = []
        for timestamp, state_name in df["ladder_state"].items():
            if state_name != previous_state:
                if previous_state is not None:
                    state_intervals.append((start, timestamp, previous_state))
                start = timestamp
                previous_state = state_name
        if previous_state is not None:
            state_intervals.append((start, df.index[-1], previous_state))
        max_state_bands = 30
        step = max(1, len(state_intervals) // max_state_bands)
        for x0, x1, state_name in state_intervals[::step][:max_state_bands]:
            color = STATE_COLORS[InstabilityLadderState[state_name]]
            fig.add_vrect(x0=x0, x1=x1, fillcolor=color, opacity=0.06, line_width=0)

    if not events.empty:
        for peak_time in pd.to_datetime(events["peak_time"]):
            fig.add_vline(x=peak_time, line_color="#c64545", line_width=1.5)

    fig.update_yaxes(type="linear" if real_counts else "log", title_text="Counts/sec" if real_counts else "Flux W/m²", row=1, col=1)
    fig.update_yaxes(title_text="FAI", range=[0, 1], tickvals=[0, 0.5, 1], row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Counts/sec" if real_counts else "Normalized HXR", row=2, col=1)
    fig.update_yaxes(title_text="NRI σ", row=2, col=1, secondary_y=True)
    fig.update_layout(
        height=680,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.13, x=0, font=dict(size=12, family="Inter, sans-serif", color="#3d3d3a")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#faf9f5",
        font_color="#141413",
        font_family="Inter, sans-serif",
        margin=dict(l=72, r=94, t=96, b=58),
        xaxis=dict(gridcolor="#e6dfd8", linecolor="#e6dfd8", zerolinecolor="#e6dfd8"),
        yaxis=dict(gridcolor="#e6dfd8", linecolor="#e6dfd8"),
        xaxis2=dict(gridcolor="#e6dfd8", linecolor="#e6dfd8"),
        yaxis3=dict(gridcolor="#e6dfd8", linecolor="#e6dfd8"),
    )
    fig.update_annotations(font_size=15, font_family="'Cormorant Garamond', serif", font_color="#141413")
    return fig

