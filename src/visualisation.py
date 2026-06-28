from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .ladder import STATE_COLORS, InstabilityLadderState


def build_ladder_gauge(state, fai: float, nri_sigma: float) -> go.Figure:
    import math

    color = STATE_COLORS[state]

    # Semicircle: 180° (left/Quiet) → 0° (right/Critical), 5 equal segments of 36°
    r_in, r_out = 0.52, 0.88
    seg_colors = ["#4caf7d", "#e3c85a", "#e89c5a", "#e05a5a", "#c64545"]
    seg_names  = ["Quiet", "Pre-Heat", "Thermal\nInstab.", "Ignition", "Critical"]

    def arc_polygon(a_start_deg: float, a_end_deg: float, n: int = 80):
        """Return (xs, ys) for a filled donut arc sector."""
        fwd = [math.radians(a_start_deg + (a_end_deg - a_start_deg) * i / n)
               for i in range(n + 1)]
        rev = list(reversed(fwd))
        xs = [r_out * math.cos(a) for a in fwd] + [r_in * math.cos(a) for a in rev]
        ys = [r_out * math.sin(a) for a in fwd] + [r_in * math.sin(a) for a in rev]
        xs.append(xs[0]); ys.append(ys[0])
        return xs, ys

    fig = go.Figure()

    # ── Draw the 5 arc segments ───────────────────────────────────────────────
    for i, (seg_col, name) in enumerate(zip(seg_colors, seg_names)):
        a0 = 180 - i * 36        # each segment = 36°
        a1 = 180 - (i + 1) * 36
        is_active = (i == state.value)
        xs, ys = arc_polygon(a1, a0)
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            fill="toself",
            fillcolor=seg_col,
            opacity=0.90 if is_active else 0.15,
            line=dict(color="#faf9f5", width=2),
            mode="lines",
            showlegend=False,
            hoverinfo="skip",
        ))

    # ── Needle ────────────────────────────────────────────────────────────────
    needle_deg = 180 - (state.value + 0.5) * 36
    needle_rad = math.radians(needle_deg)
    nx, ny = 0.72 * math.cos(needle_rad), 0.72 * math.sin(needle_rad)
    fig.add_trace(go.Scatter(
        x=[0, nx], y=[0, ny],
        mode="lines",
        line=dict(color=color, width=3.5),
        showlegend=False, hoverinfo="skip",
    ))
    # hub circle
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode="markers",
        marker=dict(size=14, color=color,
                    line=dict(color="#faf9f5", width=2.5)),
        showlegend=False, hoverinfo="skip",
    ))

    # ── Segment boundary tick marks ───────────────────────────────────────────
    for i in range(6):
        a = math.radians(180 - i * 36)
        fig.add_trace(go.Scatter(
            x=[0.90 * math.cos(a), 0.97 * math.cos(a)],
            y=[0.90 * math.sin(a), 0.97 * math.sin(a)],
            mode="lines",
            line=dict(color="#b0a898", width=1.5),
            showlegend=False, hoverinfo="skip",
        ))

    # ── Segment name labels (centre of each arc) ──────────────────────────────
    for i, (name, seg_col) in enumerate(zip(seg_names, seg_colors)):
        center_deg = 180 - (i + 0.5) * 36
        center_rad = math.radians(center_deg)
        lr = r_out + 0.14
        lx = lr * math.cos(center_rad)
        ly = lr * math.sin(center_rad)
        is_active = (i == state.value)
        fig.add_annotation(
            x=lx, y=ly,
            text=name.replace("\n", "<br>"),
            showarrow=False,
            font=dict(
                size=9 if not is_active else 10,
                color=seg_col if is_active else "#8a857c",
                family="Inter, sans-serif",
            ),
            align="center",
        )

    # ── Centre text annotations ───────────────────────────────────────────────
    state_display = {
        "QUIET": "Quiet",
        "PREHEATING": "Pre-Heating",
        "THERMAL_INSTABILITY": "Thermal Instability",
        "IGNITION": "Ignition",
        "CRITICAL": "Critical",
    }
    fig.add_annotation(
        x=0, y=0.08,
        text="Instability Ladder",
        showarrow=False,
        font=dict(size=12, color="#8a857c",
                  family="'Cormorant Garamond', serif"),
    )
    fig.add_annotation(
        x=0, y=-0.22,
        text=f"<b>{state_display.get(state.name, state.name)}</b>",
        showarrow=False,
        font=dict(size=19, color=color,
                  family="'Cormorant Garamond', serif"),
    )
    fig.add_annotation(
        x=0, y=-0.42,
        text=f"FAI {fai:.2f}  ·  NRI {nri_sigma:.1f}σ",
        showarrow=False,
        font=dict(size=11, color="#59554e",
                  family="Inter, sans-serif"),
    )

    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=12, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, range=[-1.35, 1.35], scaleanchor="y"),
        yaxis=dict(visible=False, range=[-0.62, 1.18]),
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

