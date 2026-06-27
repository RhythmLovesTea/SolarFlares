from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .ladder import STATE_COLORS, InstabilityLadderState


def build_ladder_gauge(state, fai: float, nri_sigma: float) -> go.Figure:
    color = STATE_COLORS[state]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=state.value,
            number={"font": {"size": 46, "color": color}},
            title={"text": "", "font": {"size": 1}},
            gauge={
                "axis": {
                    "range": [0, 4],
                    "tickvals": [0, 1, 2, 3, 4],
                    "ticktext": ["G", "Y", "O", "R", "C"],
                    "tickfont": {"size": 13, "color": "#6c6a64"},
                },
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 1], "color": "#eaf3eb"},
                    {"range": [1, 2], "color": "#fdf6e3"},
                    {"range": [2, 3], "color": "#fef0e7"},
                    {"range": [3, 4], "color": "#fdeaea"},
                ],
                "bgcolor": "#efe9de",
                "borderwidth": 0,
            },
        )
    )
    fig.update_layout(
        height=300,
        margin=dict(l=25, r=25, t=20, b=36),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#141413",
        annotations=[
            dict(text="Instability Ladder", x=0.5, y=0.98, showarrow=False, font=dict(size=15, color=color, family="'Cormorant Garamond', serif")),
            dict(text=f"FAI {fai:.2f} | NRI {nri_sigma:.1f}σ", x=0.5, y=-0.05, showarrow=False, font=dict(size=12, color="#6c6a64", family="Inter, sans-serif")),
        ],
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

