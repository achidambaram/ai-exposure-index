from __future__ import annotations

import difflib
import sys
from pathlib import Path
from typing import List, Tuple

# Ensure the project root is on sys.path so absolute imports work
# when Streamlit executes this file as a standalone script.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ai_exposure_api.config import (
    ABILITY_CONTRIBUTIONS_FILE,
    METRICS_FILE,
    OCCUPATION_LOOKUP_FILE,
    TRAINING_FILE,
)
from ai_exposure_api.data_pipeline import TASK_AUTOMATION_FILE
from ai_exposure_api.utils import load_json

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Exposure Index",
    page_icon="\U0001f916",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---- Global ---- */
    .block-container { padding-top: 2rem; }

    /* ---- Metric cards ---- */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8f9fc 0%, #eef1f8 100%);
        border: 1px solid #e2e6ee;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,.06);
    }

    /* ---- Risk-level badges ---- */
    .risk-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .risk-green  { background: #dcfce7; color: #166534; }
    .risk-yellow { background: #fef9c3; color: #854d0e; }
    .risk-red    { background: #fee2e2; color: #991b1b; }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] > div { padding-top: 1.5rem; }

    /* ---- Score bar ---- */
    .score-bar-bg {
        width: 100%; height: 18px;
        background: #e5e7eb; border-radius: 10px;
        overflow: hidden;
    }
    .score-bar-fill {
        height: 100%; border-radius: 10px;
        transition: width .4s ease;
    }

    /* ---- Disclaimer box ---- */
    .disclaimer-box {
        background: #f0f4ff;
        border-left: 4px solid #6366f1;
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 0.88rem;
        margin: 12px 0;
        color: #334155;
    }

    /* ---- Tooltip-style helper ---- */
    .term-help {
        border-bottom: 1px dashed #94a3b8;
        cursor: help;
        color: inherit;
    }

    /* ---- Pipeline step ---- */
    .pipeline-step {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .pipeline-arrow {
        font-size: 1.5rem;
        color: #94a3b8;
        text-align: center;
        padding: 4px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
RISK_COLORS = {"Green": "#22c55e", "Yellow": "#eab308", "Red": "#ef4444"}
RISK_CSS = {"Green": "risk-green", "Yellow": "risk-yellow", "Red": "risk-red"}

# ---------------------------------------------------------------------------
# Target occupations — only these 4 are shown in the app
# ---------------------------------------------------------------------------
TARGET_SOCS = {
    "51-8012": "NERC-Certified System Operators",
    "17-2071": "Electric Distribution Planners",
    "49-2095": "Relay Technicians",
    "41-9022": "Utility Right of Way (ROW) Agents",
}

# Plain-English risk labels
RISK_LABELS = {
    "Green": "Low Exposure",
    "Yellow": "Moderate Exposure",
    "Red": "High Exposure",
}

RISK_DESCRIPTIONS = {
    "Green": "Most tasks in this occupation are difficult for current AI to perform.",
    "Yellow": "Some tasks in this occupation can be assisted or partially automated by AI.",
    "Red": "Many tasks in this occupation overlap with current AI capabilities.",
}


def risk_badge(band: str) -> str:
    css = RISK_CSS.get(band, "risk-yellow")
    label = RISK_LABELS.get(band, band)
    return f'<span class="risk-badge {css}">{label}</span>'


def score_bar(score: float, band: str) -> str:
    color = RISK_COLORS.get(band, "#6b7280")
    pct = max(0, min(100, score * 100))
    return (
        f'<div class="score-bar-bg">'
        f'<div class="score-bar-fill" style="width:{pct:.1f}%;background:{color};"></div>'
        f"</div>"
    )


def score_pct(score: float) -> str:
    """Format score as a percentage for readability."""
    return f"{score * 100:.0f}%"


def fuzzy_rank(query: str, candidates: List[str], limit: int = 10) -> List[Tuple[str, float]]:
    q = query.strip().lower()
    scored = []
    for candidate in candidates:
        c = candidate.lower()
        ratio = difflib.SequenceMatcher(None, q, c).ratio()
        if q in c:
            ratio += 0.35
        scored.append((candidate, round(min(ratio, 1.0), 4)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:limit]


def show_disclaimer() -> None:
    """Display the standard disclaimer about AI exposure scores."""
    st.markdown(
        '<div class="disclaimer-box">'
        "<strong>Important context:</strong> This tool estimates how much an occupation's tasks "
        "overlap with current AI capabilities. It is <strong>not</strong> a prediction that jobs will disappear. "
        "Many factors determine career outcomes, including experience, education, adaptability, "
        "interpersonal skills, and industry context. Use these results as one input among many "
        "when thinking about career planning."
        "</div>",
        unsafe_allow_html=True,
    )


def explain_score(score: float, band: str) -> str:
    """Return a plain-English sentence explaining what the score means."""
    pct = score * 100
    if band == "Red":
        return f"About {pct:.0f}% of this occupation's skill profile overlaps with AI capabilities, suggesting **high exposure** to AI-driven change."
    elif band == "Yellow":
        return f"About {pct:.0f}% of this occupation's skill profile overlaps with AI capabilities, suggesting **moderate exposure** to AI-driven change."
    else:
        return f"About {pct:.0f}% of this occupation's skill profile overlaps with AI capabilities, suggesting **low exposure** to AI-driven change."


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_lookup() -> pd.DataFrame:
    df = pd.read_csv(OCCUPATION_LOOKUP_FILE)
    df = df[df["soc_code"].astype(str).isin(TARGET_SOCS.keys())]
    df["role_label"] = df["soc_code"].astype(str).map(TARGET_SOCS)
    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_contributions() -> pd.DataFrame:
    df = pd.read_csv(ABILITY_CONTRIBUTIONS_FILE)
    return df[df["soc_code"].astype(str).isin(TARGET_SOCS.keys())].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_task_automation() -> pd.DataFrame:
    if not TASK_AUTOMATION_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(TASK_AUTOMATION_FILE)
    return df[df["soc_code"].astype(str).isin(TARGET_SOCS.keys())].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_training() -> pd.DataFrame:
    return pd.read_csv(TRAINING_FILE)


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    return load_json(METRICS_FILE)


def data_ready() -> bool:
    return OCCUPATION_LOOKUP_FILE.exists() and TRAINING_FILE.exists()


# ---------------------------------------------------------------------------
# Shared detail component (used in Search page expanders)
# ---------------------------------------------------------------------------
def _show_occupation_detail(row: pd.Series, contrib_df: pd.DataFrame, task_df: pd.DataFrame | None = None) -> None:
    if pd.notna(row.get("description")):
        st.markdown(f"*{row['description']}*")

    # Plain-English explanation
    st.markdown(explain_score(row["aei_score"], row["risk_band"]))

    # ---- Task Automation Breakdown ----------------------------------------
    occ_tasks = pd.DataFrame()
    if task_df is not None and not task_df.empty:
        occ_tasks = task_df[task_df["soc_code"].astype(str) == str(row["soc_code"])]

    if not occ_tasks.empty:
        st.markdown("---")
        st.markdown("### Task-Level Automation Breakdown")
        st.caption(
            "Each core task is rated for how easily AI could assist or automate it, "
            "based on the skills it requires. Job importance shows how critical "
            "O*NET rates each task for the role."
        )

        # Summary counts
        high_count = len(occ_tasks[occ_tasks["automation_level"] == "High"])
        mod_count = len(occ_tasks[occ_tasks["automation_level"] == "Moderate"])
        low_count = len(occ_tasks[occ_tasks["automation_level"] == "Low"])
        total = len(occ_tasks)

        tc1, tc2, tc3 = st.columns(3)
        tc1.metric(
            "Easily Automatable",
            f"{high_count} of {total} tasks",
            f"{high_count / total * 100:.0f}%",
        )
        tc2.metric(
            "Partially Automatable",
            f"{mod_count} of {total} tasks",
            f"{mod_count / total * 100:.0f}%",
        )
        tc3.metric(
            "Hard to Automate",
            f"{low_count} of {total} tasks",
            f"{low_count / total * 100:.0f}%",
        )

        # Horizontal stacked bar for visual summary
        fig_stack = go.Figure()
        fig_stack.add_trace(go.Bar(
            y=["Tasks"], x=[high_count], name="Easily Automatable",
            orientation="h", marker_color="#ef4444",
            text=[f"{high_count}"], textposition="inside",
        ))
        fig_stack.add_trace(go.Bar(
            y=["Tasks"], x=[mod_count], name="Partially Automatable",
            orientation="h", marker_color="#eab308",
            text=[f"{mod_count}"], textposition="inside",
        ))
        fig_stack.add_trace(go.Bar(
            y=["Tasks"], x=[low_count], name="Hard to Automate",
            orientation="h", marker_color="#22c55e",
            text=[f"{low_count}"], textposition="inside",
        ))
        fig_stack.update_layout(
            barmode="stack", height=100,
            margin=dict(t=5, b=5, l=5, r=5),
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_stack, use_container_width=True)

        # Task list grouped by automation level
        LEVEL_CONFIG = [
            ("High", "\U0001f534 Easily Automatable Tasks", "These tasks have high overlap with current AI capabilities."),
            ("Moderate", "\U0001f7e1 Partially Automatable Tasks", "AI can assist with these tasks but likely needs human oversight."),
            ("Low", "\U0001f7e2 Hard-to-Automate Tasks", "These tasks require human judgment, physical presence, or complex reasoning that AI handles poorly."),
        ]
        for level, heading, explanation in LEVEL_CONFIG:
            level_tasks = occ_tasks[occ_tasks["automation_level"] == level]
            if level_tasks.empty:
                continue
            st.markdown(f"**{heading} ({len(level_tasks)})**")
            st.caption(explanation)
            for _, t in level_tasks.iterrows():
                auto_pct = t["automation_score"] * 100
                imp_pct = t["task_importance"] / 5 * 100  # O*NET importance is 1-5 scale
                st.markdown(
                    f"- {t['task_description']}  \n"
                    f"  *Automation potential: {auto_pct:.1f}% · Job importance: {imp_pct:.1f}%*"
                )

    # ---- Skill-Level Breakdown --------------------------------------------
    st.markdown("---")
    st.markdown("### Skill-Level Breakdown")

    occ_contrib = contrib_df[contrib_df["soc_code"].astype(str) == str(row["soc_code"])]
    if occ_contrib.empty:
        st.caption("No detailed skill data available for this occupation.")
        return

    top = occ_contrib.nlargest(10, "ability_contribution")

    # Explain "why" — top contributing skills
    st.markdown("The skills below contribute most to this occupation's AI exposure:")
    top3 = top.head(3)
    for _, ab in top3.iterrows():
        ab_pct = ab["ability_contribution"] / row["aei_score"] * 100 if row["aei_score"] > 0 else 0
        exposure_label = "high" if ab["exposure_sigmoid"] > 0.66 else ("moderate" if ab["exposure_sigmoid"] > 0.33 else "low")
        importance_pct = ab["ability_weight"] * 100
        st.markdown(
            f"- **{ab['ability_name']}** — accounts for ~{ab_pct:.0f}% of the score "
            f"({exposure_label} AI overlap, {importance_pct:.0f}% job importance)"
        )

    # Chart with readable labels
    fig = px.bar(
        top.sort_values("ability_contribution"),
        y="ability_name",
        x="ability_contribution",
        orientation="h",
        color="exposure_sigmoid",
        color_continuous_scale="RdYlGn_r",
        labels={
            "ability_contribution": "Contribution to Score",
            "ability_name": "",
            "exposure_sigmoid": "AI Overlap",
        },
    )
    fig.update_layout(
        height=320,
        margin=dict(t=10, b=20, l=10, r=10),
        coloraxis_colorbar=dict(thickness=12, len=0.6),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Detailed skill data**")
    st.caption(
        "**Skill Importance** = how much this skill matters for the job (from O*NET) | "
        "**AI Overlap** = how well current AI can perform it (from AIOE) | "
        "**Contribution** = importance × AI overlap, showing how much it adds to the overall score"
    )
    display_df = top[["ability_name", "ability_weight", "exposure_sigmoid", "ability_contribution"]].copy()
    display_df = display_df.rename(columns={
        "ability_name": "Skill",
        "ability_weight": "Skill Importance",
        "exposure_sigmoid": "AI Overlap",
        "ability_contribution": "Contribution to Score",
    })
    st.dataframe(
        display_df.reset_index(drop=True).style.format(
            {"Skill Importance": "{:.0%}", "AI Overlap": "{:.0%}", "Contribution to Score": "{:.0%}"}
        ),
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## \U0001f916 AI Exposure Index")
    st.caption("Understand how AI capabilities relate to different occupations")
    st.divider()
    page = st.radio(
        "Navigate",
        [
            "\U0001f4ca Dashboard",
            "\U0001f50d Search",
            "\U0001f4cb Compare",
            "\U00002139 How It Works",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown(
        "**What do the levels mean?**\n\n"
        "- \U0001f7e2 **Low Exposure** (under 33%)\n"
        "  Few tasks overlap with AI\n\n"
        "- \U0001f7e1 **Moderate Exposure** (33%-66%)\n"
        "  Some tasks can be AI-assisted\n\n"
        "- \U0001f534 **High Exposure** (above 66%)\n"
        "  Many tasks overlap with AI capabilities"
    )
    st.divider()
    st.caption(
        "This tool measures task overlap with AI — "
        "it does **not** predict job loss."
    )

# ---------------------------------------------------------------------------
# Guard: data must exist
# ---------------------------------------------------------------------------
if not data_ready():
    st.error(
        "Data assets not found. Run the data pipeline first:\n\n"
        "```bash\npython -m ai_exposure_api.cli fetch-data\npython -m ai_exposure_api.cli train-model\n```"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
lookup = load_lookup()
contributions = load_contributions()
task_automation = load_task_automation()
metrics = load_metrics().get("rule_based", {})

# =========================================================================
# PAGE: Dashboard
# =========================================================================
if page == "\U0001f4ca Dashboard":
    st.markdown("# \U0001f4ca Dashboard")
    st.caption("AI exposure overview for 4 utility-sector roles")

    show_disclaimer()

    # -- Score comparison bar chart for all 4 roles -------------------------
    st.subheader("AI Exposure by Role")
    dash_df = lookup.sort_values("aei_score", ascending=True).copy()
    dash_df["display_name"] = dash_df["role_label"] + "\n(" + dash_df["occupation_title"] + ")"
    fig_bar = px.bar(
        dash_df,
        y="display_name",
        x="aei_score",
        color="risk_band",
        color_discrete_map=RISK_COLORS,
        orientation="h",
        labels={"aei_score": "AI Exposure Score", "display_name": ""},
    )
    fig_bar.update_layout(
        height=300,
        margin=dict(t=10, b=20),
        showlegend=False,
        xaxis=dict(range=[0, 1.05], tickformat=".0%"),
    )
    fig_bar.add_vline(x=0.33, line_dash="dash", line_color="#22c55e", opacity=0.5, annotation_text="Low")
    fig_bar.add_vline(x=0.66, line_dash="dash", line_color="#ef4444", opacity=0.5, annotation_text="High")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # -- Detail cards for each role -----------------------------------------
    for _, row in lookup.sort_values("aei_score", ascending=False).iterrows():
        with st.container():
            c1, c2, c3 = st.columns([3, 1.5, 0.8])
            with c1:
                st.markdown(f"### {row['role_label']}")
                st.caption(f"{row['occupation_title']} (SOC {row['soc_code']})")
            with c2:
                st.markdown(score_bar(row["aei_score"], row["risk_band"]), unsafe_allow_html=True)
                st.caption(f"AI Exposure: {score_pct(row['aei_score'])}")
            with c3:
                st.markdown(risk_badge(row["risk_band"]), unsafe_allow_html=True)

            with st.expander("Why this rating? See details"):
                _show_occupation_detail(row, contributions, task_automation)
            st.markdown("")


# =========================================================================
# PAGE: Search (role detail view)
# =========================================================================
elif page == "\U0001f50d Search":
    st.markdown("# \U0001f50d Role Detail View")
    st.markdown("Select a role to explore its AI exposure breakdown.")

    show_disclaimer()

    role_options = {row["role_label"]: row["soc_code"] for _, row in lookup.iterrows()}
    selected_role = st.selectbox("Choose a role", list(role_options.keys()))

    if selected_role:
        soc = role_options[selected_role]
        row = lookup[lookup["soc_code"].astype(str) == soc].iloc[0]

        st.markdown(f"## {row['role_label']}")
        st.caption(f"O*NET Occupation: {row['occupation_title']} (SOC {row['soc_code']})")

        c1, c2, c3 = st.columns([3, 1.5, 0.8])
        with c1:
            st.markdown(score_bar(row["aei_score"], row["risk_band"]), unsafe_allow_html=True)
        with c2:
            st.markdown(f"**AI Exposure: {score_pct(row['aei_score'])}**")
        with c3:
            st.markdown(risk_badge(row["risk_band"]), unsafe_allow_html=True)

        st.markdown("")
        _show_occupation_detail(row, contributions, task_automation)


# =========================================================================
# PAGE: Compare
# =========================================================================
elif page == "\U0001f4cb Compare":
    st.markdown("# \U0001f4cb Compare Roles")
    st.caption("Compare AI exposure across utility-sector roles side by side")

    show_disclaimer()

    all_roles = lookup["role_label"].dropna().unique().tolist()
    selected = st.multiselect(
        "Select roles to compare",
        all_roles,
        default=all_roles,
    )

    if not selected:
        st.info("Select at least one role above.")
        st.stop()

    comp = lookup[lookup["role_label"].isin(selected)].copy()

    # -- Score comparison bar chart -----------------------------------------
    fig_bar = px.bar(
        comp.sort_values("aei_score", ascending=True),
        y="role_label",
        x="aei_score",
        color="risk_band",
        color_discrete_map=RISK_COLORS,
        orientation="h",
        labels={"aei_score": "AI Exposure Score", "role_label": ""},
    )
    fig_bar.update_layout(
        height=max(220, len(selected) * 60),
        margin=dict(t=10, b=20),
        showlegend=False,
        xaxis=dict(range=[0, 1.05], tickformat=".0%"),
    )
    fig_bar.add_vline(x=0.33, line_dash="dash", line_color="#22c55e", opacity=0.5, annotation_text="Low")
    fig_bar.add_vline(x=0.66, line_dash="dash", line_color="#ef4444", opacity=0.5, annotation_text="High")
    st.plotly_chart(fig_bar, use_container_width=True)

    # -- Detail cards -------------------------------------------------------
    st.markdown("---")
    cols = st.columns(min(len(selected), 4))
    for idx, (_, row) in enumerate(comp.iterrows()):
        with cols[idx % len(cols)]:
            st.markdown(f"### {row['role_label']}")
            st.caption(f"{row['occupation_title']} (SOC {row['soc_code']})")
            st.markdown(
                f"AI Exposure: **{score_pct(row['aei_score'])}** {risk_badge(row['risk_band'])}",
                unsafe_allow_html=True,
            )
            st.caption(RISK_DESCRIPTIONS.get(row["risk_band"], ""))

            # Top contributors
            occ_contrib = contributions[contributions["soc_code"].astype(str) == str(row["soc_code"])]
            if not occ_contrib.empty:
                top5 = occ_contrib.nlargest(5, "ability_contribution")
                st.markdown("**Top contributing skills:**")
                fig_c = px.bar(
                    top5,
                    x="ability_contribution",
                    y="ability_name",
                    orientation="h",
                    color_discrete_sequence=["#6366f1"],
                    labels={"ability_contribution": "Contribution", "ability_name": ""},
                )
                fig_c.update_layout(
                    height=200,
                    margin=dict(t=5, b=5, l=5, r=5),
                    showlegend=False,
                    xaxis=dict(showticklabels=False),
                )
                st.plotly_chart(fig_c, use_container_width=True, key=f"compare_{idx}")

    # -- Radar chart --------------------------------------------------------
    if len(selected) >= 2:
        st.markdown("---")
        st.subheader("Skill profile comparison")
        st.caption("Shows how the top shared skills compare across selected roles")

        sel_socs = comp["soc_code"].astype(str).tolist()
        sel_contrib = contributions[contributions["soc_code"].astype(str).isin(sel_socs)]
        top_abilities = (
            sel_contrib.groupby("ability_name")["ability_contribution"]
            .mean()
            .nlargest(8)
            .index.tolist()
        )

        # Build SOC-to-label mapping for the legend
        soc_to_label = dict(zip(comp["soc_code"].astype(str), comp["role_label"]))

        fig_radar = go.Figure()
        for soc, label in soc_to_label.items():
            occ_c = sel_contrib[
                (sel_contrib["soc_code"].astype(str) == soc)
                & (sel_contrib["ability_name"].isin(top_abilities))
            ]
            values = []
            for ab in top_abilities:
                match = occ_c[occ_c["ability_name"] == ab]
                values.append(float(match["ability_contribution"].values[0]) if not match.empty else 0)
            fig_radar.add_trace(
                go.Scatterpolar(r=values + [values[0]], theta=top_abilities + [top_abilities[0]], name=label, fill="toself", opacity=0.55)
            )
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, showticklabels=False)),
            height=450,
            margin=dict(t=30, b=30),
        )
        st.plotly_chart(fig_radar, use_container_width=True)


# =========================================================================
# PAGE: How It Works
# =========================================================================
elif page == "\U00002139 How It Works":
    st.markdown("# \U00002139 How It Works")
    st.markdown("A simple explanation of how AI exposure scores are calculated.")

    st.markdown("---")

    # Pipeline visualization
    st.subheader("The scoring pipeline")
    st.markdown(
        "Each occupation's score is built in four steps:"
    )

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.markdown(
            '<div class="pipeline-step">'
            "<strong>1. Skill Data</strong><br>"
            "<small>We start with the skills required for each job, "
            "sourced from the U.S. Department of Labor (O*NET).</small>"
            "</div>",
            unsafe_allow_html=True,
        )
    with p2:
        st.markdown(
            '<div class="pipeline-step">'
            "<strong>2. AI Capability Match</strong><br>"
            "<small>Each skill is rated for how well current AI can "
            "perform it, based on published research (AIOE).</small>"
            "</div>",
            unsafe_allow_html=True,
        )
    with p3:
        st.markdown(
            '<div class="pipeline-step">'
            "<strong>3. Weighted Scoring</strong><br>"
            "<small>Skills are weighted by how important they are to the job. "
            "More important skills count more toward the score.</small>"
            "</div>",
            unsafe_allow_html=True,
        )
    with p4:
        st.markdown(
            '<div class="pipeline-step">'
            "<strong>4. Final Score</strong><br>"
            "<small>All weighted skill scores are combined into a single "
            "percentage (0-100%) representing overall AI exposure.</small>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("")

    # Key terms
    st.subheader("Key terms explained")
    st.markdown(
        "| Term in the app | What it means |\n"
        "|---|---|\n"
        "| **AI Exposure Score** | A percentage (0-100%) showing how much a job's required skills overlap with what AI can currently do. Higher = more overlap. |\n"
        "| **Low / Moderate / High Exposure** | The exposure level: Low (under 33%), Moderate (33-66%), High (above 66%). |\n"
        "| **Skill Importance** | How critical a particular skill is for performing the job (from O*NET data). |\n"
        "| **AI Overlap** | How capable current AI systems are at performing a specific skill (from AIOE research). |\n"
        "| **Contribution to Score** | How much a single skill influences the overall exposure score for a job. |\n"
        "| **SOC Code** | The Standard Occupational Classification code — a government ID number for each occupation. |"
    )

    st.markdown("---")

    # Data sources
    st.subheader("Data sources")
    st.markdown(
        "- **O*NET 30.2** — The Occupational Information Network from the U.S. Department of Labor. "
        "Provides detailed skill requirements for ~1,000 occupations.\n"
        "- **AIOE (AI Occupational Exposure)** — A research dataset that maps AI capabilities to "
        "specific human skills, estimating how well AI can perform each one."
    )

    st.markdown("---")

    # Important caveats
    st.subheader("What this tool does NOT tell you")
    st.markdown(
        "- **It does not predict job loss.** A high score means task overlap, not that the job will disappear.\n"
        "- **It does not account for** organizational inertia, regulation, cost of adoption, or social preferences.\n"
        "- **It is a snapshot** based on current AI capabilities. As AI evolves, these scores will change.\n"
        "- **Individual outcomes depend on many factors:** experience, specialization, adaptability, location, "
        "industry, and interpersonal skills all matter."
    )

    show_disclaimer()
