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
        color: #1e293b;
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #1e293b !important;
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
        color: #1e293b;
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

    # ---- How Was This Score Calculated? (full transparency) ------------------
    st.markdown("---")
    st.markdown("### How was this score calculated?")
    st.markdown(
        "The section below shows **every input and calculation step** that produced "
        f"this occupation's AI Exposure Score of **{row['aei_score'] * 100:.0f}%**. "
        "See the *How It Works* page for a full explanation of the methodology."
    )

    with st.expander("View full calculation details", expanded=False):
        # Step 1: Show all skills with raw inputs
        st.markdown("#### All skills and their raw inputs")
        st.markdown(
            "This table shows every skill used in the calculation, including the "
            "original ratings from O*NET and the AI exposure rating from research. "
            "These are the raw numbers before any processing."
        )

        all_skills = occ_contrib.sort_values("ability_contribution", ascending=False).copy()
        raw_df = all_skills[[
            "ability_name", "importance_raw", "level_raw", "ability_exposure",
            "importance_norm_global", "level_norm_global",
            "exposure_sigmoid", "ability_weight", "ability_contribution",
        ]].copy()
        raw_df = raw_df.rename(columns={
            "ability_name": "Skill",
            "importance_raw": "Importance (raw, 1-7 scale)",
            "level_raw": "Level (raw, 1-7 scale)",
            "ability_exposure": "AI Exposure Rating (raw)",
            "importance_norm_global": "Importance (normalized 0-100%)",
            "level_norm_global": "Level (normalized 0-100%)",
            "exposure_sigmoid": "AI Overlap (after sigmoid, 0-100%)",
            "ability_weight": "Skill Weight (share of job)",
            "ability_contribution": "Contribution to Score",
        })
        st.dataframe(
            raw_df.reset_index(drop=True).style.format({
                "Importance (raw, 1-7 scale)": "{:.2f}",
                "Level (raw, 1-7 scale)": "{:.2f}",
                "AI Exposure Rating (raw)": "{:.3f}",
                "Importance (normalized 0-100%)": "{:.0%}",
                "Level (normalized 0-100%)": "{:.0%}",
                "AI Overlap (after sigmoid, 0-100%)": "{:.0%}",
                "Skill Weight (share of job)": "{:.1%}",
                "Contribution to Score": "{:.1%}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        # Step 2: Show the aggregation
        st.markdown("#### How the final score was built")
        total_contribution = all_skills["ability_contribution"].sum()
        n_skills = len(all_skills)
        top5_contrib = all_skills.head(5)["ability_contribution"].sum()
        top5_pct = top5_contrib / total_contribution * 100 if total_contribution > 0 else 0

        st.markdown(
            f"1. **{n_skills} skills** were evaluated for this occupation.\n"
            f"2. Each skill's **weight** was calculated as: "
            f"70% × normalized importance + 30% × normalized level, "
            f"then divided by the total so all weights add up to 100%.\n"
            f"3. Each skill's **contribution** = its weight × its AI overlap.\n"
            f"4. All contributions were summed: **raw score = {total_contribution:.4f}**\n"
            f"5. This raw score was then normalized across all 682 occupations to produce "
            f"the final score of **{row['aei_score'] * 100:.0f}%**.\n"
        )

        st.markdown(
            f"The top 5 skills account for **{top5_pct:.0f}%** of this occupation's score. "
            f"The remaining {n_skills - 5} skills account for {100 - top5_pct:.0f}%."
        )

        # Step 3: Reading guide
        st.markdown("#### How to read this data")
        st.markdown(
            "- **Importance (raw)** and **Level (raw)** come directly from O*NET — "
            "these are government ratings of what the job requires.\n"
            "- **AI Exposure Rating (raw)** comes from published AI research (AIOE) — "
            "this is a researcher's assessment of how well AI performs each skill.\n"
            "- **Normalized** columns convert raw values to a 0–100% scale so they can be compared fairly.\n"
            "- **Skill Weight** is each skill's share of the total job profile "
            "(all weights add up to 100%).\n"
            "- **Contribution** is the portion of the final score that this skill is responsible for. "
            "A high contribution means the skill is both important to the job AND highly automatable by AI.\n\n"
            "If you believe a rating seems incorrect for your organization's context, "
            "remember that these are national averages — your specific role may differ."
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
        st.markdown(
            "This radar chart compares the **top 8 skills** that contribute most to AI exposure "
            "across the selected roles. Each spoke is a skill, and the distance from the center "
            "shows how much that skill **contributes to the role's AI exposure score** "
            "(further out = larger contribution). A role with a bigger overall shape has "
            "higher AI exposure across more skills. Use this to spot which skills drive "
            "differences between roles."
        )

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

    # ---- Detailed Methodology ------------------------------------------------
    st.subheader("Detailed scoring methodology")
    st.markdown(
        "Below is the exact process used to calculate every score in this tool. "
        "No steps are hidden — what you see here is the complete calculation."
    )

    st.markdown("#### Step 1: Gather the raw ingredients")
    st.markdown(
        "For every occupation, we start with two pieces of information about each skill:\n\n"
        "- **Importance** — How important is this skill to the job? "
        "Rated on a 1–7 scale by the U.S. Department of Labor (O*NET). "
        "For example, *Written Comprehension* might be rated 6 out of 7 for a Manager role.\n"
        "- **Level** — How advanced does this skill need to be? "
        "Also rated 1–7 by O*NET. A Manager might need level 5 *Written Comprehension*, "
        "while an entry-level role might only need level 2.\n\n"
        "We also get a third number from published AI research (AIOE):\n\n"
        "- **AI Exposure Rating** — How well can current AI perform this skill? "
        "This is a number from the research literature that rates each human skill "
        "based on how capable today's AI systems are at performing it."
    )
    st.info(
        "**Why this step?** To measure AI exposure, we need to know two things: "
        "what skills a job actually requires (from O*NET, a trusted government source), "
        "and how good AI is at each of those skills (from peer-reviewed research). "
        "Without both pieces, we would be guessing. Using established, public data sources "
        "ensures the results are grounded in evidence, not opinion."
    )

    st.markdown("#### Step 2: Standardize the numbers")
    st.markdown(
        "Different skills have different raw scales, so we put everything on a common 0–100% scale "
        "using a method called **min-max normalization**. This simply means:\n\n"
        "> *Take each value, subtract the lowest value across all occupations, "
        "and divide by the range (highest minus lowest). The result is always between 0% and 100%.*\n\n"
        "For example, if importance ratings across all occupations range from 1.2 to 6.8, "
        "a skill with importance 4.0 would become: (4.0 − 1.2) ÷ (6.8 − 1.2) = **50%**.\n\n"
        "We also convert the AI Exposure Rating into a 0–100% scale using a "
        "**sigmoid function** — this is a standard mathematical curve that smoothly maps "
        "any number into a value between 0% and 100%, preventing extreme outliers from "
        "dominating the results."
    )
    st.info(
        "**Why this step?** The raw numbers from different sources use different scales — "
        "O*NET rates importance on 1–7 while AI exposure ratings use a completely different range. "
        "If we compared them directly, the numbers with larger scales would unfairly dominate. "
        "Standardizing puts everything on the same 0–100% playing field so that no single "
        "data source overpowers the others."
    )
    st.markdown(
        "**Why two different methods (min-max and sigmoid)?** Each method is chosen to match "
        "the type of data it handles:\n\n"
        "- **Min-max normalization** is used for O*NET importance and level ratings (the 1–7 scores). "
        "These values are already clean and bounded — they have a known range with no extreme outliers. "
        "Min-max simply rescales them to 0–100% by mapping the lowest observed value to 0% and the "
        "highest to 100%. It preserves the original spacing between values, so a skill rated 5 out of 7 "
        "stays proportionally higher than one rated 3.\n"
        "- **Sigmoid** is used for the AIOE AI exposure ratings. Unlike O*NET scores, these research "
        "ratings don't have a fixed range — they can be very large positive numbers, very negative "
        "numbers, or cluster around zero in unpredictable ways. If we used min-max here, a single "
        "extreme outlier would compress all other values into a narrow band, making most skills look "
        "nearly identical. The sigmoid curve handles this gracefully: values near zero map to ~50%, "
        "large positives approach 100%, large negatives approach 0%, and the transition is smooth. "
        "No single extreme value can distort the rest.\n\n"
        "**In short:** min-max works well when the data is already clean and bounded (O*NET). "
        "Sigmoid is needed when the data has no guaranteed range and could contain outliers "
        "that would break min-max (AIOE research ratings)."
    )

    st.markdown("#### Step 3: Combine importance and level into a single skill weight")
    st.markdown(
        "For each skill, we blend the importance and level scores together using a fixed ratio:\n\n"
        "> **Skill Weight = 70% × Importance + 30% × Level**\n\n"
        "We weight importance more heavily because a skill used frequently creates more AI exposure "
        "than a skill used rarely at an advanced level — and AI capability doesn't follow human difficulty "
        "(AI excels at complex math but struggles with basic physical coordination).\n\n"
        "Then, we make sure all skill weights for a single occupation add up to 100%. "
        "This way, each skill's weight represents its share of the overall job profile."
    )
    st.info(
        "**Why this step?** A job uses many skills, but not all skills matter equally. "
        "A pilot relies heavily on *Spatial Orientation* but barely uses *Persuasion* — "
        "so *Spatial Orientation* should count much more when assessing that job's AI exposure. "
        "We use a 70/30 blend because how frequently and critically a skill is used (importance) "
        "is a stronger signal of exposure risk than how advanced the skill needs to be (level). "
        "For example, a skill used every day at a basic level is more relevant to AI exposure "
        "than a skill rarely used but at an expert level. Normalizing to 100% ensures we are "
        "measuring each skill's *share* of the job — so occupations with many skills and "
        "occupations with few skills are compared fairly."
    )

    st.markdown("#### Step 4: Calculate each skill's contribution")
    st.markdown(
        "Now we multiply two things together for each skill:\n\n"
        "> **Contribution = Skill Weight × AI Overlap**\n\n"
        "This tells us: *of the portion of the job that depends on this skill, "
        "how much can AI handle?*\n\n"
        "- A skill that is very important (high weight) AND highly automatable (high AI overlap) "
        "will contribute a lot to the score.\n"
        "- A skill that is unimportant OR not automatable will contribute very little."
    )
    st.info(
        "**Why this step?** This is where the two key questions come together: "
        "*\"How much does this job depend on this skill?\"* and *\"How good is AI at this skill?\"* "
        "Multiplying them captures the intuition that AI exposure only matters when both conditions "
        "are true — the skill must be important to the job AND AI must be capable of performing it. "
        "A skill that AI is great at but the job barely uses? Low contribution. "
        "A skill the job relies on heavily but AI cannot do? Also low contribution. "
        "Only skills that are both important and automatable drive the score up."
    )

    st.markdown("#### Step 5: Sum up and normalize the final score")
    st.markdown(
        "We add up all the skill contributions for an occupation to get a raw score. "
        "Then we normalize this across all 682 occupations using the same min-max method "
        "from Step 2, so the final AI Exposure Score is on a 0–100% scale where:\n\n"
        "- **0%** = the least-exposed occupation in the dataset\n"
        "- **100%** = the most-exposed occupation in the dataset\n\n"
        "Finally, the score is placed into a risk band:\n\n"
        "| Score range | Risk band | What it means |\n"
        "|---|---|---|\n"
        "| 0% – 32% | Low Exposure | The job's skills have relatively little overlap with current AI capabilities |\n"
        "| 33% – 66% | Moderate Exposure | A meaningful portion of the job's skills overlap with what AI can do |\n"
        "| 67% – 100% | High Exposure | Most of the job's core skills overlap with current AI capabilities |"
    )
    st.info(
        "**Why this step?** Summing the contributions gives us a single number that represents "
        "the total AI exposure for the entire job — not just one skill at a time. "
        "We then normalize across all 682 occupations so the score is *relative*: it tells you "
        "how this job compares to every other job in the dataset. Without this normalization, "
        "raw scores would be hard to interpret because they depend on how many skills a job has "
        "and how the underlying data happens to be scaled. The risk bands (Low, Moderate, High) "
        "provide a simple, actionable summary — the 33% and 66% thresholds divide all occupations "
        "into roughly equal thirds, so each band contains a similar number of occupations."
    )

    # ---- Worked Example ------------------------------------------------------
    st.markdown("---")
    st.subheader("Worked example: Relay Technicians")

    # Load real data for Relay Technicians (SOC 49-2095)
    _ex_data = contributions[contributions["soc_code"].astype(str) == "49-2095"].copy()
    _ex_data = _ex_data.sort_values("ability_contribution", ascending=False).reset_index(drop=True)
    _n_skills = len(_ex_data)

    st.markdown(
        f"Below is a real, step-by-step walkthrough using **Relay Technicians** (SOC 49-2095), "
        f"one of the four occupations tracked in this tool. The tables show **all {_n_skills} skills** "
        f"evaluated for this role, sorted from highest to lowest contribution. "
        f"Each step builds on the previous one."
    )

    # -- Step 1 table --
    st.markdown("##### Step 1 result: Gather the raw ingredients")
    st.markdown(
        "We look up each skill's ratings from the data sources. "
        "Importance and Level come from O*NET (1–7 scale). "
        "The AI Exposure Rating comes from AIOE research."
    )
    _step1 = _ex_data[["ability_name", "importance_raw", "level_raw", "ability_exposure"]].copy()
    _step1.columns = ["Skill", "Importance (O*NET, 1–7)", "Level (O*NET, 1–7)", "AI Exposure Rating (AIOE)"]
    st.dataframe(
        _step1.style.format({
            "Importance (O*NET, 1–7)": "{:.2f}",
            "Level (O*NET, 1–7)": "{:.2f}",
            "AI Exposure Rating (AIOE)": "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Skills with *negative* AI exposure ratings (like Trunk Strength at -1.721) are ones "
        "AI performs very poorly at — typically physical skills. "
        "Positive ratings mean AI handles the skill reasonably well."
    )

    # -- Step 2 table --
    st.markdown("##### Step 2 result: Standardize the numbers")
    st.markdown(
        "Now we convert each raw value to a 0-to-1 scale. Importance and Level use "
        "**min-max normalization** (rescale based on the lowest and highest values across all occupations). "
        "The AI Exposure Rating uses a **sigmoid function** (a smooth curve that maps any number to 0-to-1)."
    )
    st.markdown(
        "**Calculations performed:**\n"
        "- Importance: (raw − min across all jobs) ÷ (max − min across all jobs) → 0 to 1\n"
        "- Level: same min-max formula\n"
        "- AI Exposure: 1 ÷ (1 + e^(−raw)) → 0 to 1"
    )
    # Get global min/max from the full contributions dataset for showing formulas
    _all_contrib = pd.read_csv(ABILITY_CONTRIBUTIONS_FILE)
    _imp_min = _all_contrib["importance_raw"].min()
    _imp_max = _all_contrib["importance_raw"].max()
    _lvl_min = _all_contrib["level_raw"].min()
    _lvl_max = _all_contrib["level_raw"].max()

    _step2 = _ex_data[["ability_name", "importance_raw", "importance_norm_global", "level_raw", "level_norm_global", "ability_exposure", "exposure_sigmoid"]].copy()

    # Build formula columns showing the actual calculation
    _step2["→ Importance calc"] = _step2.apply(
        lambda r: f"({r['importance_raw']:.2f} − {_imp_min:.2f}) ÷ ({_imp_max:.2f} − {_imp_min:.2f}) = {r['importance_norm_global']:.2f}", axis=1
    )
    _step2["→ Level calc"] = _step2.apply(
        lambda r: f"({r['level_raw']:.2f} − {_lvl_min:.2f}) ÷ ({_lvl_max:.2f} − {_lvl_min:.2f}) = {r['level_norm_global']:.2f}", axis=1
    )
    _step2["→ AI Overlap calc"] = _step2.apply(
        lambda r: f"1 ÷ (1 + e^(−{r['ability_exposure']:.3f})) = {r['exposure_sigmoid']:.2f}", axis=1
    )

    _step2_display = _step2[["ability_name", "importance_raw", "→ Importance calc", "level_raw", "→ Level calc", "ability_exposure", "→ AI Overlap calc"]].copy()
    _step2_display.columns = ["Skill", "Importance (raw)", "→ Importance (normalized)", "Level (raw)", "→ Level (normalized)", "AI Exposure (raw)", "→ AI Overlap (after sigmoid)"]
    st.dataframe(
        _step2_display.style.format({
            "Importance (raw)": "{:.2f}",
            "Level (raw)": "{:.2f}",
            "AI Exposure (raw)": "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown(
        "**How to read these normalized values:**\n\n"
        "All normalized values are on a **0-to-1 scale**, where 0 is the lowest across all occupations "
        "and 1 is the highest. Think of them as a ranking relative to every other job:"
    )
    st.markdown(
        "| Value range | Importance (how much the job uses this skill) | Level (how advanced the skill needs to be) | AI Overlap (how well AI performs this skill) |\n"
        "|---|---|---|---|\n"
        "| **0.00 – 0.20** | This skill is barely used in the job — most other jobs rely on it more | The skill is needed at only a basic level — simpler than nearly all other jobs | AI is very poor at this skill (typical for physical abilities like strength or coordination) |\n"
        "| **0.21 – 0.40** | The job uses this skill less than most other jobs | Below-average complexity required | AI has limited capability with this skill |\n"
        "| **0.41 – 0.60** | Average reliance — the job uses this skill about as much as most jobs | The skill is needed at a moderate level, typical across occupations | AI has moderate capability — it can assist but may need human oversight |\n"
        "| **0.61 – 0.80** | The job relies on this skill more than most — it's a meaningful part of the work | Above-average complexity — requires solid proficiency | AI handles this skill well and could perform most related tasks |\n"
        "| **0.81 – 1.00** | This is a core skill — the job depends on it heavily, more than nearly all other jobs | Expert-level proficiency required — among the most demanding across all jobs | AI is highly capable at this skill (typical for information processing and pattern recognition) |"
    )
    st.caption(
        "The sigmoid transformation is especially visible with physical skills: "
        "negative raw values become low AI overlap scores (below 0.30), "
        "while positive raw values become high AI overlap scores (above 0.60). "
        "All values are now on the same 0-to-1 scale and ready to combine."
    )

    # -- Step 3 table --
    st.markdown("##### Step 3 result: Combine importance and level into skill weights")
    st.markdown(
        f"For each skill, we blend importance and level using the 70/30 ratio, "
        f"then divide by the total across all {_n_skills} skills so weights add up to 100%."
    )
    st.markdown(
        "**Calculations performed:**\n"
        "- Blended score = 0.70 × Importance (normalized) + 0.30 × Level (normalized)\n"
        f"- Skill Weight = Blended score ÷ sum of all {_n_skills} blended scores → expressed as a percentage of the job"
    )
    _step3 = _ex_data[["ability_name", "importance_norm_global", "level_norm_global", "ability_weight"]].copy()
    _blended = 0.70 * _ex_data["importance_norm_global"] + 0.30 * _ex_data["level_norm_global"]
    _blend_sum = _blended.sum()

    _step3["→ Blended Score calc"] = _step3.apply(
        lambda r: f"0.70×{r['importance_norm_global']:.2f} + 0.30×{r['level_norm_global']:.2f} = {0.70 * r['importance_norm_global'] + 0.30 * r['level_norm_global']:.3f}", axis=1
    )
    _step3["→ Skill Weight calc"] = _step3.apply(
        lambda r: f"{0.70 * r['importance_norm_global'] + 0.30 * r['level_norm_global']:.3f} ÷ {_blend_sum:.3f} = {r['ability_weight']:.1%}", axis=1
    )

    _step3_display = _step3[["ability_name", "importance_norm_global", "level_norm_global", "→ Blended Score calc", "→ Skill Weight calc"]].copy()
    _step3_display.columns = ["Skill", "Importance (normalized)", "Level (normalized)", "→ Blended Score", "→ Skill Weight (share of job)"]
    st.dataframe(
        _step3_display.style.format({
            "Importance (normalized)": "{:.2f}",
            "Level (normalized)": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("**How to read these values:**")
    st.markdown(
        "- **Blended Score:** A single number that captures how much this job relies on this skill, "
        "combining both how important it is (70% weight) and how advanced it needs to be (30% weight). "
        "The blended score uses the same 0-to-1 scale:"
    )
    st.markdown(
        "| Blended score range | What it means |\n"
        "|---|---|\n"
        "| **0.00 – 0.20** | This skill has very low importance and level for this job |\n"
        "| **0.21 – 0.40** | Below-average reliance on this skill |\n"
        "| **0.41 – 0.60** | Average reliance — the job uses this skill a moderate amount |\n"
        "| **0.61 – 0.80** | Above-average reliance — this skill matters quite a bit |\n"
        "| **0.81 – 1.00** | Very high reliance — this is a core skill for the job |"
    )
    st.markdown(
        "For example, Information Ordering has a blended score of **0.597** (average range) — "
        "this means it has a moderate combination of importance and required level for this job. "
        "The blended score is an intermediate value — "
        "it only becomes meaningful in the next column when converted to a share of the job.\n\n"
        f"- **Skill Weight (share of job):** The blended score divided by the sum of all {_n_skills} blended scores. "
        "This converts the blended score into each skill's proportional share of the overall job. "
        "Information Ordering's weight of **2.7%** means it accounts for 2.7% of what this job requires. "
        f"All {_n_skills} skill weights add up to 100%."
    )

    # -- Step 4 table --
    st.markdown("##### Step 4 result: Calculate each skill's contribution")
    st.markdown(
        "We multiply each skill's weight by its AI overlap to get its contribution to the score."
    )
    st.markdown(
        "**Calculation performed:**\n"
        "- Contribution = Skill Weight × AI Overlap"
    )
    _step4 = _ex_data[["ability_name", "ability_weight", "exposure_sigmoid", "ability_contribution"]].copy()
    _step4["→ Contribution calc"] = _step4.apply(
        lambda r: f"{r['ability_weight']:.4f} × {r['exposure_sigmoid']:.2f} = {r['ability_contribution']:.4f}", axis=1
    )
    _step4_display = _step4[["ability_name", "ability_weight", "exposure_sigmoid", "→ Contribution calc"]].copy()
    _step4_display.columns = ["Skill", "Skill Weight", "AI Overlap", "→ Contribution"]
    st.dataframe(
        _step4_display.style.format({
            "Skill Weight": "{:.4f}",
            "AI Overlap": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Compare skills with similar weights but different AI overlap: for example, physical skills "
        "like Trunk Strength and Dynamic Strength have low AI overlap (below 0.20) so they contribute "
        "very little, while cognitive skills like Information Ordering have high AI overlap (0.87) "
        "and contribute much more. This is how physical skills pull the overall score down."
    )

    # -- Step 5 summary --
    _raw_score = _ex_data["ability_contribution"].sum()
    _final_score = _ex_data["aei_score"].iloc[0]
    _final_pct = _final_score * 100
    _band = _ex_data["risk_band"].iloc[0]
    _band_label = "Low" if _band == "Green" else ("Moderate" if _band == "Yellow" else "High")

    st.markdown("##### Step 5 result: Sum up and normalize the final score")
    st.markdown(
        "**Calculation performed:**\n"
        f"- Add up all {_n_skills} skill contributions → raw score\n"
        "- Normalize raw score across all 682 occupations (min-max) → final score\n"
        "- Assign risk band based on final score"
    )
    st.dataframe(
        pd.DataFrame({
            "Calculation": [
                f"Sum of all {_n_skills} skill contributions",
                "Normalized across 682 occupations (min-max)",
                "Risk band assignment",
            ],
            "Result": [
                f"Raw score = {_raw_score:.4f}",
                f"Final AI Exposure Score = {_final_pct:.0f}%",
                f"{_final_pct:.0f}% falls {'below 33%' if _band == 'Green' else ('between 33%–66%' if _band == 'Yellow' else 'above 66%')} → {_band_label} Exposure ({_band})",
            ],
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown(
        f"**What does this tell us?** Relay Technicians score **{_final_pct:.0f}% ({_band_label} Exposure)** — "
        "just above the low/moderate boundary. The relatively low score reflects that many of "
        "this role's key skills are physical or hands-on (like Trunk Strength, Manual Dexterity, "
        "Static Strength), where AI capability is limited. The skills that AI *can* handle well "
        "(like Information Ordering and Memorization) have moderate importance but are not the "
        "dominant part of the job."
    )

    st.markdown("---")

    # ---- Task Automation Methodology -----------------------------------------
    st.subheader("How task automation scores are calculated")
    st.markdown(
        "In addition to the overall score, each occupation's individual tasks are rated "
        "for automation potential. Here is how:"
    )

    st.markdown("**Where the inputs come from:**")
    st.markdown(
        "- **Average AI Overlap** — This is the mean of the AI Overlap values (the sigmoid-transformed "
        "AIOE ratings) across all of the occupation's skills. We already calculated AI Overlap for each "
        "skill in Step 2 of the scoring pipeline above. The average across all skills gives a single "
        "number representing how exposed the overall occupation is to AI. For Relay Technicians, "
        "the average AI Overlap across all 52 skills is **0.50**.\n"
        "- **Task Importance** — Each task listed in O*NET has an importance rating on a 1–5 scale, "
        "where 1 = not important and 5 = extremely important. This is separate from the *skill* importance "
        "used earlier — it rates the importance of *specific tasks* (e.g., \"Inspect equipment\") rather than "
        "*general abilities* (e.g., \"Written Comprehension\"). We normalize this to 0–1 by dividing by 5."
    )

    st.markdown("**The formula:**")
    st.markdown(
        "> **Task Automation Score = Average AI Overlap × (0.50 + 0.50 × Task Importance ÷ 5)**\n\n"
        "In plain English: tasks that are more important to the job get a slightly higher "
        "automation score, because automating important tasks has more impact. "
        "The 50/50 split ensures that even less-important tasks still receive a meaningful "
        "score based on the occupation's overall AI exposure."
    )

    st.markdown("**Tasks are then categorized:**")
    st.markdown(
        "| Automation score | Category | Meaning |\n"
        "|---|---|---|\n"
        "| Above 0.66 | Easily Automatable | AI could likely perform this task with minimal human involvement |\n"
        "| 0.33 – 0.66 | Partially Automatable | AI could assist but humans would still need to be involved |\n"
        "| Below 0.33 | Hard to Automate | This task requires abilities that AI currently handles poorly |"
    )

    st.info(
        "**Why task-level scores?** The overall AI Exposure Score tells you about the job as a whole, "
        "but jobs are made up of many individual tasks — and not all tasks are equally automatable. "
        "Task-level scores let you see *which specific parts* of a job are most affected by AI, "
        "which is more actionable for workforce planning. The formula weights more important tasks "
        "slightly higher because automating a task that takes up a large part of someone's day "
        "has more real-world impact than automating a rarely performed task."
    )

    # ---- Task Automation Worked Example --------------------------------------
    st.markdown("##### Worked example: Relay Technicians tasks")

    _task_data = pd.DataFrame()
    if TASK_AUTOMATION_FILE.exists():
        _all_tasks = pd.read_csv(TASK_AUTOMATION_FILE)
        _task_data = _all_tasks[_all_tasks["soc_code"].astype(str) == "49-2095"].copy()

    if not _task_data.empty:
        _task_data = _task_data.sort_values("automation_score", ascending=False).reset_index(drop=True)

        # Calculate the mean AI overlap for this occupation
        _relay_contrib = contributions[contributions["soc_code"].astype(str) == "49-2095"]
        _mean_ai_overlap = _relay_contrib["exposure_sigmoid"].mean()

        st.markdown(
            f"**Step 1:** Calculate the average AI Overlap across all {len(_relay_contrib)} skills "
            f"for Relay Technicians: **{_mean_ai_overlap:.4f}**\n\n"
            f"**Step 2:** For each task, look up its importance rating from O*NET (1–5 scale)\n\n"
            f"**Step 3:** Apply the formula: Automation Score = {_mean_ai_overlap:.4f} × (0.50 + 0.50 × Importance ÷ 5)"
        )

        # Build the table with inline calculations
        _task_display = _task_data[["task_description", "task_importance", "automation_score", "automation_level"]].copy()
        _task_display["→ Automation Score calc"] = _task_data.apply(
            lambda r: f"{_mean_ai_overlap:.4f} × (0.50 + 0.50 × {r['task_importance']:.2f} ÷ 5) = {r['automation_score']:.4f}", axis=1
        )

        _task_table = _task_display[["task_description", "task_importance", "→ Automation Score calc", "automation_level"]].copy()
        _task_table.columns = ["Task", "Importance (O*NET, 1–5)", "→ Automation Score calculation", "→ Category"]
        st.dataframe(
            _task_table.style.format({
                "Importance (O*NET, 1–5)": "{:.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Tasks with higher importance ratings get slightly higher automation scores because "
            "the formula scales them up. But all tasks share the same Average AI Overlap — "
            "so the occupation's overall AI exposure is the primary driver, "
            "while task importance provides a secondary adjustment."
        )
    else:
        st.caption("Task automation data not available for this example.")

    st.markdown("---")

    # Data sources
    st.subheader("Data sources")
    st.markdown(
        "**O*NET 30.2** — The Occupational Information Network, maintained by the U.S. Department of Labor. "
        "It provides detailed skill requirements for ~1,000 occupations, including how important each skill is "
        "to the job (importance rating, 1–7) and how advanced it needs to be (level rating, 1–7). "
        "This is the source for the Importance and Level values used in Steps 1–3 above."
    )
    st.markdown(
        "**AIOE (AI Occupational Exposure)** — A peer-reviewed research dataset that rates how well "
        "current AI systems can perform each human skill. Researchers evaluated AI capabilities across "
        "dozens of abilities (e.g., written comprehension, deductive reasoning, manual dexterity) and "
        "assigned each one an exposure score. Positive values mean AI performs the skill well; negative "
        "values mean AI performs poorly (common for physical skills). This is the source for the "
        "AI Exposure Rating used in Steps 1–2 above, which becomes the \"AI Overlap\" value after "
        "the sigmoid transformation."
    )
    st.markdown(
        "**How they work together:** O*NET tells us *what skills a job needs and how much it relies on them*. "
        "AIOE tells us *how good AI is at each of those skills*. By combining both, we can estimate "
        "how much of a job's skill profile overlaps with current AI capabilities."
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
