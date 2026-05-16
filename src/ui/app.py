"""Interactive demo for the Programme → Job alignment dataset.

Reads the latest export produced by ``src/export_results.py`` and renders the
top-10 hybrid matches for any chosen study programme.

Run with:

    .venv/bin/streamlit run src/ui/app.py
"""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import streamlit as st

from src.scraping.config import DATA_DIR

# ── Branding (Vilnius Tech / VGTU palette) ────────────────────────────────────
PRIMARY = "#0F2D52"      # institutional navy
ACCENT = "#C8102E"       # signature red
BG = "#F4F4F7"           # light background
CARD_BG = "#FFFFFF"
INK = "#1A1A1A"
MUTED = "#6B7280"
CHIP_BG = "#EEF1F6"

EXPORT_CSV = DATA_DIR.parent / "experiments" / "results" / "exports" / "programme_job_mapping.csv"


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_rankings() -> pd.DataFrame:
    df = pd.read_csv(EXPORT_CSV)
    df["programme_key"] = df["programme_name"] + " — " + df["institution"]
    return df


# ── Rendering helpers ─────────────────────────────────────────────────────────

def _score_bar(score: float, max_score: float) -> str:
    pct = 0 if max_score == 0 else (score / max_score) * 100
    return f"""
    <div class="score-track">
      <div class="score-fill" style="width:{pct:.1f}%;"></div>
    </div>
    """


def _gap_chips(raw: str) -> str:
    if not raw or pd.isna(raw):
        return ""
    chips = "".join(
        f'<span class="chip">{html.escape(g.strip())}</span>'
        for g in str(raw).split(";")
        if g.strip()
    )
    return f'<div class="chips-row"><span class="chips-label">Missing from programme</span>{chips}</div>'


def _meta_line(sector: str, location: str) -> str:
    parts = [p for p in (sector, location) if p and not pd.isna(p) and str(p).lower() != "nan"]
    if not parts:
        return ""
    return f'<div class="job-meta">{html.escape(" · ".join(parts))}</div>'


def render_card(row: pd.Series, max_score: float, delay_ms: int) -> str:
    title = html.escape(str(row["job_title"]))
    url = html.escape(str(row["job_url"]))
    rank = int(row["rank"])
    score = float(row["hybrid_score"])
    return f"""
    <div class="card" style="animation-delay:{delay_ms}ms;">
      <div class="rank-badge">#{rank}</div>
      <div class="card-body">
        <div class="job-title">{title}</div>
        {_meta_line(row.get("employer_sector", ""), row.get("location", ""))}
        <div class="score-row">
          {_score_bar(score, max_score)}
          <span class="score-val">{score:.3f}</span>
        </div>
        {_gap_chips(row.get("top_skill_gaps", ""))}
        <a class="job-link" href="{url}" target="_blank" rel="noopener">View job ↗</a>
      </div>
    </div>
    """


# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = f"""
<style>
.stApp {{ background: {BG}; }}
.block-container {{ padding-top: 2rem; max-width: 980px; }}

.hero {{
  border-left: 4px solid {ACCENT};
  padding: 0.4rem 1rem;
  margin-bottom: 1.5rem;
}}
.hero-eyebrow {{
  color: {MUTED}; font-size: 0.78rem; letter-spacing: 0.12em;
  text-transform: uppercase; margin-bottom: 0.25rem;
}}
.hero-title {{
  color: {PRIMARY}; font-weight: 700; font-size: 1.7rem;
  line-height: 1.15;
}}

.programme-header {{
  background: {PRIMARY}; color: white;
  padding: 1.1rem 1.4rem;
  border-radius: 10px;
  margin: 1.2rem 0 1.5rem 0;
  animation: fadeUp 0.4s ease-out both;
}}
.programme-header .ph-name {{ font-size: 1.15rem; font-weight: 600; }}
.programme-header .ph-inst {{ font-size: 0.85rem; opacity: 0.78; margin-top: 0.15rem; }}

.card {{
  background: {CARD_BG};
  border: 1px solid #E5E7EB;
  border-radius: 12px;
  padding: 1rem 1.2rem 1rem 1.2rem;
  margin-bottom: 0.85rem;
  display: flex; gap: 1rem;
  opacity: 0;
  animation: fadeUp 0.45s ease-out forwards;
  transition: transform 0.18s ease, box-shadow 0.18s ease;
}}
.card:hover {{
  transform: translateY(-2px);
  box-shadow: 0 6px 18px rgba(15, 45, 82, 0.10);
}}
.rank-badge {{
  flex: 0 0 44px; height: 44px;
  background: {PRIMARY}; color: white;
  border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
  font-weight: 700; font-size: 0.95rem;
}}
.card-body {{ flex: 1; min-width: 0; }}
.job-title {{
  color: {INK}; font-weight: 600; font-size: 1.02rem;
  line-height: 1.3; margin-bottom: 0.15rem;
}}
.job-meta {{
  color: {MUTED}; font-size: 0.82rem; margin-bottom: 0.55rem;
}}

.score-row {{
  display: flex; align-items: center; gap: 0.65rem;
  margin: 0.4rem 0 0.55rem 0;
}}
.score-track {{
  flex: 1; height: 7px; background: #ECEEF2; border-radius: 4px; overflow: hidden;
}}
.score-fill {{
  height: 100%;
  background: linear-gradient(90deg, {PRIMARY} 0%, {ACCENT} 100%);
  border-radius: 4px;
  animation: grow 0.6s ease-out both;
}}
.score-val {{
  color: {PRIMARY}; font-weight: 600; font-size: 0.85rem;
  font-variant-numeric: tabular-nums;
  min-width: 3rem; text-align: right;
}}

.chips-row {{
  display: flex; flex-wrap: wrap; gap: 0.35rem; align-items: center;
  margin: 0.3rem 0 0.5rem 0;
}}
.chips-label {{
  color: {MUTED}; font-size: 0.72rem; text-transform: uppercase;
  letter-spacing: 0.08em; margin-right: 0.3rem;
}}
.chip {{
  background: {CHIP_BG}; color: {PRIMARY};
  padding: 0.18rem 0.55rem; border-radius: 999px;
  font-size: 0.78rem; font-weight: 500;
}}

.job-link {{
  color: {ACCENT}; text-decoration: none; font-size: 0.85rem; font-weight: 600;
}}
.job-link:hover {{ text-decoration: underline; }}

@keyframes fadeUp {{
  from {{ opacity: 0; transform: translateY(8px); }}
  to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes grow {{
  from {{ width: 0; }}
}}

/* Tighten Streamlit's default widget label */
div[data-testid="stSelectbox"] label {{
  color: {PRIMARY}; font-weight: 600; font-size: 0.85rem;
  text-transform: uppercase; letter-spacing: 0.08em;
}}
</style>
"""


# ── App ───────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Programme → Job Alignment",
        page_icon="🎓",
        layout="centered",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-eyebrow">Master Thesis · Vilnius Tech</div>
          <div class="hero-title">Study Programme → Job Market Alignment</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = load_rankings()
    programmes = sorted(df["programme_key"].unique())

    choice = st.selectbox(
        "Choose a study programme",
        programmes,
        index=None,
        placeholder="Start typing or pick from the list…",
    )

    if not choice:
        st.markdown(
            f'<p style="color:{MUTED}; margin-top:1rem;">'
            f'Pick a programme above to see the top-10 hybrid matches from the job-ad corpus.'
            f'</p>',
            unsafe_allow_html=True,
        )
        return

    rows = df[df["programme_key"] == choice].sort_values("rank").head(10)
    if rows.empty:
        st.warning("No rankings available for this programme.")
        return

    programme_name = rows.iloc[0]["programme_name"]
    institution = rows.iloc[0]["institution"]
    max_score = float(rows["hybrid_score"].max())

    st.markdown(
        f"""
        <div class="programme-header">
          <div class="ph-name">{html.escape(str(programme_name))}</div>
          <div class="ph-inst">{html.escape(str(institution))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cards_html = "".join(
        render_card(row, max_score=max_score, delay_ms=60 * i)
        for i, (_, row) in enumerate(rows.iterrows())
    )
    st.markdown(cards_html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
