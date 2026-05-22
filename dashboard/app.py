"""Concise Immunonet dashboard: models grouped by supervised / deep / DA / unsupervised."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

DASHBOARD_DIR = Path(__file__).resolve().parent
REPO_ROOT = DASHBOARD_DIR.parent
DATA_PATH = DASHBOARD_DIR / "experiment_data.json"
BASELINE_CSV = REPO_ROOT / "experiments" / "step0_baselines" / "results.csv"
V5_JSON = REPO_ROOT / "experiments" / "step3_self_attention" / "results" / "gene_attention_v5_results.json"
V5_CM_COMBINED = (
    REPO_ROOT / "experiments" / "step3_self_attention" / "results" / "confusion_matrices_v5.png"
)

FAMILIES = [
    ("supervised_classical", "Supervised (classical)", "LogReg, RF, SVM on labelled source"),
    ("deep_supervised", "Deep learning (supervised)", "GeneAttention, patch transformer + CE"),
    ("domain_adaptation", "Domain adaptation", "MMD, DANN, CDAN, joint VAE+alignment"),
    ("unsupervised_generative", "Unsupervised / generative", "VAE pretraining or recon-centric paths"),
]

FAMILY_COLORS = {
    "supervised_classical": "#5c8cbc",
    "deep_supervised": "#2e8b57",
    "domain_adaptation": "#d97a3e",
    "unsupervised_generative": "#9b59b6",
    "other": "#7f8c8d",
}


@st.cache_data
def load_data():
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def build_df(raw: dict) -> pd.DataFrame:
    df = pd.DataFrame(raw["model_comparison"])
    if "family" not in df.columns:
        df["family"] = "other"
    if BASELINE_CSV.exists():
        try:
            bl = pd.read_csv(BASELINE_CSV)
            for _, row in bl.iterrows():
                name = {"LogisticRegression": "LogReg", "RandomForest": "RF"}.get(
                    row["model"], row["model"]
                )
                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            [
                                {
                                    "model": f"{name} (CSV) [{row['domain']}]",
                                    "family": "supervised_classical",
                                    "type": "Baseline live",
                                    "source_f1": row["f1_macro"] if row["domain"] == "source" else np.nan,
                                    "target_f1": row["f1_macro"] if row["domain"] == "target" else np.nan,
                                    "notebook_or_script": str(BASELINE_CSV),
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )
        except Exception:
            pass
    fam_order = {k: i for i, (k, _, _) in enumerate(FAMILIES)}
    df["_fam_rank"] = df["family"].map(lambda x: fam_order.get(x, 99))
    return df.sort_values(["_fam_rank", "model"]).drop(columns=["_fam_rank"])


def load_v5(base: dict, proj: dict):
    v5 = json.loads(json.dumps(base["gene_attention_v5_eval"]))
    disk = None
    if V5_JSON.exists():
        try:
            disk = json.loads(V5_JSON.read_text(encoding="utf-8"))
        except Exception:
            disk = None
    if disk and "source_test" in disk:
        try:
            v5["source_macro_f1"] = disk["source_test"].get("macro_f1", v5["source_macro_f1"])
            v5["target_macro_f1"] = disk["target_eval"].get("macro_f1", v5["target_macro_f1"])
            src_pc = disk["source_test"].get("per_class_f1") or {}
            tgt_pc = disk["target_eval"].get("per_class_f1") or {}
            for c in proj["classes"]:
                if c in v5["per_class"] and c in src_pc:
                    v5["per_class"][c]["f1_src"] = src_pc[c]
                if c in v5["per_class"] and c in tgt_pc:
                    v5["per_class"][c]["f1_tgt"] = tgt_pc[c]
        except (KeyError, TypeError):
            pass
    return v5, disk


def fig_faceted_target_f1(df: pd.DataFrame) -> go.Figure:
    sub = df.dropna(subset=["target_f1"]).copy()
    if sub.empty:
        return go.Figure().update_layout(title="No target F1 values")
    labels = {k: v for k, v, _ in FAMILIES}
    sub["group"] = sub["family"].map(labels).fillna("Other")
    order = [v for _, v, _ in FAMILIES] + (["Other"] if (sub["group"] == "Other").any() else [])
    cat_order = [x for x in order if x in set(sub["group"])]
    sub["group"] = pd.Categorical(sub["group"], categories=cat_order, ordered=True)
    sub = sub.sort_values(["group", "target_f1"])
    fig = px.bar(
        sub,
        x="target_f1",
        y="model",
        facet_row="group",
        color="family",
        color_discrete_map=FAMILY_COLORS,
        orientation="h",
        height=max(480, 110 * sub["group"].nunique() + len(sub) * 22),
        category_orders={"group": cat_order},
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(
        title="Target macro-F1 by model family (GSE126030)",
        showlegend=False,
        margin=dict(l=80, r=20, t=60, b=40),
    )
    fig.update_xaxes(title="Macro-F1", range=[0, max(0.55, sub["target_f1"].max() * 1.05)])
    return fig


def fig_grouped_metrics(df: pd.DataFrame) -> go.Figure:
    sub = df.dropna(subset=["source_f1", "target_f1"], how="all").copy()
    sub = sub[sub["source_f1"].notna() | sub["target_f1"].notna()]
    rows = []
    for _, r in sub.iterrows():
        if pd.notna(r["source_f1"]):
            rows.append({"model": r["model"], "family": r["family"], "split": "Source F1", "value": r["source_f1"]})
        if pd.notna(r["target_f1"]):
            rows.append({"model": r["model"], "family": r["family"], "split": "Target F1", "value": r["target_f1"]})
    if not rows:
        return go.Figure().update_layout(title="No F1 metrics")
    long = pd.DataFrame(rows)
    labels = {k: v for k, v, _ in FAMILIES}
    long["group"] = long["family"].map(labels).fillna("Other")
    group_order = [v for _, v, _ in FAMILIES] + (
        ["Other"] if (long["group"] == "Other").any() else []
    )
    cat_order = [x for x in group_order if x in set(long["group"])]
    long["group"] = pd.Categorical(long["group"], categories=cat_order, ordered=True)
    fig = px.bar(
        long,
        x="value",
        y="model",
        color="split",
        facet_row="group",
        barmode="group",
        orientation="h",
        color_discrete_sequence=["#3498db", "#e74c3c"],
        height=max(520, len(long) * 18),
        category_orders={"group": cat_order},
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(
        title="Source vs target macro-F1 (rows grouped by family)",
        legend_title_text="",
        margin=dict(l=80, r=20, t=50, b=40),
    )
    fig.update_xaxes(title="Macro-F1", range=[0, 1])
    return fig


def fig_family_heatmap(df: pd.DataFrame) -> go.Figure:
    sub = df.dropna(subset=["source_f1", "target_f1"]).copy()
    if sub.empty:
        return go.Figure().update_layout(title="Need both F1 columns")
    fam_order = [k for k, _, _ in FAMILIES]
    sub["_o"] = sub["family"].map({k: i for i, k in enumerate(fam_order)}).fillna(99)
    sub = sub.sort_values(["_o", "model"])
    z = sub[["source_f1", "target_f1"]].values.T
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=sub["model"].tolist(),
            y=["Source", "Target"],
            zmin=0,
            zmax=1,
            colorscale="YlGnBu",
            text=np.round(z, 3).astype(str),
            texttemplate="%{text}",
        )
    )
    fig.update_layout(
        title="Heatmap: models with both domain F1s (table order = family blocks)",
        xaxis_tickangle=-50,
        height=380,
        margin=dict(b=200),
    )
    return fig


def fig_v5_compact(v5: dict) -> tuple[go.Figure, go.Figure]:
    pc = v5["per_class"]
    classes = list(pc.keys())
    z = [[pc[c]["f1_src"] for c in classes], [pc[c]["f1_tgt"] for c in classes]]
    h1 = go.Figure(
        data=go.Heatmap(
            z=z,
            x=classes,
            y=["Src F1", "Tgt F1"],
            zmin=0,
            zmax=1,
            colorscale="RdYlGn",
            text=[[f'{pc[c]["f1_src"]:.2f}' for c in classes], [f'{pc[c]["f1_tgt"]:.2f}' for c in classes]],
            texttemplate="%{text}",
        )
    )
    h1.update_layout(title="GeneAtt v5 per-class F1", height=320, xaxis_tickangle=-30)
    key_p, key_r = "precision_tgt", "recall_tgt"
    arr = np.array([[pc[c][key_p], pc[c][key_r]] for c in classes]).T
    h2 = go.Figure(
        data=go.Heatmap(
            z=arr,
            x=classes,
            y=["Precision", "Recall"],
            zmin=0,
            zmax=1,
            colorscale="Blues",
            text=np.round(arr, 2).astype(str),
            texttemplate="%{text}",
        )
    )
    h2.update_layout(title="GeneAtt v5 target: precision & recall", height=300, xaxis_tickangle=-30)
    return h1, h2


def main():
    st.set_page_config(page_title="Immunonet", layout="wide")
    raw = load_data()
    proj = raw["project"]
    df = build_df(raw)
    v5, v5_disk = load_v5(raw, proj)

    st.title("Immunonet results")
    st.caption(
        f"{proj['task']} · Best: **{proj['best_model']}** (target macro-F1 ≈ {proj['best_target_macro_f1']})"
    )

    tabs = st.tabs(
        ["By family", "GeneAtt v5", "Paths"]
    )

    with tabs[0]:
        st.markdown("**Families:** classical supervised · deep (supervised) · domain adaptation · unsupervised/generative.")

        c1, c2, c3 = st.columns(3)
        tgt = df.dropna(subset=["target_f1"])
        c1.metric("Best target F1", f"{tgt['target_f1'].max():.3f}" if len(tgt) else "—")
        both = df.dropna(subset=["source_f1", "target_f1"])
        c2.metric("Models (both domains)", len(both))
        c3.metric("Classes", len(proj["classes"]))

        st.plotly_chart(fig_faceted_target_f1(df), use_container_width=True)
        st.plotly_chart(fig_grouped_metrics(df), use_container_width=True)
        st.plotly_chart(fig_family_heatmap(df), use_container_width=True)

        for key, title, blurb in FAMILIES:
            part = df[df["family"] == key]
            if part.empty:
                continue
            with st.expander(f"{title} — {blurb}", expanded=(key == "deep_supervised")):
                show = part[
                    ["model", "source_f1", "target_f1", "type"]
                    + (["notes"] if "notes" in part.columns else [])
                ].copy()
                st.dataframe(show, hide_index=True, use_container_width=True)

    with tabs[1]:
        st.write(
            f"Macro-F1: **{v5['source_macro_f1']:.3f}** source · **{v5['target_macro_f1']:.3f}** target"
        )
        if v5_disk:
            st.caption(f"Merged `{V5_JSON.name}`")
        h1, h2 = fig_v5_compact(v5)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(h1, use_container_width=True)
        with c2:
            st.plotly_chart(h2, use_container_width=True)
        if V5_CM_COMBINED.exists():
            st.image(str(V5_CM_COMBINED), use_container_width=True)

    with tabs[2]:
        st.caption("Notebook paths under `experiments/`")
        for ex in raw["experiments_index"]:
            st.markdown(f"**{ex['folder']}** — {ex['title']}")
            for fn in ex["files"]:
                st.code(str(REPO_ROOT / "experiments" / ex["folder"] / fn), language="text")

    with st.sidebar:
        st.markdown("**Data** `experiment_data.json` · optional `results.csv` / `gene_attention_v5_results.json`")
        prog = pd.DataFrame(raw["gene_progression_timeline"])
        st.markdown("**GeneAtt → target F1**")
        st.dataframe(prog, hide_index=True)


if __name__ == "__main__":
    main()
