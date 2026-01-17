import math
import random
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from math import comb

from network import build_house_road_network
from viz import plot_network_3d  # (or plot_network_2d if you add it)

st.set_page_config(page_title="Find your Cat Model", layout="wide")
st.title("🐱 Find Cat Model")

st.markdown(
    """
This app estimates the probability that a cat meets at least one person her owner knows.
Choose **Network** for an OpenStreetMap-based neighbourhood graph, or **Statistical** for the simpler distributions.
"""
)

# -----------------------------
# Utility functions (statistical)
# -----------------------------
def binomial_prob_at_least_one(H, F, N, q=1.0):
    if H <= 0 or N <= 0 or F <= 0 or q <= 0:
        return 0.0
    p_engage = q * (F / H)
    p_engage = max(0.0, min(1.0, p_engage))
    return 1 - (1 - p_engage) ** N


def hypergeometric_p_k(H, F, N, k):
    if k > F or k > N:
        return 0.0
    return comb(F, k) * comb(H - F, N - k) / comb(H, N)


def analytic_prob_at_least_one(H, F, N, q=1.0):
    max_k = min(F, N)
    p_no_engagement = 0.0
    for k in range(0, max_k + 1):
        p_k = hypergeometric_p_k(H, F, N, k)
        p_no_engagement += ((1 - q) ** k) * p_k
    return 1 - p_no_engagement


def monte_carlo_prob_weighted(
    H, F, N, q=1.0, sims=5000,
    max_radius=1.0,
    alpha=2.0
):
    successes = 0

    radii = np.sqrt(np.random.rand(H)) * max_radius
    weights = np.exp(-alpha * radii)
    p_visit = weights / weights.sum()

    for _ in range(sims):
        known = set(random.sample(range(H), F))
        visited_idx = np.random.choice(H, size=N, replace=False, p=p_visit)
        visited = set(visited_idx)
        visited_known = visited & known

        engaged = False
        for _ in visited_known:
            if random.random() < q:
                engaged = True
                break
        if engaged:
            successes += 1

    return successes / sims


# -----------------------------
# TOP-LEVEL APP MODE (2 tabs)
# -----------------------------
tab_network_mode, tab_stat_mode = st.tabs(["🗺️ Network", "📊 Statistical"])

# -----------------------------
# SIDEBAR: show controls depending on which mode user is in
# -----------------------------
# Streamlit note: both tabs are rendered, so sidebar will show whichever branch runs.
# We use a sidebar toggle to choose which sidebar panel is active.
# (This avoids confusing mixed sidebars across tabs.)
st.sidebar.header("App Mode")
app_mode = st.sidebar.radio("Choose mode", ["Network", "Statistical"], index=0)

st.sidebar.markdown("---")

# -----------------------------
# NETWORK MODE
# -----------------------------
if app_mode == "Network":
    st.sidebar.header("Network Inputs")

    with st.sidebar.form("net_params"):
        place = st.text_input("Home area (postcode / place)", value="N1 2QF, UK")
        radius_m = st.slider("Radius (meters)", 200, 3000, 1609, step=50)
        friend_pct = st.slider("Friend houses (%)", 0, 100, 10, step=1)
        show_friends = st.checkbox("Highlight friend houses", value=True)





        show_network = st.checkbox("Show network", value=True)

        apply = st.form_submit_button("Rebuild network")

    st.sidebar.markdown("---")
    st.sidebar.caption("Network is constructed from OpenStreetMap. First load can take a moment.")

    # Clear cache only when user clicks Apply
    if apply:
        st.cache_resource.clear()
        st.rerun()

 

    @st.cache_resource
    def load_network(_place: str, _radius_m: int, _friend_pct: int):
        return build_house_road_network(place=_place, radius_m=_radius_m, friend_pct=_friend_pct)

    G = load_network(place, int(radius_m), int(friend_pct))

    
    with tab_network_mode:
        st.subheader("Neighbourhood network")

        if show_network:
            with st.spinner("Building network from OpenStreetMap..."):
                G = load_network(place, int(radius_m),int(friend_pct))

            fig = plot_network_3d(G, show_friends=show_friends)

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Enable **Show network** in the sidebar to render the graph.")

        st.caption("Tip: drag to rotate, scroll to zoom, right-drag (or two-finger drag) to pan.")

        # --- quick network summary ---
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()

        house_nodes = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "house")
        pub_nodes = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "pub")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Nodes", f"{num_nodes:,}")
        c2.metric("Edges", f"{num_edges:,}")
        c3.metric("Houses", f"{house_nodes:,}")
        c4.metric("Pubs", f"{pub_nodes:,}")

# -----------------------------
# STATISTICAL MODE
# -----------------------------
else:
    st.sidebar.header("Statistical Inputs")

    H = st.sidebar.number_input("Total houses in area (H)", min_value=1, value=2000)
    F = st.sidebar.number_input("Known houses (F)", min_value=0, value=20)
    N = st.sidebar.number_input("Visits / houses visited (N)", min_value=1, value=3)

    q = st.sidebar.slider(
        "Engagement probability per known-house visit (q)",
        min_value=0.0, max_value=1.0, value=1.0, step=0.05
    )

    mode = st.sidebar.radio("Model", ["Binomial", "Hypergeometric", "Monte Carlo"])

    sims = 5000
    max_radius = 1.0
    alpha = 2.0
    if mode == "Monte Carlo":
        sims = st.sidebar.slider("Simulation runs", 1000, 20000, 5000, step=1000)
        max_radius = st.sidebar.slider("Max distance from home (synthetic, miles)", 0.1, 2.0, 1.0, step=0.1)
        alpha = st.sidebar.slider("Decay rate (alpha)", 0.1, 5.0, 2.0, step=0.1)

    st.sidebar.markdown("---")

    # Tabs inside Statistical
    with tab_stat_mode:
        tab_results, tab_plot, tab_explain = st.tabs(["Results", "Probability plot", "Model explanation"])

        # ---------- Results ----------
        with tab_results:
            st.subheader("Results")

            if mode == "Hypergeometric":
                p = analytic_prob_at_least_one(H, F, N, q)
                st.metric("Probability (hypergeometric)", f"{p*100:.2f}%")
                st.caption("House-based model (distinct houses).")

            elif mode == "Binomial":
                p = binomial_prob_at_least_one(H, F, N, q)
                st.metric("Probability (binomial)", f"{p*100:.2f}%")
                st.caption("Visit-based model (independent visits).")

            else:
                p_mc = monte_carlo_prob_weighted(H, F, N, q, sims, max_radius, alpha)
                st.metric("Probability (Monte Carlo)", f"{p_mc*100:.2f}%")
                st.caption(f"Estimated from {sims:,} simulations (distance-weighted synthetic visits).")

        # ---------- Plot ----------
        with tab_plot:
            st.subheader("Probability vs. N")

            # analytic curve uses the selected analytic model; if in MC mode, we show hypergeometric as baseline
            def analytic_for_curve(H_, F_, N_i, q_, model_name):
                if model_name == "Hypergeometric":
                    return analytic_prob_at_least_one(H_, F_, N_i, q_)
                if model_name == "Binomial":
                    return binomial_prob_at_least_one(H_, F_, N_i, q_)
                # baseline in Monte Carlo mode
                return analytic_prob_at_least_one(H_, F_, N_i, q_)

            if "show_mc_curve" not in st.session_state:
                st.session_state["show_mc_curve"] = False

            if mode == "Monte Carlo":
                if st.button("Toggle Monte Carlo curve (slower)"):
                    st.session_state["show_mc_curve"] = not st.session_state["show_mc_curve"]
            else:
                st.session_state["show_mc_curve"] = False

            max_N = min(H, 2000)
            Ns = np.linspace(1, max_N, 200, dtype=int)
            Ns = sorted(set(Ns))

            analytic_probs = [analytic_for_curve(H, F, n_i, q, mode) for n_i in Ns]
            base_label = mode if mode != "Monte Carlo" else "Hypergeometric (baseline)"

            df_plot = pd.DataFrame({"N": Ns, "Model": [base_label]*len(Ns), "Probability": analytic_probs})

            chart = (
                alt.Chart(df_plot)
                .mark_line()
                .encode(
                    x=alt.X("N", title="N (visits / houses visited)"),
                    y=alt.Y("Probability", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("Model", title="Model"),
                    tooltip=[alt.Tooltip("N"), alt.Tooltip("Model"), alt.Tooltip("Probability", format=".3f")],
                )
            )

            if st.session_state["show_mc_curve"] and mode == "Monte Carlo":
                Ns_mc = np.linspace(1, max_N, 60, dtype=int)
                Ns_mc = sorted(set(Ns_mc))
                sims_curve = min(1000, sims)

                mc_probs = [
                    monte_carlo_prob_weighted(H, F, n_i, q, sims_curve, max_radius, alpha)
                    for n_i in Ns_mc
                ]
                df_mc = pd.DataFrame({"N": Ns_mc, "Model": ["Monte Carlo"]*len(Ns_mc), "Probability": mc_probs})

                df_all = pd.concat([df_plot, df_mc], ignore_index=True)

                chart = (
                    alt.Chart(df_all)
                    .mark_line()
                    .encode(
                        x=alt.X("N", title="N (visits / houses visited)"),
                        y=alt.Y("Probability", scale=alt.Scale(domain=[0, 1])),
                        color=alt.Color("Model", title="Model"),
                        tooltip=[alt.Tooltip("N"), alt.Tooltip("Model"), alt.Tooltip("Probability", format=".3f")],
                    )
                )

            st.altair_chart(chart.interactive(), use_container_width=True)

        # ---------- Explanation ----------
        with tab_explain:
            st.subheader("Model explanation")

            e1, e2, e3 = st.tabs(["📍 Binomial", "🧮 Hypergeometric", "🎲 Monte Carlo"])

            with e1:
                st.markdown(
                    """
**Why binomial?** Each visit is a trial with two outcomes: engagement (success) or not (failure).
Across **N** visits there are many ways successes can occur, and the binomial model accounts for all of them.
We usually care about the probability of **at least one** success.
"""
                )
                st.latex(r"p_{\text{engage}} = q\frac{F}{H}")
                st.latex(r"P(\ge 1) = 1-(1-p_{\text{engage}})^N")

            with e2:
                st.markdown("House-based: cat visits **N distinct houses** out of **H**, with **F** known houses.")
                st.latex(
                    r"""
P(\text{no engagement})
=
\sum_{k=0}^{\min(F,N)}
(1-q)^k\;
\frac{\binom{F}{k}\binom{H-F}{N-k}}
     {\binom{H}{N}}
"""
                )
                st.latex(r"P(\ge 1)=1-P(\text{no engagement})")

            with e3:
                st.markdown(
                    """
Monte Carlo simulates many “lost cat” scenarios and estimates the probability empirically.
In the weighted version, visit probabilities decay with distance using an exponential weight.
"""
                )
                st.latex(
                    r"""
\hat{p} \approx
\frac{\text{# simulations with at least one engagement}}{\text{# simulations}}
"""
                )

    # In Statistical mode, keep Network tab minimal
    with tab_network_mode:
        st.info("Switch to **Network** in the sidebar to explore the OpenStreetMap-based neighbourhood graph.")
