import math
import random
import numpy as np
import streamlit as st
from math import comb

st.set_page_config(page_title="Babuskha – Lost Cat Model")

st.title("🐱 Babuskha – Lost Cat Probability Explorer")

st.markdown(
    """
This app estimates the probability that a lost cat meets someone the owner knows,
using either an **analytic (hypergeometric)** model or a **Monte Carlo simulation**.
"""
)

# -----------------------------
# Utility functions
# -----------------------------

def hypergeometric_p_k(H, F, N, k):
    """Probability of exactly k known houses visited."""
    if k > F or k > N:
        return 0
    return comb(F, k) * comb(H - F, N - k) / comb(H, N)


def analytic_prob_at_least_one(H, F, N, q=1.0):
    """
    Probability of at least one engagement with engagement probability q.
    q = 1 reduces to the simpler no-engagement model.
    """
    max_k = min(F, N)
    p_no_engagement = 0.0

    for k in range(0, max_k + 1):
        p_k = hypergeometric_p_k(H, F, N, k)
        p_no_engagement += ((1 - q) ** k) * p_k

    return 1 - p_no_engagement


def monte_carlo_prob(H, F, N, q=1.0, sims=5000):
    """Monte Carlo estimate."""
    successes = 0

    for _ in range(sims):
        # Randomly choose which houses are known
        known = set(random.sample(range(H), F))

        # Sample N distinct visited houses
        visited = set(random.sample(range(H), N))

        # Known houses actually visited
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
# Sidebar controls
# -----------------------------

st.sidebar.header("Model Inputs")

H = st.sidebar.number_input("Total houses in area (H)", min_value=1, value=2000)
F = st.sidebar.number_input("Known houses (F)", min_value=0, value=20)
N = st.sidebar.number_input("Distinct houses visited (N)", min_value=1, value=3)

q = st.sidebar.slider(
    "Engagement probability per known-house visit (q)",
    min_value=0.0, max_value=1.0, value=1.0, step=0.05
)

mode = st.sidebar.radio("Model", ["Analytic", "Monte Carlo"])
sims = 5000

if mode == "Monte Carlo":
    sims = st.sidebar.slider("Simulation runs", 1000, 20000, 5000, step=1000)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: start simple, then explore sensitivity.")


# -----------------------------
# Run model
# -----------------------------

st.subheader("Results")

if mode == "Analytic":
    p = analytic_prob_at_least_one(H, F, N, q)
    st.metric("Probability (analytic)", f"{p*100:.2f}%")
    st.caption("Computed using the hypergeometric model.")

else:
    p_mc = monte_carlo_prob(H, F, N, q, sims)
    st.metric("Probability (Monte Carlo)", f"{p_mc*100:.2f}%")
    st.caption(f"Estimated from {sims:,} simulated scenarios.")

# -----------------------------
# P plot
# -----------------------------

import pandas as pd
import altair as alt
import numpy as np

st.subheader("Probability vs. Number of Houses Visited")

# -------------------------
# Monte Carlo curve toggle
# -------------------------
if "show_mc_curve" not in st.session_state:
    st.session_state["show_mc_curve"] = False

if mode == "Monte Carlo":
    if st.button("Compute Monte Carlo curve (slower)"):
        st.session_state["show_mc_curve"] = not st.session_state["show_mc_curve"]
else:
    # Leaving MC mode → disable curve + show hint
    st.session_state["show_mc_curve"] = False
    st.caption("Select Monte Carlo mode on the left to enable the Monte Carlo curve.")

# -------------------------
# Build analytic curve
# -------------------------
max_N = min(H, 2000)

num_points = 200  # subsample for speed
Ns = np.linspace(1, max_N, num_points, dtype=int)
Ns = sorted(set(Ns))

analytic_probs = [
    analytic_prob_at_least_one(H, F, N_i, q)
    for N_i in Ns
]

df_plot = pd.DataFrame({
    "N": Ns,
    "Model": ["Analytic"] * len(Ns),
    "Probability": analytic_probs,
})

# Base analytic line
chart = (
    alt.Chart(df_plot)
    .mark_line()
    .encode(
        x=alt.X("N", title="N (houses visited)"),
        y=alt.Y("Probability", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Model", title="Model"),
        tooltip=[
            alt.Tooltip("N", title="N (houses visited)"),
            "Model",
            alt.Tooltip("Probability", format=".3f"),
        ],
    )
)

# -------------------------
# Optional Monte Carlo curve
# -------------------------
if st.session_state["show_mc_curve"] and mode == "Monte Carlo":
    num_points_mc = 60  # fewer points for performance
    Ns_mc = np.linspace(1, max_N, num_points_mc, dtype=int)
    Ns_mc = sorted(set(Ns_mc))

    sims_curve = min(1000, sims)  # cap sims per point

    mc_probs = [
        monte_carlo_prob(H, F, N_i, q, sims_curve)
        for N_i in Ns_mc
    ]

    df_mc = pd.DataFrame({
        "N": Ns_mc,
        "Model": ["Monte Carlo"] * len(Ns_mc),
        "Probability": mc_probs,
    })

    df_all = pd.concat([df_plot, df_mc], ignore_index=True)

    chart = (
        alt.Chart(df_all)
        .mark_line()
        .encode(
            x=alt.X("N", title="N (houses visited)"),
            y=alt.Y("Probability", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Model", title="Model"),
            tooltip=[
                alt.Tooltip("N", title="N (houses visited)"),
                "Model",
                alt.Tooltip("Probability", format=".3f"),
            ],
        )
    )

st.altair_chart(chart.interactive(), use_container_width=True)

st.caption(
    "Analytic curve is always shown. "
    "Click the button (in Monte Carlo mode) to add a simulated comparison curve."
)


# -----------------------------
# Explanation panel
# -----------------------------

st.subheader("Model Explanation")

tab1, tab2 = st.tabs(["🧮 Analytic model", "🎲 Monte Carlo model"])

with tab1:
    st.markdown(
        """
### Analytic (Hypergeometric) Model

**Intuition:**  
The cat visits \(N\) distinct houses out of \(H\) total.  
\(F\) of those houses belong to people the owner knows.  
We treat this like **drawing houses without replacement**.

- \(H\) — total houses  
- \(F\) — known houses  
- \(N\) — distinct houses visited  
- \(q\) — probability of engagement when a known house is visited
"""
    )

    st.markdown("**No-engagement probability:**")
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

    st.markdown("**Probability of meeting ≥ 1 known person:**")
    st.latex(
r"""
P(\text{meet ≥ 1 known person})
=
1 - P(\text{no engagement})
"""
    )

    st.markdown(
        """
**In words:**  
The only way the cat *doesn’t* meet someone they know  
is if every house visited is unknown, or engagement fails every time —  
so we compute that case and subtract from 1.
"""
    )


with tab2:
    st.markdown(
        """
### Monte Carlo (Simulation) Model

**Intuition:**  
Instead of solving the formula directly, we **simulate many missing-cat scenarios** and count how often a meeting happens.

For each simulation:

1. Randomly choose which houses are *known houses*  
2. Randomly choose the \(N\) houses the cat visits  
3. For each known house visited, engagement happens with probability \(q\)  
4. Mark success if engagement happens at least once

Repeat thousands of times →  
the fraction of successes ≈ the probability.

**Why it’s useful**

- Easy to add realism (distance-decay, spatial wandering, clustering)
- Lets us compare outcomes with the analytic model
- Great for experimentation and visualisation
"""
    )
