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

def binomial_prob_at_least_one(H, F, N, q=1.0):
    """
    Visit-based model (binomial approximation).
    Each visit is a trial:
      p_engage = q * (F/H)
    Probability of at least one engagement over N visits.
    """
    if H <= 0 or N <= 0 or F <= 0 or q <= 0:
        return 0.0

    p_engage = q * (F / H)
    # Clamp just in case of weird inputs
    p_engage = max(0.0, min(1.0, p_engage))

    return 1 - (1 - p_engage) ** N


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


mode = st.sidebar.radio(
    "Model",
    ["Hypergeometric (house-based)", "Binomial (visit-based)", "Monte Carlo"]
)

sims = 5000

if mode == "Monte Carlo":
    sims = st.sidebar.slider("Simulation runs", 1000, 20000, 5000, step=1000)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: start simple, then explore sensitivity.")


# -----------------------------
# Run model
# -----------------------------

st.subheader("Results")

if mode == "Hypergeometric (house-based)":
    p = analytic_prob_at_least_one(H, F, N, q)
    st.metric("Probability (hypergeometric)", f"{p*100:.2f}%")
    st.caption("House-based model using the hypergeometric distribution (distinct houses visited).")

elif mode == "Binomial (visit-based)":
    p = binomial_prob_at_least_one(H, F, N, q)
    st.metric("Probability (binomial)", f"{p*100:.2f}%")
    st.caption("Visit-based model using the binomial distribution (visits treated as independent trials).")

else:  # Monte Carlo
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
# Helper: pick analytic model for curve
# -------------------------
def analytic_for_curve(H, F, N_i, q, mode):
    if mode == "Hypergeometric (house-based)":
        return analytic_prob_at_least_one(H, F, N_i, q)
    elif mode == "Binomial (visit-based)":
        return binomial_prob_at_least_one(H, F, N_i, q)
    else:
        # If in Monte Carlo mode, default to hypergeometric for the line
        return analytic_prob_at_least_one(H, F, N_i, q)

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
# Build analytic curve (hypergeometric or binomial)
# -------------------------
max_N = min(H, 2000)

num_points = 200  # subsample for speed
Ns = np.linspace(1, max_N, num_points, dtype=int)
Ns = sorted(set(Ns))

analytic_probs = [
    analytic_for_curve(H, F, N_i, q, mode)
    for N_i in Ns
]

df_plot = pd.DataFrame({
    "N": Ns,
    "Model": [mode if mode != "Monte Carlo" else "Hypergeometric (house-based)"] * len(Ns),
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
    "The curve shows the selected analytic model (hypergeometric or binomial). "
    "In Monte Carlo mode, you can optionally add a simulated comparison curve."
)

# -----------------------------
# Explanation panel
# -----------------------------

st.subheader("Model explanation")

tab1, tab2, tab3 = st.tabs([
    "📍 Binomial (visit-based)",
    "🧮 Hypergeometric (house-based)",
    "🎲 Monte Carlo"
])

# -------------------------------------------------------------------
# TAB 1 — BINOMIAL (VISIT-BASED)
# -------------------------------------------------------------------
with tab1:
    st.markdown(
        """
### 📍 Binomial (Visit-Based) Model

**Idea (what this model assumes):**  
We treat each **visit** as a separate opportunity for engagement.  
The cat may revisit the same house multiple times — every arrival is another chance.

- **H** — total houses  
- **F** — houses belonging to people the owner knows  
- **N** — number of visits the cat makes  
- **q** — probability of engagement when a known house is visited
"""
    )

    st.markdown("**Probability a visit is to a known house:**")
    st.latex(r"p_{\text{known}} = \frac{F}{H}")

    st.markdown("**Probability a visit results in engagement:**")
    st.latex(r"p_{\text{engage}} = q \cdot p_{\text{known}} = q \cdot \frac{F}{H}")

    st.markdown("#### Binomial distribution")

    st.markdown("Let \(X\) be the number of visits that result in an engagement.")
    st.latex(r"X \sim \text{Binomial}(N,\; p_{\text{engage}})")

    st.markdown("The probability of **exactly \(k\) engagements** is:")
    st.latex(
r"""
P(X = k)
=
\binom{N}{k}
\left(p_{\text{engage}}\right)^k
\left(1-p_{\text{engage}}\right)^{N-k}
"""
    )

    st.markdown("Probability of **no engagement on any visit**:")
    st.latex(r"P(X = 0) = (1 - p_{\text{engage}})^N")

    st.markdown("Therefore, the probability of **at least one engagement** is:")
    st.latex(
r"""
P(\text{at least one engagement})
=
1 - (1 - p_{\text{engage}})^N
=
1 - \left(1 - q\frac{F}{H}\right)^N
"""
    )

    st.markdown(
        """
### 🐈 Intuition

Every visit is like flipping a weighted coin:

- Unknown house → nothing happens  
- Known house → engagement happens with probability **q**  
- As soon as **one** visit succeeds, we count it as success overall

This model fits situations where we care about **opportunities across visits**, including revisits.
"""
    )

# -------------------------------------------------------------------
# TAB 2 — HYPERGEOMETRIC (HOUSE-BASED)
# -------------------------------------------------------------------
with tab2:
    st.markdown(
        """
### 🧮 Hypergeometric (House-Based) Model

**Idea (what this model assumes):**  
We think in terms of **distinct houses**, not visits.  
What matters is which **unique houses** the cat ends up in.

- **H** — total houses  
- **F** — houses belonging to people the owner knows  
- **N** — number of **distinct** houses visited  
- **q** — probability of engagement when a known house is visited
"""
    )

    st.markdown("Let \(K\) be the number of known houses among the \(N\) visited houses.")
    st.markdown("**Hypergeometric probability of visiting exactly \(k\) known houses:**")
    st.latex(
r"""
P(K = k)
=
\frac{
\binom{F}{k}\,
\binom{H-F}{N-k}
}{
\binom{H}{N}
}
"""
    )

    st.markdown(
        """
If the cat visits \(k\) known houses, engagement must fail at **all \(k\)** of them for no meeting to occur.  
That has probability \((1-q)^k\).

So the total probability of **no engagement at all** is:
"""
    )
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

    st.markdown("Therefore, the probability of **meeting at least one known person** is:")
    st.latex(r"P(\text{meet ≥ 1 known person}) = 1 - P(\text{no engagement})")

    st.markdown(
        """
### 🐈 Intuition

Imagine all houses as balls in a bag:

- **F** friend-house balls  
- **H − F** ordinary balls  

The cat's journey is like drawing \(N\) balls **without replacement**.  
We are asking: *what is the chance that at least one of the drawn balls is a friend-house ball?*  
The term \(q\) allows for the fact that even at a friend’s house, engagement might fail.
"""
    )

# -------------------------------------------------------------------
# TAB 3 — MONTE CARLO (SIMULATION)
# -------------------------------------------------------------------
with tab3:
    st.markdown(
        """
### 🎲 Monte Carlo (Simulation) Model

**Idea:**  
Instead of solving the probability with a formula, we **simulate the scenario many times**
and see how often a meeting happens.
"""
    )

    st.markdown("#### One simulation run looks like this:")

    st.markdown(
        """
1. Randomly choose **F** known houses out of **H**  
2. Randomly choose **N** distinct visited houses out of **H**  
3. Find which of the visited houses are known houses  
4. For each visited known house, flip an engagement coin with probability **q**  
5. If **any** of those flips succeed, we count that run as “meeting happened”
"""
    )

    st.markdown("After many runs, the estimated probability is:")
    st.latex(
r"""
\text{Estimated probability}
\approx
\frac{\text{number of runs with a meeting}}
     {\text{total number of runs}}
"""
    )

    st.markdown(
        """
### 🧠 Why Monte Carlo is useful

- Works when the analytic maths becomes complicated  
  (e.g. distance weighting, biased movement, street networks)  
- Lets us **test and compare** the analytic models  
- Makes the randomness tangible: each run is one possible “lost cat” story

Monte Carlo is like replaying the missing-cat story thousands of times  
and counting how often the cat meets someone you know.
"""
    )

