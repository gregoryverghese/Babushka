# viz.py
import plotly.graph_objects as go


def plot_network_3d(G, max_edges=20000, show_houses=True, show_pubs=True, show_friends=True):

    """
    Interactive 3D visualisation of the road + house network.
    Z-axis is artificial (used only for visual separation).
    """

    # --- Separate nodes by type ---
    street_x, street_y, street_z = [], [], []
    house_x, house_y, house_z = [], [], []
    pub_x, pub_y, pub_z, pub_text = [], [], [], []
    friend_x, friend_y, friend_z = [], [], []


    for n, data in G.nodes(data=True):
        x = data.get("x")
        y = data.get("y")
        if x is None or y is None:
            continue


        if data.get("node_type") == "pub":
            pub_x.append(x)
            pub_y.append(y)
            pub_z.append(2.0)  # float them slightly above houses
            
            pub_text.append(data.get("name", "Pub"))

        elif data.get("node_type") == "house":
            if data.get("is_friend"):
                friend_x.append(x); friend_y.append(y); friend_z.append(1.0)
            else:
                house_x.append(x); house_y.append(y); house_z.append(1.0)


        else:
            street_x.append(x)
            street_y.append(y)
            street_z.append(0.0)

    # --- Build edge traces ---
    edge_x, edge_y, edge_z = [], [], []
    count = 0

    for u, v in G.edges():
        if count > max_edges:
            break

        u_data = G.nodes[u]
        v_data = G.nodes[v]

        if "x" not in u_data or "x" not in v_data:
            continue

        edge_x += [u_data["x"], v_data["x"], None]
        edge_y += [u_data["y"], v_data["y"], None]
        edge_z += [
            1.0 if u_data.get("node_type") == "house" else 0.0,
            1.0 if v_data.get("node_type") == "house" else 0.0,
            None
        ]
        count += 1

    # --- Plotly traces ---
    edges = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode="lines",
        line=dict(color="lightgray", width=1),
        hoverinfo="none",
        name="Edges"
    )

    streets = go.Scatter3d(
        x=street_x, y=street_y, z=street_z,
        mode="markers",
        marker=dict(size=2, color="black"),
        name="Street nodes"
    )

    houses = go.Scatter3d(
        x=house_x, y=house_y, z=house_z,
        mode="markers",
        marker=dict(size=3, color="red"),
        name="Houses"
    )



    pubs = go.Scatter3d(
    x=pub_x, y=pub_y, z=pub_z,
    mode="markers",
    marker=dict(size=5, color="blue"),
    name="Pubs",
    text=pub_text,
    hovertemplate="%{text}<extra></extra>"
    )

    friends = go.Scatter3d(
        x=friend_x, y=friend_y, z=friend_z,
        mode="markers",
        marker=dict(size=4, color="green"),
        name="Friend houses"
)




    fig = go.Figure(data=[edges, streets, houses, pubs])
    if show_friends:
        fig.add_trace(friends)

    fig.update_layout(
        height=700,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=dict(
                eye=dict(x=7.0, y=7.0, z=7.0),
                center=dict(x=0, y=0, z=0)
            )
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(yanchor="top", y=0.95)
    )

    return fig
