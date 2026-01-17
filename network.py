# network.py
import osmnx as ox
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point


def build_house_road_network(place="N1 4QU, UK", radius_m=800, friend_pct=0, seed=42):

    """
    Builds a combined network:
    - street nodes/edges from OSM
    - house nodes attached as leaf nodes to nearest street node
    """
    # --- Street network ---
    G_streets = ox.graph_from_address(
        place,
        dist=radius_m,
        simplify=True
    )

    # IMPORTANT: project street graph to a local metric CRS (meters)
    G_streets = ox.project_graph(G_streets)

    # Get street nodes as GeoDataFrame (now in meters)
    nodes, edges = ox.graph_to_gdfs(G_streets, nodes=True, edges=True)

    # --- Building footprints ---
    buildings = ox.features_from_address(
        place,
        dist=radius_m,
        tags={"building": True}
    )

    buildings = buildings[buildings.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()

    # IMPORTANT: project buildings to the SAME CRS as the street graph
    buildings = buildings.to_crs(nodes.crs)

    # Now centroids are also in meters
    buildings["centroid"] = buildings.geometry.centroid

    # --- Attach houses as leaf nodes ---
    nearest_nodes = ox.distance.nearest_nodes(
        G_streets,
        X=buildings["centroid"].x.values,
        Y=buildings["centroid"].y.values
    )

    G = G_streets.copy()

    for i, (_, row) in enumerate(buildings.iterrows()):
        house_id = f"house_{i}"
        street_node = int(nearest_nodes[i])
        c = row["centroid"]

        G.add_node(
            house_id,
            x=float(c.x),
            y=float(c.y),
            node_type="house"
        )

        # Distance (meters) from house centroid to the nearest street node geometry
        dist = c.distance(nodes.loc[street_node].geometry)

        G.add_edge(street_node, house_id, length=float(dist), edge_type="stub")
        G.add_edge(house_id, street_node, length=float(dist), edge_type="stub")

        # --- Pubs (amenity=pub) ---
    pubs = ox.features_from_address(
        place,
        dist=radius_m,
        tags={"amenity": "pub"}
    )

    # Keep points/polygons and project to same CRS
    pubs = pubs[pubs.geometry.type.isin(["Point", "Polygon", "MultiPolygon"])].copy()
    pubs = pubs.to_crs(nodes.crs)

    # Convert polygons to centroids so everything is a point
    pubs["centroid"] = pubs.geometry.centroid

    nearest_pub_nodes = ox.distance.nearest_nodes(
        G_streets,
        X=pubs["centroid"].x.values,
        Y=pubs["centroid"].y.values
    )

    for j, (_, row) in enumerate(pubs.iterrows()):
        pub_id = f"pub_{j}"
        street_node = int(nearest_pub_nodes[j])
        c = row["centroid"]

        pub_name = row.get("name", "Pub")

        G.add_node(
            pub_id,
            x=float(c.x),
            y=float(c.y),
            node_type="pub",
            name=str(pub_name)
        )

        dist = c.distance(nodes.loc[street_node].geometry)

        G.add_edge(street_node, pub_id, length=float(dist), edge_type="stub")
        G.add_edge(pub_id, street_node, length=float(dist), edge_type="stub")

    import random

    rng = random.Random(seed)
    house_ids = [n for n, d in G.nodes(data=True) if d.get("node_type") == "house"]

    # how many friends based on percentage
    F_friends = int(round((friend_pct / 100.0) * len(house_ids)))
    F_friends = min(F_friends, len(house_ids))

    # reset tag
    for n in house_ids:
        G.nodes[n]["is_friend"] = False

    # sample friends
    for n in rng.sample(house_ids, F_friends):
        G.nodes[n]["is_friend"] = True


    return G

