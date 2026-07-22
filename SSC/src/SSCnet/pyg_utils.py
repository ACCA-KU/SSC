"""Shared PyG adapters for the migrated SSC network modules."""

from D4CMPP2.networks.src.GCN import graph_sum_pool
from D4CMPP2.networks.src.pyg_hetero import relation_graph, relation_sum


def unpack_ssc_inputs(batch):
    """Resolve canonical D4CMPP2 ISA keys and historical SSC aliases."""

    aliases = {
        "graph": ("compound_graphs", "graph"),
        "r_node": ("compound_r_node", "r_node"),
        "i_node": ("compound_i_node", "i_node"),
        "r_edge": ("compound_r2r_edge", "r_edge"),
        "d_edge": ("compound_d2d_edge", "d_edge"),
        "solv_graph": ("solvent_graphs", "solv_graph"),
        "solv_node_feats": ("solvent_r_node", "solv_node_feats"),
        "solv_edge_feats": ("solvent_r2r_edge", "solv_edge_feats"),
    }
    resolved = {}
    missing = []
    for name, candidates in aliases.items():
        value = next((batch[key] for key in candidates if key in batch), None)
        if value is None:
            missing.append(candidates[0])
        resolved[name] = value
    if missing:
        raise ValueError(
            f"SSC network input is missing required D4CMPP2 fields {missing!r}."
        )
    return resolved


def isa_relation_graphs(graph):
    """Return the real/image/dot homogeneous views of an ISA PyG graph."""

    return (
        relation_graph(graph, "r_nd", "r2r"),
        relation_graph(graph, "i_nd", "i2i"),
        relation_graph(graph, "d_nd", "d2d"),
    )


def expand_to_dot_nodes(graph, graph_features):
    """Repeat one graph-level representation for every fragment (dot) node."""

    dot_graph = relation_graph(graph, "d_nd", "d2d")
    return graph_features[dot_graph.batch]


def image_to_dot(graph, image_node):
    return relation_sum(graph, "i_nd", "i2d", "d_nd", image_node)


def dot_to_real(graph, dot_node):
    return relation_sum(graph, "d_nd", "d2r", "r_nd", dot_node)


__all__ = [
    "dot_to_real",
    "expand_to_dot_nodes",
    "graph_sum_pool",
    "image_to_dot",
    "isa_relation_graphs",
    "relation_graph",
    "unpack_ssc_inputs",
]
