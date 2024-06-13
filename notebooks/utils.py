import networkx as nx
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

# Constants
# https://davidmathlogic.com/colorblind/#%23005AB5-%23DC3220
VOTE_COLOR_0 = "#DC3220"  # red
VOTE_COLOR_1 = "#005AB5"  # blue

TRUST_COLOR_YES = "black"
TRUST_COLOR_NO = "lightgray"
   

import subprocess

    
def export_to_pdf(filename):
    # inkscape --export-type=pdf image.svg
    subprocess.run(["inkscape", "--export-type=pdf", filename])
    


def gen_graph(m, n):
    num_nodes = m * n

    # Arrange node positions as a 2D grid (row-major order)
    G = nx.grid_2d_graph(m, n)

    pos = {}
    for i in range(num_nodes):
        row = m - (i // m)
        col = i % m
        pos[i] = col, row

    # Relabel nodes from (x, y) coordinates to 1, 2, ... N
    mapping = {n: i for i, n in enumerate(G.nodes)}
    G = nx.relabel_nodes(G, mapping)

    # Convert to directed graph to draw arrows
    G = G.to_directed()

    return G, pos


def draw_grid(G, pos, votes, trust_edges=None, is_reliable=None, show_legend=True,
node_size=1400,
font_size=24,
arrow_size=20):

    full_trust = trust_edges is None
    
    if trust_edges is not None:
        trust_edges = [trust_edges[e] for e in G.edges]

    # Draw
    fig, ax = plt.subplots(figsize=(4, 4)) #, tight_layout=True)
    
    ax.margins(0.08)

    # Nodes
    # node_size = 1400
    # font_size = 24
    colors = [VOTE_COLOR_0 if vote else VOTE_COLOR_1 for vote in votes]
    
    if is_reliable is None:
        nx.draw_networkx_nodes(G, pos=pos, ax=ax,
                            node_color=colors,
                            node_size=node_size,
                            edgecolors="black",
                            linewidths=2.0)
    else:
        nx.draw_networkx_nodes(G, pos=pos, ax=ax,
                      node_color=[c for c, r in zip(colors, is_reliable) if r],
                      node_size=node_size,
                      edgecolors="black",
                      linewidths=2.0,
                      node_shape="o",
                      nodelist = [i for i, x in enumerate(is_reliable) if x])
        nx.draw_networkx_nodes(G, pos=pos, ax=ax,
                            node_color=[c for c, r in zip(colors, is_reliable) if not r],
                            node_size=node_size,
                            edgecolors="black",
                            linewidths=2.0,
                            node_shape="s",
                            nodelist = [i for i, x in enumerate(is_reliable) if not x])

    nx.draw_networkx_labels(
        G,
        pos=pos,
        ax=ax,
        font_color="white",
        font_family="Times New Roman",
        font_weight="bold",
        font_size=font_size,
        labels={n: n+1 for n in G.nodes})

    # Edges
    if full_trust:
        edgecolors = TRUST_COLOR_YES
    else:
        edgecolors = [TRUST_COLOR_YES if trusts else TRUST_COLOR_NO for trusts in trust_edges]

    nx.draw_networkx_edges(
        G,
        pos=pos,
        ax=ax,
        edge_color=edgecolors,
        connectionstyle='arc3, rad = 0.15',
        arrowsize=arrow_size,
        node_size=node_size,
        width=1.3);

    if show_legend:
        # Explain values
        legend_elements = [
            mpl.lines.Line2D([0], [0], marker='o', color='w', label='0', markerfacecolor=VOTE_COLOR_0, markersize=24),
            mpl.lines.Line2D([0], [0], marker='o', color='w', label='1', markerfacecolor=VOTE_COLOR_1, markersize=24),
        ]

        legend_font_size = 18

        legend = ax.legend(
            handles=legend_elements,
            loc="upper center",
            ncol=2,
            fontsize=legend_font_size,
            bbox_to_anchor=(0.5, 0)
        )
        legend.set_title("Node local value", prop={"size": legend_font_size})

    return fig, ax
