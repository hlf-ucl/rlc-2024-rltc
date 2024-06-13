"""
Simple toy example of environment.
Renders an example execution, figure used in the paper.
"""

from game_ext.environment import Environment
import networkx as nx
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update({
    # serif doesn't work for some reason
    'font.family': 'sans-serif',
    'font.sans-serif': 'Times New Roman'
})


VOTE_COLOR_0 = "#DC3220"  # red
VOTE_COLOR_1 = "#005AB5"  # blue

TRUST_COLOR_YES = "black"
TRUST_COLOR_NO = "lightgray"


def render(env, ax=None, show_legend=False):
    """Draw the current state of the environment.
    Takes an optional matplotlib Axes argument to draw on.
    """

    if ax is None:
        fig, ax = plt.subplots()
    
    ax.margins(0.1)
    node_size = 1400
    font_size = 24

    num_processes = env.num_processes

    # Create dictionary of node positions to draw (for networkx)
    pos = {}
    m = int(np.sqrt(num_processes))
    for i in range(num_processes):
        row = m - (i // m)
        col = i % m
        pos[i] = col, row
    G = env.graph.to_directed()

    # Colors represent 0/1 votes
    colors = [VOTE_COLOR_0, VOTE_COLOR_1]

    ax.set_title(f"$t = {env.current_timestep}$", fontsize=font_size)
    node_shape_dict = {'o': env.reliable_processes,
                        's': env.unreliable_processes}
    for shape, nodes in node_shape_dict.items():
        nx.draw_networkx_nodes(G, nodelist=[n.id for n in nodes],
                                pos=pos, ax=ax,
                                node_color=[colors[p.vote] for p in nodes],
                                node_size=node_size,
                                node_shape=shape,
                                edgecolors="black",
                                linewidths=2.0)

    # Represent trust by edges
    edgecolors = []
    for edge in G.edges:
        i, j = edge
        p_j = env.processes[j]
        color = TRUST_COLOR_YES if p_j.trusts[i] else TRUST_COLOR_NO
        edgecolors.append(color)

    nx.draw_networkx_edges(G, edge_color=edgecolors,
                            pos=pos, ax=ax,
                            connectionstyle='arc3, rad = 0.15',
                            arrowsize=20,
                            node_size=node_size,
                            width=1.3)

    nx.draw_networkx_labels(G, pos=pos, ax=ax,
                            font_color="white", font_size=font_size,
                            font_weight='bold',
                            labels={n: n+1 for n in G.nodes})

    if show_legend:
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='0',
                markerfacecolor=colors[0], markersize=24),
            Line2D([0], [0], marker='o', color='w', label='1',
                markerfacecolor=colors[1], markersize=24),
        ]
        legend = ax.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.1),
                loc="center", ncol=2, fontsize=18)
        legend.set_title("Node local value", prop={"size": 18})


def main():
    n = 9
    p = 0.3
    T = 30
    f = 8/9
    seed = 42
    
    env = Environment(n, f, p, T, seed=seed)
    env.reset()

    assert len(env.unreliable_processes) == 1
    bad_agent = env.unreliable_processes[0]
    print(bad_agent.id)
    print([p.id for p in bad_agent.neighbours])

    show_frames = [1, 2, 29]
    fig, axs = plt.subplots(ncols=4, figsize=(18, 4))  # tight_layout=True)

    render(env, ax=axs[0])

    actions = np.zeros(len(env.reliable_processes), dtype=int)

    use_trust = True
    if use_trust:
        todolist = {
            0: [(0, 1), (4, 1)],
            1: [(2, 1)]
        }
    else:
        todolist = {}

    ax_idx = 1
    for t in range(T):    

        if t in show_frames:
            render(env, ax=axs[ax_idx])
            ax_idx += 1


        actions.fill(0)
        if t in todolist:

            todo = todolist[t]
            for src, dst in todo:
                for i, p in enumerate(env.reliable_processes):
                    if p.id == src:
                        for j, q in enumerate(p.neighbours):
                            if q.id == dst:
                                actions[i] = j + 1
                                break

        env.step(actions)

    plt.subplots_adjust(wspace=0.2)

    axs[-1].set_title("$t = T$", fontsize=24, font="Times New Roman", fontweight="bold")
    fig.savefig("out/image.svg", bbox_inches="tight")
    plt.show()

    # # Convert to pdf
    # import subprocess
    # subprocess.run(["inkscape", "--export-type=pdf", "image.svg"])
    # inkscape --export-type=pdf image.svg


if __name__ == "__main__":
    main()