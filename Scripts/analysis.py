import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")

x = np.arange(1, 12)
layer_names = ["CNN"] + [f"T{i}" for i in range(1, 13)]

fig, axes = plt.subplots(
    1, 13,
    figsize=(22, 3.2),
    sharex=True,
    sharey=True
)

for layer, ax in enumerate(axes):
    sns.lineplot(
        x=x,
        y=results_dash[layer],
        ax=ax,
        color="#4C72B0",
        marker="d",
        markersize=5,
        label="dash-tash"
    )

    sns.lineplot(
        x=x,
        y=results_task[layer],
        ax=ax,
        color="#C44E52",
        marker="s",
        markersize=5,
        label="task-dask"
    )

    ax.set_title(layer_names[layer], fontsize=11)

    ax.set_xlim(1, 12)
    ax.set_ylim(0, 1)

    ax.set_xticks(range(1, 12))
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["1", "", "", "", "", "6", "", "", "", "", "11"])

    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.grid(True, linewidth=0.4)

    # only keep y tick labels on first panel
    if layer != 0:
        ax.tick_params(labelleft=False)

    # remove repeated legends
    ax.legend().remove()

# shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    frameon=True,
    fontsize=10,
    bbox_to_anchor=(0.5, 1.08)
)

# shared axis labels
fig.text(
    0.5,
    -0.05,
    "Continuum step",
    ha="center",
    fontsize=12
)

fig.text(
    0,
    0.5,
    "Relative /t/ similarity",
    va="center",
    rotation="vertical",
    fontsize=12
)

sns.despine()

plt.subplots_adjust(
    left=0.03,
    right=0.95,
    wspace=0.1
)

plt.savefig("base_pretrained_improved_style.png", dpi=300, bbox_inches="tight")
plt.show()
