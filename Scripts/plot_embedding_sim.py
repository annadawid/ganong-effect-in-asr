import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import embedding_sim
import extract_embeddings

def main(first_pair, second_pair, model_name):
    """
    Execute the complete speech continuum analysis pipeline:
    extraction of the consonant embeddings, computing embedding similarity
    relative to /t/, and plotting the results.

    Args:
        first_pair (str): Name of the first continuum, e.g. 'dash-tash'.
        second_pair (str): Name of the second continuum, e.g. 'task-dask'.
        model_name (str): Name of the ASR model.
            Options:
                - 'Whisper'
                - 'wav2vec2_finetuned'
                - 'wav2vec2_pretrained'
    """
    results_first_pair, results_second_pair = embedding_sim.main(first_pair, second_pair, model_name)
    plot_embed_sim(results_first_pair, results_second_pair, pair1_label=first_pair, pair2_label=second_pair)

def plot_embed_sim(embed_sim_pair1, embed_sim_pair2, pair1_label, pair2_label):
    """
    Generate and save a 13-panel plot comparing layer-wise relative
    embedding similarities.

    Args:
        embed_sim_pair1 (dict): Dictionary mapping each layer (13 in total) to a
            list of 11 float similarity scores for the first pair.
            Format: {layer: [sim_01, ..., sim_11]}
        embed_sim_pair2 (dict): Dictionary mapping each layer (13 in total) to a
            list of 11 float similarity scores for the second pair.
            Format: {layer: [sim_01, ..., sim_11]}
        pair1_label (str): String label for the first pair used in the plot legend
            (e.g., 'dash-tash').
        pair2_label (str): String label for the second pair used in the plot legend
            (e.g., 'task-dask').

    """
    sns.set_theme(style="whitegrid")

    x = np.arange(1, 12)
    layer_names = ["CNN"] + [f"T{i}" for i in range(1, 13)]

    fig, axes = plt.subplots(
        2, 7,
        figsize=(12, 5),
        sharex=True,
        sharey=True
    )

    axes = axes.flatten()

    for layer, ax in enumerate(axes):

        # hide empty final subplot
        if layer >= len(layer_names):
            ax.axis("off")
            continue

        sns.lineplot(
            x=x,
            y=embed_sim_pair1[layer],
            ax=ax,
            color="#4C72B0",
            marker="d",
            markersize=5,
            label=pair1_label
        )

        sns.lineplot(
            x=x,
            y=embed_sim_pair2[layer],
            ax=ax,
            color="#C44E52",
            marker="s",
            markersize=5,
            label=pair2_label
        )

        ax.set_title(layer_names[layer], fontsize=10)

        ax.set_xlim(1, 11)
        ax.set_ylim(0, 1)

        ax.set_xticks(range(1, 12))
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["1", "", "", "", "", "6", "", "", "", "", "11"])

        ax.set_xlabel("")
        ax.set_ylabel("")

        ax.grid(True, linewidth=0.4)

        # remove duplicate y-axis labels
        if layer % 7 != 0:
            ax.tick_params(labelleft=False)

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
        bbox_to_anchor=(0.5, 1.02)
    )

    # shared axis labels
    fig.text(
        0.5,
        0.02,
        "Continuum step",
        ha="center",
        fontsize=12
    )

    fig.text(
        -0.03,
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

    plt.savefig("embedding_similarity_results.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main("dash-tash", "task-dask", model_name="Whisper")
