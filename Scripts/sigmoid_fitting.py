import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import seaborn as sns


def main(whisper_data, wav2vec2_data, parameter, y_axis_name):
    fitted_whisper = run_all_layers(whisper_data)
    fitted_wav2vec2 = run_all_layers(wav2vec2_data)

    plot_param_change(fitted_whisper, fitted_wav2vec2, parameter, y_axis_name)

    return fitted_whisper, fitted_wav2vec2


def sigmoid(x, L, x0, k):
    """
    Compute values of a sigmoid function.
    Where:
        L controls the upper asymptote,
        x0 controls the midpoint,
        k controls the steepness of the curve.
    """
    return L / (1 + np.exp(-k * (x - x0)))

def fit_single_layer(layer_df):
    """
    Fit a sigmoid curve to embedding similarity values for a single layer.
    Args:
        layer_df (pd.DataFrame): Data for one model layer.
            Expected columns:
                - 'Step'
                - 't-similarity'

    Returns:
        popt (np.ndarray): Optimized sigmoid parameters: [L, x0, k]

        mae (float): Mean Absolute Error between observed and fitted
            similarity values.
    """
    x_data = np.arange(1, 12)
    y_data = layer_df["t-similarity"].values

    # Fit the data points to a sigmoid curve
    popt, _ = curve_fit(sigmoid, x_data, y_data, p0=[1.0, 6.0, 1.0])
    L, x0, k = popt
    y_fitted = sigmoid(x_data, L, x0, k)

    # Calculate Mean Absolute Error (MAE)
    mae = np.mean(np.abs(y_data - y_fitted))

    return popt, mae

def run_all_layers(embed_data):
    """
    Fit sigmoid curves independently for every model layer.

    Before fitting, t-similarity values are averaged across continua
    for each layer and continuum step.

    Args:
        embed_data (pd.DataFrame): Embedding similarity data for one ASR model.
            Expected columns:
                - 'Layer'
                - 'Step'
                - 't-similarity'

    Returns:
        pd.DataFrame: Layer-wise sigmoid fitting results with columns:
            - 'Layer': layer index
            - 'k': Sigmoid slope parameter
            - 'mae': Mean Absolute Error of the fitted curve
    """
    # Average over pairs
    embed_data = embed_data.groupby(['Layer', 'Step'])['t-similarity'].mean().reset_index()

    results_list = []

    for layer in range(13):
        mask = embed_data['Layer'].isin([layer])
        popt, mae = fit_single_layer(embed_data[mask])

        results_list.append({
            'Layer': layer,
            'k': popt[2],
            'mae': mae
        })

    return pd.DataFrame(results_list)

def plot_param_change(whisper_df, wav2vec2_df, parameter, y_axis_name):
    """
    Plot layer-wise changes in sigmoid fitting parameters for wav2vec2 and Whisper.

    Args:
        whisper_df (pd.DataFrame): Layer-wise fitted parameters for Whisper.
            Expected columns include:
                - 'Layer'
                - selected parameter column

        wav2vec2_df (pd.DataFrame): Layer-wise fitted parameters for wav2vec2.
            Expected columns include:
                - 'Layer'
                - selected parameter column

        parameter (str): Column name to plot.
            Examples:
                - 'k'
                - 'mae'

        y_axis_name (str): Label displayed on the y-axis.

    Returns:
        None
    """

    layers = ["CNN"] + [f"T{i}" for i in range(1, 13)]
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5))

    whisper_param_mean = whisper_df[parameter].mean()
    wav2vec2_param_mean = wav2vec2_df[parameter].mean()

    # Whisper parameter change
    sns.lineplot(data=whisper_df, x=layers, y=parameter,
                 color='#800080', marker='o', label='Whisper')

    # Whisper parameter mean
    plt.axhline(
        whisper_param_mean,
        linestyle="--",
        alpha=0.5,
        color='#800080',
        label="Whisper mean"
    )

    # wav2vec2 parameter change
    sns.lineplot(data=wav2vec2_df, x=layers, y=parameter,
                 color='#808080', marker='s', label='wav2vec2')

    # wav2vec2 parameter mean
    plt.axhline(
        wav2vec2_param_mean,
        linestyle="--",
        alpha=0.5,
        color='#808080',
        label="wav2vec2 mean"
    )

    plt.xlabel("Layer", fontsize=12)
    plt.ylabel(y_axis_name, fontsize=12)
    plt.xticks(np.arange(0, 13, 1))

    sns.despine()
    plt.legend(frameon=True)
    plt.savefig(f"layerwise_{parameter}.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    wav2vec2_data = pd.read_csv("../Data/wav2vec2_pretrained_embed_data.csv"))
    whisper_data = pd.read_csv("../Data/Whisper_embed_data.csv"))
    main(whisper_data, wav2vec2_data, 'k', 'Sigmoid Slope (k)')
