import extract_embeddings
import numpy as np
import scipy
import pandas as pd

def main(first_pair, second_pair, model_name):
    """
    Run the embedding similarity analysis pipeline for two continua.

    This function extracts embeddings for each continuum, calculates
    /t/ and /d/ endpoints, computes relative /t/ similarity
    scores across all layers and continuum steps, saves the results to a CSV
    file, and returns the similarity scores.

    Args:
        first_pair (str): Name of the first continuum, e.g. 'dash-tash'.
        second_pair (str): Name of the second continuum, e.g. 'task-dask'.
        model_name (str): Name of the ASR model.
            Options:
                - 'Whisper'
                - 'wav2vec2_finetuned'
                - 'wav2vec2_pretrained'

    Returns:
            results_first_pair (dict): Dictionary mapping each layer (13 in total) to a
                list of 11 float similarity scores for the first pair.
                Format: {layer: [sim_01, ..., sim_11]}
            results_second_pair (dict):  Dictionary mapping each layer (13 in total) to a
                list of 11 float similarity scores for the second pair.
                Format: {layer: [sim_01, ..., sim_11]}
    """
    first_pair_embed, second_pair_embed = extract_embeddings.main(first_pair, second_pair, model_name)

    endpoints = get_endpoints(first_pair_embed, second_pair_embed)

    results_first_pair = get_embed_sim(first_pair_embed, endpoints)
    results_second_pair = get_embed_sim(second_pair_embed, endpoints)

    save_to_csv(results_first_pair, results_second_pair, first_pair, second_pair, model_name)

    return results_first_pair, results_second_pair

def get_endpoints(t_embeddings, d_embeddings):
    """
    Compute mean /t/ and /d/ endpoint representations for each wav2vec2 layer.

    Args:
        t_embeddings (dict): Nested dictionary of embeddings for the /t/-biased continuum.

        d_embeddings (dict): Nested dictionary of embeddings for the /d/-biased continuum.

    Returns:
        endpoints (dict): Nested dictionary in the format:
            {layer: {"t": np.ndarray, "d": np.ndarray}}, where each array is a (768,) vector
    """
    endpoints = {}

    for layer in range(13):
        t = np.mean([
            d_embeddings[layer]["11"],
            t_embeddings[layer]["11"]
        ], axis=0)

        d = np.mean([
            d_embeddings[layer]["01"],
            t_embeddings[layer]["01"]
        ], axis=0)

        endpoints[layer] = {"t": t, "d": d}

    return endpoints

def cosine_distance(x, endpoint):
    """ Compute cosine distance between two embedding vectors. """
    dot_product = np.dot(x, endpoint)
    norm_x = np.linalg.norm(x)
    norm_endpoint = np.linalg.norm(endpoint)

    return 1 - (dot_product / (norm_x * norm_endpoint))

def relative_t_similarity(x, t_endpoint, d_endpoint):
    """
    Compute relative similarity of an embedding to the /t/ endpoint.
    Values closer to 1 indicate greater similarity to /t/,
    and values closer to 0 indicate greater similarity to /d/.

    Args:
        x (np.ndarray): Input embedding vector.

        t_endpoint (np.ndarray): Mean /t/ endpoint representation, shape (768,)

        d_endpoint (np.ndarray): Mean /d/ endpoint representation, shape (768,)

    Returns:
        sim_x_t (float): Relative /t/ similarity score.
    """
    cos_dist_x_t = cosine_distance(x, t_endpoint)
    cos_dist_x_d = cosine_distance(x, d_endpoint)
    sim_x_t = 1 - (cos_dist_x_t / (cos_dist_x_t + cos_dist_x_d))

    return sim_x_t

def get_embed_sim(embed_dict_input, endpoints):
    """
    Compute relative /t/ similarity for all continuum steps and layers.

    Args:
        embed_dict_input (dict): Nested embedding dictionary in the format:
            {layer: {step: np.ndarray}}

        endpoints (dict): Dictionary containing mean /t/ and /d/ endpoint vectors.

    Returns:
        dict: Dictionary mapping each layer to a list of similarity values
            across continuum steps. Format: {layer: [sim_01, ..., sim_11]}
    """
    similarities = {}
    steps = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]

    for layer in range(13):
        t = endpoints[layer]["t"]
        d = endpoints[layer]["d"]
        similarities[layer] = []

        for step in steps:
            embed_sim = relative_t_similarity(embed_dict_input[layer][step], t, d)
            similarities[layer].append(embed_sim)

    return similarities

def save_to_csv(results_first_pair, results_second_pair, first_pair, second_pair, model_name):
    """Save embedding similarity results from two continua to a CSV file."""

    data = []

    for layer in range(13):
        for idx, val in enumerate(results_first_pair[layer]):
            data.append({
                'Layer': layer,
                'Step': idx,
                'Model': model_name,
                'Pair': first_pair,
                't-similarity': val})
        for idx2, val2 in enumerate(results_second_pair[layer]):
            data.append({
                'Layer': layer,
                'Step': idx2,
                'Model': model_name,
                'Pair': second_pair,
                't-similarity': val2})

    df = pd.DataFrame(data)
    df.to_csv(f'{model_name}_embed_data.csv', index=False)

if __name__ == "__main__":
    main("dash-tash", "task-dask", "Whisper")
    main("dash-tash", "task-dask", "wav2vec2_pretrained")
    main("dash-tash", "task-dask", "wav2vec2_finetuned")
