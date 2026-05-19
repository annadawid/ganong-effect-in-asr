import extract_embeddings
import numpy as np

def main(first_pair, second_pair, model_type='finetuned'):
    first_pair_embed, second_pair_embed = extract_embeddings.main(first_pair, second_pair, model_type)

    endpoints = get_endpoints(first_pair_embed, second_pair_embed)

    results_first_pair = get_embed_sim(first_pair_embed, endpoints)
    results_second_pair = get_embed_sim(second_pair_embed, endpoints)

    return results_first_pair, results_second_pair

def get_endpoints(t_embeddings, d_embeddings):
    """
    Compute mean /t/ and /d/ endpoint representations for each Wav2Vec2 layer.

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

if __name__ == "__main__":
    main("dash-tash", "task-dask", "finetuned")
