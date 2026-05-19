import os
from pathlib import Path
import librosa
import numpy as np
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import Wav2Vec2Model, Wav2Vec2Processor

processor = None
models = {}

def main(first_pair, second_pair, model_type='finetuned'):
    first_pair_embed = get_continuum_embeddings(first_pair, model_type)
    second_pair_embed = get_continuum_embeddings(second_pair, model_type)

    return first_pair_embed, second_pair_embed

def initialize_models():
    """ Load the models. """
    global processor, models

    if processor is not None and models:
        return

    load_dotenv()
    hf_token = os.environ["HF_TOKEN"]
    login(token=hf_token)

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    models = {
        "finetuned": Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h"),
        "pretrained": Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    }

def get_continuum_embeddings(continuum_name, model_type='finetuned'):
    """
    Extract mean Wav2Vec2 embeddings from the initial consonant region
    of the continuum.

    Args:
        continuum_name (str): Name of the continuum, e.g. 'dash-tash'.
        model (str): Which Wav2Vec2 version to use. Options include 'finetuned'
            and 'pretrained'.

    Returns:
        embed_dict (dict): Nested dictionary of embeddings in the format:
            {layer{step{np.ndarray}}}, where the array is a (768,) vector
            representing the initial consonant.
    """
    initialize_models()
    model = models[model_type]
    model.eval()

    start_frame = int(0.05 / 0.02)
    steps = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]
    embed_dict = {layer: {} for layer in range(13)}
    base_path = Path("Stimuli/Continua")

    with torch.no_grad():
        for i, step in enumerate(steps):
            sound_path = (base_path/continuum_name/f"{continuum_name}_F0_{step}_VOT_{step}.wav")
            sound, sr = librosa.load(sound_path, sr=16000)
            sound_tensor = processor(sound, sampling_rate=sr, return_tensors='pt')
            # max: prevent an error for the first frame where VOT = 0
            end_frame = max(int((0.05 + i * 0.009) / 0.02), start_frame + 1)

            embeddings = model(**sound_tensor, output_hidden_states=True)

            for layer in range(13):
                # average frame-level representations across the consonant region
                mean = embeddings.hidden_states[layer][0, start_frame:end_frame, :].mean(dim=0)
                embed_dict[layer][step] = mean.numpy()

    return embed_dict

if __name__ == "__main__":
    main("dash-tash", "task-dask", "finetuned")
