import librosa
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from dotenv import load_dotenv
from huggingface_hub import login
import torch
import os
import numpy as np
from pathlib import Path

load_dotenv(Path(".."))
hf_token = os.environ["HF_TOKEN"]
login(token=hf_token)

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
models = {
    "finetuned": Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h"),
    "pretrained": Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
}

def get_continuum_embeddings(sound_name, model='finetuned'):
    """
    Extract mean Wav2Vec2 embeddings from the initial consonant region
    of the continuum.

    Args:
        sound_name (str): Name of the continuum, e.g. 'dash-tash'.
        model (str): Which Wav2Vec2 version to use. Options include 'finetuned'
            and 'pretrained'.

    Returns:
        embed_dict (dict): Nested dictionary of embeddings in the format:
            {layer{step{np.ndarray}}}, where the array is a (768,) vector
            representing the initial consonant.
    """
    model = models[model]
    model.eval()
    start_frame = int(0.05 / 0.02)
    steps = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]
    embed_dict = {}
    base_path = Path("Stimuli/Continua")

    for layer in range(13):
        embed_dict[layer] = {}

    with torch.no_grad():
        for i, step in enumerate(steps):
            sound_path = (base_path/sound_name/f"{sound_name}_F0_{step}_VOT_{step}.wav")
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
    dash_tash_embed = get_continuum_embeddings("dash-tash")
    task_dask_embed = get_continuum_embeddings("task-dask")
