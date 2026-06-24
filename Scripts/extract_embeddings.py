import os
from pathlib import Path
import librosa
import numpy as np
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from transformers import WhisperModel, WhisperProcessor

processors = {}
models = {}

def main(first_pair, second_pair, model_name):
    """
    Extract embeddings for two phonetic continua using a specified ASR model.

    Args:
        first_pair (str): Name of the first continuum, e.g. 'dash-tash'.
        second_pair (str): Name of the second continuum, e.g. 'task-dask'.
        model_name (str): Name of the ASR model.
            Options:
                - 'Whisper'
                - 'wav2vec2_finetuned'
                - 'wav2vec2_pretrained'

    Returns:
        first_pair_embed (dict): Nested embedding dictionary for the first
            continuum in the format:
            {layer: {step: np.ndarray}}

        second_pair_embed (dict): Nested embedding dictionary for the second
            continuum in the format:
            {layer: {step: np.ndarray}}
    """
    first_pair_embed = get_continuum_embeddings(first_pair, model_name)
    second_pair_embed = get_continuum_embeddings(second_pair, model_name)

    return first_pair_embed, second_pair_embed

def initialize_models():
    """ Load the models. """
    global processors, models

    if processors and models is not None:
        return

    load_dotenv()
    hf_token = os.environ["HF_TOKEN"]
    login(token=hf_token)

    models = {
        "wav2vec2_finetuned": Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h"),
        "wav2vec2_pretrained": Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base"),
        "Whisper": WhisperModel.from_pretrained("openai/Whisper-small")
    }

    processors = {
        "wav2vec2_finetuned": Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h"),
        "wav2vec2_pretrained": Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base"),
        "Whisper": WhisperProcessor.from_pretrained("openai/Whisper-small")
    }

def get_continuum_embeddings(continuum_name, model_name):
    """
    Extract mean Whisper's encoder or wav2vec2 embeddings from the initial consonant region
    of the continuum.

    Args:
        continuum_name (str): Name of the continuum, e.g. 'dash-tash'.
        model_name (str): Name of the ASR model.
            Options:
                - 'Whisper'
                - 'wav2vec2_finetuned'
                - 'wav2vec2_pretrained'

    Returns:
        embed_dict (dict): Nested dictionary of embeddings in the format:
            {layer{step{np.ndarray}}}, where the array is a (768,) vector
            representing the initial consonant.
    """
    initialize_models()
    model = models[model_name]
    model.eval()

    start_frame = int(0.05 / 0.02)
    steps = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]
    embed_dict = {layer: {} for layer in range(13)}
    base_path = Path("Stimuli/Continua")

    with torch.no_grad():
        for i, step in enumerate(steps):
            sound_path = (base_path/continuum_name/f"{continuum_name}_F0_{step}_VOT_{step}.wav")
            sound, sr = librosa.load(sound_path, sr=16000)
            # max: prevent an error for the first frame where VOT = 0
            end_frame = max(int((0.05 + i * 0.009) / 0.02), start_frame + 1)

            if model_name == "Whisper":
                sound_tensor = processors[model_name](sound, sampling_rate=sr, return_tensors='pt').input_features
                embeddings = model.encoder(input_features=sound_tensor, output_hidden_states=True)
            else:
                sound_tensor = processors[model_name](sound, sampling_rate=sr, return_tensors='pt')
                embeddings = model(**sound_tensor, output_hidden_states=True)

            for layer in range(13):
                # average frame-level representations across the consonant region
                mean = embeddings.hidden_states[layer][0, start_frame:end_frame, :].mean(dim=0)
                embed_dict[layer][step] = mean.numpy()

    return embed_dict

if __name__ == "__main__":
    main("dash-tash", "task-dask", "Whisper")
