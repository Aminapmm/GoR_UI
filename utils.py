import os
import json
import pickle
import time
import datetime
import random

import torch
import openai
import networkx as nx
import numpy as np


def show_time():
    time_stamp = '\033[1;31;40m[' + str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + ']\033[0m'

    return time_stamp


def get_device(index=0):
    return torch.device("cuda:" + str(index) if torch.cuda.is_available() else "cpu")


def text_wrap(text):
    return '\033[1;31;40m' + str(text) + '\033[0m'


def load_nx(path) -> nx.Graph:
    return nx.read_graphml(path)


def store_nx(nx_obj, path):
    nx.write_graphml_lxml(nx_obj, path)


def write_to_json(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)


def write_to_pkl(data, output_file):
    with open(output_file, 'wb') as file:
        pickle.dump(data, file)


def read_from_pkl(output_file):
    with open(output_file, 'rb') as file:
        data = pickle.load(file)

    return data


def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


def print_metrics(metrics):
    for k, v in metrics.items():
        ff = "{} " + k + " ("
        metric = metrics[k]
        for sub_k in metric.keys():
            ff += sub_k + "/"
        ff = ff[:-1] + "): "
        for sub_v in metric.values():
            ff += format(sub_v, ".4f") + "/"
        ff = ff[:-1]
        print(ff.format(show_time()))


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


import requests
import time

def get_llm_response_via_ollama(
    prompt,
    OLLAMA_HOST="http://localhost:11434",
    LLM_MODEL="mistral:latest",
    TAU=1.0,
    TOP_P=1.0,
    N=1,
    SEED=42,  # Note: Ollama might not support `seed` or multiple responses yet
    MAX_TRIALS=5,
    TIME_GAP=5
):
    '''
    res = get_llm_response_via_ollama(prompt='hello')  # Default: TAU Sampling
    '''
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TAU,
        "top_p": TOP_P,
        "stream": False  # Set to True for streaming
    }

    response = None
    while MAX_TRIALS:
        MAX_TRIALS -= 1
        try:
            r = requests.post(url, json=payload)
            r.raise_for_status()
            response = r.json()
            break
        except Exception as e:
            print("Error:", e)
            print("Retrying...")
            time.sleep(TIME_GAP)

    if response is None:
        raise Exception(f"Reach MAX_TRIALS={MAX_TRIALS}")

    return response.get("message", {}).get("content", "[No content returned]")


if __name__ == '__main__':
    pass
