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


def get_llm_response_via_api(prompt,
                             API_BASE="https://api.together.xyz",
                             API_KEY="[YOUR_API_KEY]",
                             LLM_MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1",
                             TAU=1.0,
                             TOP_P=1.0,
                             N=1,
                             SEED=42,
                             MAX_TRIALS=5,
                             TIME_GAP=5):
    '''
    res = get_llm_response_via_api(prompt='hello')  # Default: TAU Sampling (TAU=1.0)
    res = get_llm_response_via_api(prompt='hello', TAU=0)  # Greedy Decoding
    res = get_llm_response_via_api(prompt='hello', TAU=0.5, N=2, SEED=None)  # Return Multiple Responses w/ TAU Sampling
    '''
    openai.api_base = API_BASE
    openai.api_key = API_KEY
    completion = None
    while MAX_TRIALS:
        MAX_TRIALS -= 1
        try:
            completion = openai.ChatCompletion.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                n=N,
                temperature=TAU,
                top_p=TOP_P,
                seed=SEED,
            )
            break
        except Exception as e:
            print(e)
            print("Retrying...")
            time.sleep(TIME_GAP)

    if completion is None:
        raise Exception("Reach MAX_TRIALS={}".format(MAX_TRIALS))
    contents = completion.choices
    if len(contents) == 1:
        return contents[0].message["content"]
    else:
        return [c.message["content"] for c in contents]


if __name__ == '__main__':
    pass
