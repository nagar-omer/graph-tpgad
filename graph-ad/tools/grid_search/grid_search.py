import hashlib
import json
from copy import deepcopy
from itertools import product

from anomaly_detection import TPGAD


def extract_search_space(search_space, root="", search_flat_dict={}):
    for k, v in search_space.items():
        if type(v) is list:
            search_flat_dict[f"{root}/{k}"] = v
        elif type(v) is dict:
            search_flat_dict.update(extract_search_space(search_space[k], root=f"{root}/{k}",
                                                         search_flat_dict=search_flat_dict))
    return search_flat_dict


def build_config(default, config):
    new_config = deepcopy(default)
    for key_path, v in config.items():
        curr_dict = new_config
        key_path = key_path[1:].split("/")
        for k in key_path[:-1]:
            if k not in curr_dict:
                curr_dict[k] = {}
            curr_dict = curr_dict[k]
        curr_dict[key_path[-1]] = v

    irrelevant_score_types = ['gmm', 'knn', "local_outlier"]
    irrelevant_score_types.remove(new_config['score']['type'])
    for k in irrelevant_score_types:
        if k in new_config['score']['params']:
            del new_config['score']['params'][k]

    sha = hashlib.md5(str(new_config).encode()).hexdigest()
    return new_config, sha


def config_iterator(default_params, search_space):
    default_params = default_params if type(default_params) is dict else json.load(open(default_params, "rt"))
    search_space = search_space if type(search_space) is dict else json.load(open(search_space, "rt"))

    overlap_configurations = set()
    config_dict = extract_search_space(search_space)
    sorted_keys = [k for k in sorted(config_dict)]
    all_configs = list(product(*[config_dict[k] for k in sorted_keys]))
    for config in all_configs:
        config, sha = build_config(default_params, {k: v for k, v in zip(sorted_keys, config)})
        if sha in overlap_configurations:
            continue
        else:
            overlap_configurations.add(sha)
            yield config


def run_grid(default_params, search_space):
    for config in config_iterator(default_params, search_space):
        t = TPGAD(config)
        t.pull_results()
        write_results()


if __name__ == '__main__':
    default_params = "C:/Users/ovedn/Desktop/Git/TPGAD/graph-ad/params/enron_param.json"
    search_space = "C:/Users/ovedn/Desktop/Git/TPGAD/graph-ad/tools/grid_search/search_space/search_space_params.json"
    for config in config_iterator(default_params, search_space):
        print(config)
    # run_grid(default_params, search_space)