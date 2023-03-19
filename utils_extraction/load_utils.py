import os
import numpy as np
import pandas as pd
import json
import re

######## JSON Load ########
json_dir = "./registration"

with open("{}.json".format(json_dir), "r") as f:
    global_dict = json.load(f)
registered_dataset_list = global_dict["dataset_list"]
registered_models = global_dict["registered_models"]
registered_prefix = global_dict["registered_prefix"]
models_layer_num = global_dict["models_layer_num"]



load_dir = "./generation_results"
# print("------ Func: set_load_dir ------\n\
# ## Input = (path) ##\n\
#     path: set the dir that loads hidden states to path.\n\
# [ATTENTION]: Only load the first 4 prompts for speed.\n\
# ")
def set_load_dir(path):
    global load_dir
    load_dir = path

def getDirList(mdl, set_name, load_dir, data_num, confusion, place, prompt_idx):
    length = len(mdl)
    filter = [w for w in os.listdir(load_dir) if (mdl == w[:length] and mdl + "_" in w and set_name + "_" in w and str(data_num) + "_" in w and confusion + "_" in w)] # and place in w)]
    if prompt_idx is not None:
        filter = [w for w in filter if int(w.split("_")[3][6:]) in prompt_idx]
    return [os.path.join(load_dir, w) for w in filter]

def organizeStates(lis, mode):
    '''
        Whether to do minus, to concat or do nothing
    '''
    if mode in ["0", "1"]:
        return lis[int(mode)]
    elif mode == "minus":
        return lis[0] - lis[1]
    elif mode == "concat":
        return np.concatenate(lis, axis = -1)
    else:
        raise NotImplementedError("This mode is not supported.")

def normalize(data, scale =True, demean = True):
    # demean the array and rescale each data point
    data = data - np.mean(data, axis = 0) if demean else data
    if not scale:
        return data
    norm = np.linalg.norm(data, axis=1)
    avgnorm = np.mean(norm)
    return data / avgnorm * np.sqrt(data.shape[1])

def loadHiddenStates(mdl, set_name, load_dir, prompt_idx, location = "encoder", layer = -1, data_num = 1000, confusion = "normal", place = "last", scale = True, demean = True, mode = "minus", verbose = True):
    '''
        Load generated hidden states, return a dict where key is the dataset name and values is a list. Each tuple in the list is the (x,y) pair of one prompt.
        if mode == minus, then get h - h'
        if mode == concat, then get np.concatenate([h,h'])
        elif mode == 0 or 1, then get h or h'
    '''

    dir_list = getDirList(mdl, set_name, load_dir, data_num, confusion, place, prompt_idx)
    append_list = ["_" + location + str(layer) for _ in dir_list]
    
    hidden_states = [
                        organizeStates(
                            [np.load(os.path.join(w, "0{}.npy".format(app))), 
                            np.load(os.path.join(w, "1{}.npy".format(app)))], 
                            mode = mode) 
                        for w, app in zip(dir_list, append_list)
                    ]

    # normalize
    hidden_states = [normalize(w, scale, demean) for w in hidden_states]
    if verbose:
        print("{} prompts for {}, with shape {}".format(len(hidden_states), set_name, hidden_states[0].shape))
    labels = [np.array(pd.read_csv(os.path.join(w, "frame.csv"))["label"].to_list()) for w in dir_list]

    return [(u,v) for u,v in zip(hidden_states, labels)]

def copy_permuted_generations(mdl, set_name, load_dir, prompt_idx, location = "encoder", layer = -1, data_num = 1000, confusion = "normal", place = "last", scale = True, demean = True, mode = "minus", verbose = True, permutation_dict=None):
    '''
        Load generated hidden states, then save a permuted version of it.
    '''

    dir_list = getDirList(mdl, set_name, load_dir, data_num, confusion, place, prompt_idx)

    # map the read dir (in generation_results) to the correct dir to save to (also in generation_results)
    # such that the data matches the extraction results
    sorted_list = sorted(dir_list, key=lambda s: int(re.search(r"prompt(\d+)", s).group(1)))
    gen_dir_to_correct_dir = {s1: s2 for s1, s2 in zip(dir_list, sorted_list)}
    # gen_dir_to_correct_dir_idx = {k:v for v,k in enumerate(dir_list)}
    
    for dir in dir_list:
        df_1 = pd.read_csv(os.path.join(dir, "frame.csv"))
        # if os.path.exists(os.path.join(dir, "test_sorted.csv")):
        #     print("Sorted df already exists, skip for {}".format(dir))
        #     continue
        save_dir = gen_dir_to_correct_dir[dir]
        # save_dir_idx = gen_dir_to_correct_dir_idx[dir]
        # print("We are saving re-formatted dataframes from {} to {}".format(dir, save_dir))
        df_permuted_train = df_1.iloc[permutation_dict[set_name][0]]
        df_permuted_train.to_csv(os.path.join(save_dir, f"train_sorted.csv"))
        # df_permuted_train.to_csv(os.path.join(save_dir, f"train_sorted_from_prompt{save_dir_idx}.csv"))
        df_permuted_test = df_1.iloc[permutation_dict[set_name][1]]
        df_permuted_test.to_csv(os.path.join(save_dir, f"test_sorted.csv"))
        # df_permuted_test.to_csv(os.path.join(save_dir, f"test_sorted_from_prompt{save_dir_idx}.csv"))

    # now save permuted versions of the LOG PROBS
    dir_list = [d.replace("generation_results", "generation_results/log_probs").replace("last", "_log_probs.csv") for d in dir_list]
    sorted_list = sorted(dir_list, key=lambda s: int(re.search(r"prompt(\d+)", s).group(1)))
    gen_dir_to_correct_dir = {s1: s2 for s1, s2 in zip(dir_list, sorted_list)}

    if not os.path.exists("./generation_results/log_probs_test_sorted"):
        os.makedirs("./generation_results/log_probs_test_sorted")
    if not os.path.exists("./generation_results/log_probs_train_sorted"):
        os.makedirs("./generation_results/log_probs_train_sorted")
    
    for dir in dir_list:
        df_1 = pd.read_csv(dir)
        # if os.path.exists(os.path.join(dir, "test_sorted.csv")):
        #     print("Sorted df already exists, skip for {}".format(dir))
        #     continue
        save_dir_test = gen_dir_to_correct_dir[dir].replace("log_probs", "log_probs_test_sorted")
        save_dir_train = save_dir_test.replace("log_probs_test_sorted", "log_probs_train_sorted")
        # print("We are saving re-formatted dataframes from {} to {}".format(dir, save_dir))
        df_permuted_train = df_1.iloc[permutation_dict[set_name][0]]
        df_permuted_train.to_csv(save_dir_train)
        df_permuted_test = df_1.iloc[permutation_dict[set_name][1]]
        df_permuted_test.to_csv(save_dir_test)


def getPermutation(data_list, rate = 0.6):
    np.random.seed(0)
    length = len(data_list[0][1])
    permutation = np.random.permutation(range(length)).reshape(-1)
    # permutation = np.arange(length)
    return [permutation[: int(length * rate)], permutation[int(length * rate):]]


# print("------ Func: getDic ------\n\
# ## Input = (mdl_name, dataset_list, prefix = \"normal\", location=\"encoder\", layer=-1, scale = True, demean = True, mode = \"minus\", verbose = True) ##\n\
#     mdl_name: name of the model\n\
#     dataset_list: list of all datasets\n\
#     prefix: the prefix used for the hidden states\n\
#     location: Either 'encoder' or 'decoder'. Determine which hidden states to load.\n\
#     layer: An index representing which layer in `location` should we load the hidden state from.\n\
#     prompt_dict: dict of prompts to consider. Default is taking all prompts (empty dict). Key is the set name and value is an index list. Only return hiiden states from corresponding prompts.\n\
#     data_num: population of the dataset. Default is 1000, and it depends on generation process.\n\
#     scale: whether to rescale the whole dataset\n\
#     demean: whether to subtract the mean\n\
#     mode: how to generate hidden states from h and h'\n\
#     verbose: Whether to print more\n\
# ## Output = [data_dict, permutation_dict] ##\n\
#     data_dict: a dict with key equals to set name, and value is a list. Each element in the list is a tuple (state, label). state has shape (#data * #dim), and label has shape (#data).\n\
#     permutation_dict: [train_idx, test_idx], where train_idx is the subset of [#data] that corresponds to the training set, and test_idx is the subset that corresponds to the test set.\n\
# ")
def getDic(mdl_name, dataset_list, prefix = "normal", location="auto", layer=-1, prompt_dict = None, data_num = 1000, scale = True, demean = True, mode = "minus", verbose = True):
    global load_dir
    if location == "auto":
        location = "decoder" if "gpt" in mdl_name else "encoder"
    if location == "decoder" and layer < 0:
        layer += models_layer_num[mdl_name]
    print("start loading {} hidden states {} for {} with {} prefix. Prompt_dict: {}, Scale: {}, Demean: {}, Mode: {}".format(location, layer, mdl_name, prefix, prompt_dict if prompt_dict is not None else "ALL", scale, demean, mode))
    prompt_dict = prompt_dict if prompt_dict is not None else {key: None for key in dataset_list}
    data_dict = {set_name: loadHiddenStates(mdl_name, set_name, load_dir, prompt_dict[set_name], location, layer, data_num = data_num, confusion = prefix, scale = scale, demean = demean, mode = mode, verbose = verbose) for set_name in dataset_list}
    permutation_dict = {set_name: getPermutation(data_dict[set_name]) for set_name in dataset_list}
    _ = {set_name: copy_permuted_generations(mdl_name, set_name, load_dir, prompt_dict[set_name], location, layer, data_num = data_num, confusion = prefix, scale = scale, demean = demean, mode = mode, verbose = verbose, permutation_dict=permutation_dict) for set_name in dataset_list}
    return data_dict, permutation_dict

# print("------ Func: get_zeros_acc ------\n\
# ## Input = csv_name, mdl_name, dataset_list, prefix, prompt_dict = None, avg = False\n\
#     csv_name: The name of csv we get accuracy from.\n\
#     mdl_name: The name of the model.\n\
#     dataset_list: List of dataset you want the accuracy from.\n\
#     prefix: The name of prefix.\n\
#     prompt_dict: Same as in getDir(). You can specify which prompt to get using this variable. Default is None, i.e. get all prompts.\n\
#     avg: Whether to average upon return. If True, will return a numbers, otherwise a dict with key from dataset_list and values being a list of accuracy.\n\
# ## Output = number / dict, depending on `avg`\n\
# "
# )
def get_zeros_acc(csv_name, mdl_name, dataset_list, prefix, prompt_dict = None, avg = False):
    zeros = pd.read_csv(os.path.join(load_dir, csv_name + ".csv"))
    zeros.dropna(subset=["calibrated"], inplace=True)
    subzeros = zeros.loc[(zeros["model"] == mdl_name) & (zeros["prefix"] == prefix)]
    
    # Extend prompt_dict to ALL dict if it is None
    if prompt_dict is None:
        prompt_dict = {key: range(1000) for key in dataset_list}
    
    # Extract accuracy, each key is a set name and value is a list of acc
    acc_dict = {}
    for dataset in dataset_list:
        filtered_csv = subzeros.loc[(subzeros["dataset"] == dataset) & (
            subzeros["prompt_idx"].isin(prompt_dict[dataset])
        )]
        acc_dict[dataset] = filtered_csv["calibrated"].to_list()

    if not avg:
        return acc_dict
    else:
        # get the dataset avg, and finally the global level avg
        return np.mean([np.mean(values) for values in acc_dict.values()])