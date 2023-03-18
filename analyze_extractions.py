"""
This is a script to pair the labels, method outputs, and the original data.
"""

import pandas as pd
import numpy as np

import os

def main():
    """
    Load the test sorted data at 
    An example file path is ./generation_results/roberta-large-mnli_imdb_1000_prompt11_normal_last/test_sorted.csv
    as a pandas dataframe

    Then, load the extracted method labels also as a dataframe 
    An example filel path is ./extraction_results/states_roberta-large-mnli_LR/all/imdb11_LR.csv

    Grab the label column and compare to make sure they agree. i.e. labels == labels for both
    """
    method = "LR"

    # Load the test sorted data as a pandas dataframe
    prompt_num = 0
    last_labels = None
    while os.path.exists(f"./generation_results/roberta-large-mnli_imdb_1000_prompt{prompt_num}_normal_last/test_sorted.csv"):
        # load our shuffled data
        test_data_path = f"./generation_results/roberta-large-mnli_imdb_1000_prompt{prompt_num}_normal_last/test_sorted.csv"
        test_data = pd.read_csv(test_data_path)
        train_data_path = f"./generation_results/roberta-large-mnli_imdb_1000_prompt{prompt_num}_normal_last/train_sorted.csv"
        train_data = pd.read_csv(train_data_path)

        
        # Load the extracted method labels also as a dataframe
        method_labels_path = f"./extraction_results/testing/states_roberta-large-mnli_{method}/all/imdb{prompt_num}_{method}.csv"
        method_labels = pd.read_csv(method_labels_path)

        labels_1 = test_data["label"]
        labels_2 = method_labels["label"]
        labels_train = train_data["label"][0:len(labels_2)]
        # print(labels_1)
        # print(labels_2)
        print(f"prompt num {prompt_num} gets {np.sum(np.abs(labels_1 - labels_2))} matched wrong")
        print(f"Using train ordering, we get {np.sum(np.abs(labels_train - labels_2))} matched wrong")

        if last_labels is not None:
            print(f"Their consistency is {np.sum(np.abs(last_labels - labels_2))} matched wrong")
        print()
        prompt_num += 1
        last_labels = labels_2

if __name__ == "__main__":
    main()