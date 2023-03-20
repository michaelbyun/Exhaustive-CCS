import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_name = "imdb-test-consolidated.xlsx"
predictions = 'CCS*zero-shot' # 'prob CCS output'
outcomes = 'GT label'
prompt = 0
n_buckets = 10
save_dir = "./calibration_graphs/"

data = pd.read_excel(file_name, engine='openpyxl')

def calibration_graph(data, n_buckets=10, prompt= "all"):
    if prompt != "all":
        data = data.iloc[400*prompt+1:400*(prompt+1)+1]

    data['bucket'] = pd.cut(data[predictions], np.linspace(0, 1, n_buckets+1))
    bucket_counts = data.groupby('bucket')[outcomes].count()
    bucket_sums = data.groupby('bucket')[outcomes].sum()
    proportions = bucket_sums / bucket_counts

    plt.figure(figsize=(10, 6))
    plt.bar(range(n_buckets), proportions, width=1, edgecolor='black')
    plt.xticks(range(n_buckets), [f'{i/n_buckets}-{(i+1)/n_buckets}' for i in range(n_buckets)], rotation=45)
    plt.xlabel(predictions)
    plt.ylabel('P(label=1)')
    plt.title(f'Calibration of {predictions} for prompt {prompt}')
    plt.show()

# for i in range(13):
#     calibration_graph(data, prompt=i)
calibration_graph(data, n_buckets=20)