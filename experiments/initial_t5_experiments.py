import argparse
import os
import shutil
import statistics
import torch

import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

from experiments.evaluation import macro_f1, weighted_f1, sentence_label_evaluation, print_evaluation
from experiments.label_converter import encode, decode
from experiments.t5_config import t5_args, SEED, TEMP_DIRECTORY
from t5.t5_model import T5Model

# from text_classification.text_classification_model import TextClassificationModel

parser = argparse.ArgumentParser(
    description='''evaluates multiple models  ''')
parser.add_argument('--model_name', required=False, help='model name', default="google/byt5-large")
parser.add_argument('--model_type', required=False, help='model type', default="byt5")
parser.add_argument('--cuda_device', required=False, help='cuda device', default=0)

arguments = parser.parse_args()

MODEL_TYPE = arguments.model_type
MODEL_NAME = arguments.model_name
cuda_device = int(arguments.cuda_device)

data_df = pd.read_json('data/rtl_comments_clean_2022-10.json')
data_df_clean = data_df[data_df.status.isin(["Published", "Archived"])]
data_df_clean['date_created'] = pd.to_datetime(data_df_clean['date_created'])

# Extract year and use 2022 data to create a test set.
data_df_clean['year'] = data_df_clean['date_created'].dt.year

train = data_df_clean[(data_df_clean["year"] != 2022)]
test = data_df_clean[(data_df_clean["year"] == 2022)]

train = train.rename(columns={'text': 'input_text', 'status': 'target_text'})
train = train[['input_text', 'target_text']]
train["prefix"] = ""

test = test.rename(columns={'status': 'labels'})
test = test[['text', 'labels']]

test_sentences = []
for index, row in test.iterrows():
    test_sentences.append("" + row['text'])

test_preds = np.empty((len(test_sentences), t5_args["n_fold"]))
macro_f1_scores = []
weighted_f1_scores = []

for i in range(t5_args["n_fold"]):
    if os.path.exists(t5_args['output_dir']) and os.path.isdir(t5_args['output_dir']):
        shutil.rmtree(t5_args['output_dir'])

    print("Started Fold {}".format(i))
    model = T5Model(MODEL_TYPE, MODEL_NAME, args=t5_args,
                                use_cuda=torch.cuda.is_available())
    train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED * i)

    model.train_model(train_df, eval_data=eval_df)

    preds = model.predict(test_sentences)
    test_preds[:, i] = preds
    macro_f1, weighted_f1 = sentence_label_evaluation(preds, test['labels'].tolist())
    macro_f1_scores.append(macro_f1)
    weighted_f1_scores.append(weighted_f1)


print("Weighted F1 scores ", weighted_f1_scores)
print("Mean weighted F1 scores", statistics.mean(weighted_f1_scores))
print("STD weighted F1 scores", statistics.stdev(weighted_f1_scores))

print("Macro F1 scores ", macro_f1_scores)
print("Mean macro F1 scores", statistics.mean(macro_f1_scores))
print("STD macro F1 scores", statistics.stdev(macro_f1_scores))

# select majority class of each instance (row)
test_predictions = []
for row in test_preds:
    row = row.tolist()
    test_predictions.append(max(set(row), key=row.count))

test["predictions"] = test_predictions


print_evaluation(test, "predictions", "labels")
test.to_csv("results_t5.tsv", sep='\t', encoding='utf-8', index=False)

