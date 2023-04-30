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
from experiments.year_transformer_config import transformer_args, SEED, TEMP_DIRECTORY
from text_classification.text_classification_model import TextClassificationModel

parser = argparse.ArgumentParser(
    description='''evaluates multiple models  ''')
parser.add_argument('--model_name', required=False, help='model name', default="xlm-roberta-large")
parser.add_argument('--model_type', required=False, help='model type', default="xlmroberta")
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

train = data_df_clean[(data_df_clean["year"] == 2015)]
test = data_df_clean[(data_df_clean["year"] == 2022)]

train = train.rename(columns={'status': 'labels'})
train = train[['text', 'labels']]

test = test.rename(columns={'status': 'labels'})
test = test[['text', 'labels']]

train['labels'] = encode(train["labels"])
test['labels'] = encode(test["labels"])


test_sentences = test['text'].tolist()

test_preds = np.zeros((len(test_sentences), transformer_args["n_fold"]))
macro_f1_scores = []
weighted_f1_scores = []

for i in range(transformer_args["n_fold"]):
    if os.path.exists(transformer_args['output_dir']) and os.path.isdir(transformer_args['output_dir']):
        shutil.rmtree(transformer_args['output_dir'])

    print("Started Fold {}".format(i))
    model = TextClassificationModel(MODEL_TYPE, MODEL_NAME, args=transformer_args,
                                use_cuda=torch.cuda.is_available())
    train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED * i)
    model.train_model(train_df, eval_df=eval_df, macro_f1=macro_f1, weighted_f1=weighted_f1,
                      accuracy=sklearn.metrics.accuracy_score)
    predictions, raw_outputs = model.predict(test_sentences)
    test_preds[:, i] = predictions
    macro_f1_score, weighted_f1_score = sentence_label_evaluation(predictions, test['labels'].tolist())
    macro_f1_scores.append(macro_f1_score)
    weighted_f1_scores.append(weighted_f1_score)

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
    test_predictions.append(int(max(set(row), key=row.count)))

test["predictions"] = test_predictions

test['predictions'] = decode(test["predictions"])
test['labels'] = decode(test["labels"])

print_evaluation(test, "predictions", "labels")
test.to_csv("results_2015_LuxemBERT.tsv", sep='\t', encoding='utf-8', index=False)

