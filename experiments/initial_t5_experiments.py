import argparse
import os
import shutil
import statistics
import torch

import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from datasets import Dataset
from datasets import load_dataset
from config.model_args import T5Args
from experiments.evaluation import sentence_label_evaluation, print_evaluation
from experiments.label_converter import encode, decode
from t5.t5_model import T5Model


parser = argparse.ArgumentParser(
    description='''evaluates multiple models  ''')
parser.add_argument('--model_name', required=False, help='model name', default="google/mt5-base")
parser.add_argument('--model_type', required=False, help='model type', default="mt5")
parser.add_argument('--cuda_device', required=False, help='cuda device', default=0)

arguments = parser.parse_args()

MODEL_TYPE = arguments.model_type
MODEL_NAME = arguments.model_name
cuda_device = int(arguments.cuda_device)

model_args = T5Args()
model_args.num_train_epochs = 10
model_args.no_save = False
model_args.fp16 = False
model_args.learning_rate = 1e-4
model_args.train_batch_size = 8
model_args.max_seq_length = 256
model_args.evaluate_generated_text = True
model_args.evaluate_during_training = True
model_args.evaluate_during_training_verbose = True
model_args.evaluate_during_training_steps = 10000
model_args.use_multiprocessing = False
model_args.use_multiprocessing_for_evaluation = False
model_args.use_multiprocessed_decoding = False
model_args.overwrite_output_dir = True
model_args.save_recent_only = True
model_args.logging_steps = 10000
model_args.manual_seed = 777
model_args.early_stopping_patience = 25
model_args.save_steps = 10000

t5_args = model_args
n_fold = 1

model_representation = MODEL_NAME.replace('/', '-')
model_args.output_dir = os.path.join("outputs", model_representation)
model_args.best_model_dir = os.path.join("outputs", model_representation, "best_model")
model_args.cache_dir = os.path.join("cache_dir", model_representation)

model_args.wandb_project = "LUX Comment Moderation"
model_args.wandb_kwargs = {"name": MODEL_NAME}

train = Dataset.to_pandas(load_dataset('instilux/lb-rtl-comments_clas', split='train'))
test = Dataset.to_pandas(load_dataset('instilux/lb-rtl-comments_clas', split='test'))

train = train[train['text'].notna()]
test = test[test['text'].notna()]

train = train.rename(columns={'text': 'input_text', 'label': 'target_text'})
train = train[['input_text', 'target_text']]
train["prefix"] = ""

test = test.rename(columns={'label': 'labels'})
test = test[['text', 'labels']]

test_sentences = []
for index, row in test.iterrows():
    test_sentences.append("" + row['text'])

test_preds = np.zeros((len(test_sentences), n_fold))
macro_f1_scores = []
weighted_f1_scores = []

for i in range(n_fold):
    if os.path.exists(model_args.output_dir) and os.path.isdir(model_args.output_dir):
        shutil.rmtree(model_args.output_dir)

    print("Started Fold {}".format(i))
    model = T5Model(MODEL_TYPE, MODEL_NAME, args=t5_args,
                                use_cuda=torch.cuda.is_available())
    train_df, eval_df = train_test_split(train, test_size=0.1, random_state=model_args.manual_seed * i)

    model.train_model(train_df, eval_data=eval_df)

    preds = model.predict(test_sentences)
    macro_f1, weighted_f1 = sentence_label_evaluation(preds, test['labels'].tolist())
    macro_f1_scores.append(macro_f1)
    weighted_f1_scores.append(weighted_f1)
    preds = encode(preds)
    test_preds[:, i] = preds


print("Weighted F1 scores ", weighted_f1_scores)
# print("Mean weighted F1 scores", statistics.mean(weighted_f1_scores))
# print("STD weighted F1 scores", statistics.stdev(weighted_f1_scores))

print("Macro F1 scores ", macro_f1_scores)
# print("Mean macro F1 scores", statistics.mean(macro_f1_scores))
# print("STD macro F1 scores", statistics.stdev(macro_f1_scores))

# select majority class of each instance (row)
test_predictions = []
for row in test_preds:
    row = row.tolist()
    test_predictions.append(int(max(set(row), key=row.count)))

test["predictions"] = decode(test_predictions)

print_evaluation(test, "predictions", "labels")
test.to_csv("results_byt5_base.tsv", sep='\t', encoding='utf-8', index=False)

