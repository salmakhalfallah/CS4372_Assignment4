import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import transformers as trf
import torch
from transformers import pipeline
from torchtext.data import bleu_score
from rouge_score import rouge_scorer
from tabulate import tabulate

# inputting and processing text data

# note to self: try to change the file path to global url
file = 'text_data.txt'

try:
    with open(file, mode='r', encoding='utf-8') as f:
        text = f.read()
except FileNotFoundError:
    print("File was not found. Please check the file path and try again.")

print("Text data hopefully loaded..")
print(text[:500])

# text summarization using transformer pipeline...

summary_pipeline = pipeline("summarization", model = "google/pegasus-xsum")

# outputting summary:

# note for tomorrow: tune max and min length if time allows

output = summary_pipeline(text, max_length = 240, min_length = 30, do_sample = False)
summary = output[0]['summary_text']
print("Summary:", summary)

# evaluating summary against original text using ROUGE metric
scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeLsum','rougeL'], use_stemmer = True)
scores_og = scorer.score(text, summary)

scores_og = pd.DataFrame.from_dict(scores_og)
index_mapping = {0: 'precision', 1: 'recall', 2: 'f-measure'}
scores_og = scores_og.rename(index = index_mapping)
print(scores_og)

# comparing with human-written summary
file = 'human_summary.txt'

try:
    with open(file, mode='r', encoding='utf-8') as f:
        human_summary = f.read()
except FileNotFoundError:
    print("File was not found. Please check the file path and try again.")

# outputting human summary scores
scores_human = scorer.score(human_summary, summary)
scores_human = pd.DataFrame.from_dict(scores_human)
scores_human = scores_human.rename(index = index_mapping)
print(scores_human)

headers = scores_human.columns.tolist()

print(tabulate(scores_og, headers = headers))

print(tabulate(scores_human, headers = headers))

# prepping data for visualization

scores_og['type'] = ['og', 'og', 'og']
scores_human['type'] = ['human', 'human', 'human']

# combining results for visualization
scores = pd.concat([scores_og, scores_human], ignore_index = False)

print(tabulate(scores, headers = scores.columns.tolist()))

numeric_cols = ["rouge1", "rouge2", "rougeLsum", "rougeL"]

scores_og[numeric_cols].T.plot(kind="bar", figsize=(12,8))
plt.title('ROUGE Scores for Generated Summary - Facebook Model (Original Text)')
plt.legend(title = 'Score Type', loc = 'upper right')
plt.xlabel('ROUGE Scores by Metric')
plt.ylabel('Scores')
plt.xticks(rotation = 0)
plt.show()

scores_human[numeric_cols].T.plot(kind="bar", figsize=(12,8))
plt.title('ROUGE Scores for Generated Summary - Facebook Model (Human Summary)')
plt.legend(title = 'Score Type', loc = 'upper right')
plt.xlabel('ROUGE Scores by Metric')
plt.ylabel('Scores')
plt.xticks(rotation = 0)
plt.show()