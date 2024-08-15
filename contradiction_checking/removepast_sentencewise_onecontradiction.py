import os
import sys

import nltk
import numpy as np
import pandas as pd

from datasets import load_metric
from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification, AutoTokenizer, pipeline
import sys
sys.path.append('../')

from utils.functions import predict_nli, remove_past_sentences

consultation_name = sys.argv[1]
model_checkpoint = sys.argv[2]
model_revision = sys.argv[3]

model_name = model_checkpoint.split("/")[-1]

labeled_proposals_couples = pd.read_csv(f"../consultation_data/nli_labeled_proposals_{consultation_name}.csv", encoding="utf8",
                                        engine='python', quoting=0, sep=';', dtype={"label": int})

pos_model_path = "ANONYMIZED, SHOULD BE RE-CREATED" #TODO: USE pos_model_creation.py to re-create the POS tagger

sentences_tokenizer = nltk.data.load("tokenizers/punkt/french.pickle")
pos_model = AutoModelForTokenClassification.from_pretrained(pos_model_path)
pos_tokenizer = AutoTokenizer.from_pretrained(pos_model_path)
nlp_token_class = pipeline('token-classification', model=pos_model, tokenizer=pos_tokenizer)

try:
    nli_model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, revision=model_revision)
    nli_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, revision=model_revision, model_max_length=512)
except OSError as error:
    print(f"No such revision '{model_revision}' for model '{model_name}'")
    quit()

exp_id = model_checkpoint[9:]
accuracy_metric = load_metric("accuracy", experiment_id=exp_id)
f1_metric = load_metric("f1", experiment_id=exp_id)


if not os.path.exists(f"../results/contradiction_checking/{consultation_name}/{model_name}{('_' + model_revision) if model_revision != 'main' else ''}"):
    os.mkdir(f"../results/contradiction_checking/{consultation_name}/{model_name}{('_' + model_revision) if model_revision != 'main' else ''}")

labeled_proposals_couples["premise"] = labeled_proposals_couples["premise"].apply(lambda proposal: remove_past_sentences(proposal, sentences_tokenizer, nlp_token_class))
labeled_proposals_couples["hypothesis"] = labeled_proposals_couples["hypothesis"].apply(lambda proposal: remove_past_sentences(proposal, sentences_tokenizer, nlp_token_class))
labeled_proposals_couples["predicted_label"] = np.nan

with open(f"../results/contradiction_checking/{consultation_name}/{model_name}{('_' + model_revision) if model_revision != 'main' else ''}/removepast_sentencewise_onecontradiction.log", "w", encoding="utf8") as file:
    for idx, row in labeled_proposals_couples.iterrows():
        premise_sentences = sentences_tokenizer.tokenize(row["premise"])
        hypothesis_sentences = sentences_tokenizer.tokenize(row["hypothesis"])

        has_contradictory_pairs = False

        for i in range(len(premise_sentences)):
            for j in range(len(hypothesis_sentences)):
                predicted_label = predict_nli(premise_sentences[i], hypothesis_sentences[j], nli_tokenizer, nli_model)
                if predicted_label:
                    has_contradictory_pairs = True
                    break
            if has_contradictory_pairs:
                break

        labeled_proposals_couples.at[idx, "predicted_label"] = int(has_contradictory_pairs)

        if idx % 5 == 0:
            file.write(f'{row["premise"]}\n\n')
        file.write(f'Label: {row["label"]};Has contradictory pairs: {has_contradictory_pairs};{row["hypothesis"]}\n')

        if idx % 5 == 4:
            file.write("===========================================\n\n")

with open(f"../results/contradiction_checking/{consultation_name}/{model_name}{('_' + model_revision) if model_revision != 'main' else ''}/removepast_sentencewise_onecontradiction_metrics.log", "w", encoding="utf8") as file:
    predictions = labeled_proposals_couples["predicted_label"].tolist()
    labels = labeled_proposals_couples["label"].tolist()
    file.write("Accuracy: ")
    file.write(str(accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]))
    file.write("\nF1 micro: ")
    file.write(str(f1_metric.compute(predictions=predictions, references=labels, average="micro")["f1"]))
    file.write("\nF1 macro: ")
    file.write(str(f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"]))
