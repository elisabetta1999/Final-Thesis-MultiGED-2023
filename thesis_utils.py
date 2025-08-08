from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, set_seed
from transformers import DataCollatorForTokenClassification
import pandas as pd
from datasets import Dataset
from datasets import Sequence, ClassLabel, Features, Value
from evaluate import load
import random
import numpy as np
import csv
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support

def read_tsv_file_and_find_sentences_without_headers(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        sentence = []
        sentences = []
        for line in file:
            line = line.strip()
            if line.startswith('#'):
                continue
            if line:
                sentence.append(line.split('\t'))
            else:
                if sentence:
                    sentences.append(sentence)
                    sentence = []
        if sentence:
            sentences.append(sentence)
    return sentences

def read_tsv_file_and_find_sentences_with_headers(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        sentence = []
        sentences = []
        header_skipped = False

        for line in file:
            line = line.strip()
            if line.startswith('#') or not line:
                if not line and sentence:
                    sentences.append(sentence)
                    sentence = []
                continue

            if not header_skipped:
                header_skipped = True
                continue
            sentence.append(line.split('\t'))
        if sentence:
            sentences.append(sentence)
    return sentences


def token_counter(list_sentences):
    token_counter = 0
    for sentence in list_sentences:
        for line in sentence:
            token_counter+=1
    return token_counter

def get_list_ids_tokens_gold_finetuning(list_sentences, is_preprocessed=False, is_test=False):
    all_sentences_ids = []
    all_sentences_tokens = []
    all_sentences_gold = []

    for idx, sentence in enumerate(list_sentences):
        sentence_tok_ids = []
        sentence_tokens = []
        sentence_gold = []

        for idx_tok, line in enumerate(sentence):
            if is_preprocessed:
                sentence_tok_ids.append(idx_tok + 1)
                sentence_tokens.append(line[1])
                sentence_gold.append(line[-1])
            if is_preprocessed == False:
                if is_test:
                    sentence_tok_ids.append(idx_tok + 1)
                    sentence_tokens.append(line[0])
                    sentence_gold.append('c')
                else:    
                    sentence_tok_ids.append(idx_tok + 1)
                    sentence_tokens.append(line[0])
                    sentence_gold.append(line[-1])

        all_sentences_ids.append(sentence_tok_ids)
        all_sentences_tokens.append(sentence_tokens)
        all_sentences_gold.append(sentence_gold)

    return all_sentences_ids, all_sentences_tokens, all_sentences_gold


def get_list_ids_tokens_gold (list_sentences):
    all_sentences_info = []
    
    for idx, sentence in enumerate(list_sentences):
        sent_info = []
        for idx_tok, line in enumerate(sentence):
            tok_tuple = (f'T{idx_tok}',line[0],line[-1])
            sent_info.append(tok_tuple)
            
        all_sentences_info.append(sent_info)
    
    return all_sentences_info

def create_list_dict(all_sentences_ids, all_sentences_tokens, all_sentences_labels):
    all_sentences_list_dict = []
    for id_token, sentence, label in zip(all_sentences_ids, all_sentences_tokens, all_sentences_labels):
        feature_dict = { 
            'id_token': id_token,
            'token': sentence,
            'labels': label,
        }
        all_sentences_list_dict.append(feature_dict)
    return all_sentences_list_dict


def map_predictions_to_words_and_save_to_file(predictions, labels, tokenized_data, output_file_path, label_list, tokenizer):
    with open(output_file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(['word', 'prediction', 'gold_label'])
        
        new_tokens, new_labels, new_predictions = [], [], []
        
        input_ids = tokenized_data["input_ids"]
        word_ids_list = tokenized_data["t_word_id"]

        for idx in range(len(predictions)):
            prediction, label, word_ids = predictions[idx], labels[idx], word_ids_list[idx]
            
            word_predictions, word_labels = {}, {}

            for token_idx, word_idx in enumerate(word_ids):
                if word_idx is None:
                    continue

                if word_idx not in word_predictions:
                    word_predictions[word_idx] = []
                word_predictions[word_idx].append(prediction[token_idx])

                if label[token_idx] != -100:
                    word_labels[word_idx] = label[token_idx]

            for word_idx in sorted(word_predictions.keys()):
                predicted_label = label_list[Counter(word_predictions[word_idx]).most_common(1)[0][0]]

                gold_label = label_list[word_labels.get(word_idx, 0)]

                word_tokens = [
                    tokenizer.convert_ids_to_tokens(input_ids[idx][i]).strip()
                    for i, w_idx in enumerate(word_ids) if w_idx == word_idx
                ]
                word = tokenizer.convert_tokens_to_string(word_tokens).strip()

                new_tokens.append(word)
                new_labels.append(gold_label)
                new_predictions.append(predicted_label)

                writer.writerow([word, predicted_label, gold_label])

        print(f"Results saved to {output_file_path}")

    return new_tokens, new_labels, new_predictions

def map_predictions_to_words_and_save_to_file_sentence(predictions, labels, tokenized_data, output_file_path, label_list, tokenizer):
    with open(output_file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(['word', 'prediction', 'gold_label'])
        
        all_tokens, all_labels, all_predictions = [], [], []

        input_ids = tokenized_data["input_ids"]
        word_ids_list = tokenized_data["t_word_id"]

        for idx in range(len(predictions)):
            prediction = predictions[idx]
            label = labels[idx]
            word_ids = word_ids_list[idx]

            word_predictions, word_labels = {}, {}

            for token_idx, word_idx in enumerate(word_ids):
                if word_idx is None:
                    continue
                word_predictions.setdefault(word_idx, []).append(prediction[token_idx])
                if label[token_idx] != -100:
                    word_labels[word_idx] = label[token_idx]

            sent_tokens, sent_labels, sent_predictions = [], [], []

            for word_idx in sorted(word_predictions.keys()):
                pred_id = Counter(word_predictions[word_idx]).most_common(1)[0][0]
                predicted_label = label_list[pred_id]
                gold_id = word_labels.get(word_idx, 0)
                gold_label = label_list[gold_id]

                word_tokens = [
                    tokenizer.convert_ids_to_tokens(input_ids[idx][i]).strip()
                    for i, w_idx in enumerate(word_ids) if w_idx == word_idx
                ]
                word = tokenizer.convert_tokens_to_string(word_tokens).strip()

                sent_tokens.append(word)
                sent_labels.append(gold_label)
                sent_predictions.append(predicted_label)

                writer.writerow([word, predicted_label, gold_label])

            writer.writerow([])

            all_tokens.append(sent_tokens)
            all_labels.append(sent_labels)
            all_predictions.append(sent_predictions)

        print(f"Results saved to {output_file_path}")

    return all_tokens, all_labels, all_predictions

import pandas as pd

def add_predictions_to_tsv_with_empty_lines(file_path, proc_predictions, output_path=None):
    """
    Aggiunge una colonna di predizioni a un file TSV con frasi separate da righe vuote,
    mantenendo la struttura originale con righe vuote tra frasi.

    Args:
        file_path (str): path del file TSV di input.
        proc_predictions (list of list): nested list di predizioni per frase.
        output_path (str, optional): path del file di output.
            Se None, sovrascrive file_path.

    """
    if output_path is None:
        output_path = file_path

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    header = lines[0].strip().split('\t')
    data_lines = lines[1:]

    sentences = []
    current = []

    for line in data_lines:
        if line.strip() == "":
            if current:
                sentences.append(current)
                current = []
        else:
            current.append(line.strip().split('\t'))
    if current:
        sentences.append(current)

    flat_data = [row for sentence in sentences for row in sentence]
    df = pd.DataFrame(flat_data, columns=header)

    flat_predictions = [pred for sentence in proc_predictions for pred in sentence]
    df["predictions"] = flat_predictions

    output_lines = ['\t'.join(df.columns) + '\n']
    i = 0
    for sentence in sentences:
        for _ in sentence:
            row = df.iloc[i]
            line = '\t'.join(map(str, row.values)) + '\n'
            output_lines.append(line)
            i += 1
        output_lines.append('\n')

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(output_lines)


from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_predictions(predictions, gold_labels, label_list=None):
    predictions_flat = predictions
    gold_labels_flat = gold_labels

    if label_list is None:
        label_list = sorted(set(gold_labels_flat).union(set(predictions_flat)))

    gold_labels_flat = [label_list.index(label) for label in gold_labels]
    predictions_flat = [label_list.index(label) for label in predictions]

    # Per-class scores
    precision, recall, fscore, support = precision_recall_fscore_support(
        gold_labels_flat,
        predictions_flat,
        beta=0.5,
        average=None,
        zero_division=0
    )

    report_lines = ["              precision    recall    f0.5-score    support"]
    for i, label in enumerate(label_list):
        report_lines.append(
            f"{label:15s} {precision[i]:.4f}    {recall[i]:.6f}    {fscore[i]:.5f}        {support[i]}"
        )

    # Macro average: mean of class-wise scores
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f05 = fscore.mean()

    # Micro average: computed globally
    micro_precision, micro_recall, micro_f05, _ = precision_recall_fscore_support(
        gold_labels_flat,
        predictions_flat,
        beta=0.5,
        average='micro',
        zero_division=0
    )

    report_lines.append("")
    report_lines.append(f"micro avg       {micro_precision:.4f}    {micro_recall:.6f}    {micro_f05:.5f}        {sum(support)}")
    report_lines.append(f"macro avg       {macro_precision:.4f}    {macro_recall:.6f}    {macro_f05:.5f}        {sum(support)}")

    print("\n".join(report_lines))

    # Confusion matrix
    cm = confusion_matrix(gold_labels_flat, predictions_flat)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_list, yticklabels=label_list)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()
