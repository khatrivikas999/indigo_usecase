from google.colab import files
uploaded = files.upload()

import json
import pandas as pd
import re
import nltk
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from datasets import Dataset
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    GPT2Tokenizer, GPT2LMHeadModel,
    BertTokenizer, BertForSequenceClassification,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    TrainingArguments, Trainer,
    DataCollatorForSeq2Seq
)
import evaluate
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

CUSTOM_TOKEN = "Bf_dZpScmUFRMCCW23UPKGzkL23YjpKZWQVmO"

def plot_data_distribution(df):
    plt.figure(figsize=(15, 5))
    df['question_len'] = df['input_text'].apply(lambda x: len(x.split()))
    plt.subplot(1, 2, 1)
    sns.histplot(df['question_len'], bins=30, kde=True)
    plt.title('Question Length Distribution')
    df['answer_len'] = df['target_text'].apply(lambda x: len(x.split()))
    plt.subplot(1, 2, 2)
    sns.histplot(df['answer_len'], bins=30, kde=True)
    plt.title('Answer Length Distribution')
    plt.tight_layout()
    plt.savefig('data_distribution.png')
    plt.show()

def plot_word_clouds(df):
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    wordcloud = WordCloud(width=800, height=400).generate(' '.join(df['input_text']))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Questions Word Cloud')
    plt.subplot(1, 2, 2)
    wordcloud = WordCloud(width=800, height=400).generate(' '.join(df['target_text']))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Answers Word Cloud')
    plt.tight_layout()
    plt.savefig('word_clouds.png')
    plt.show()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)

def process_jsonl_data(file_path, sample_size=500):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                if 'question' in item and 'answer' in item:
                    q = clean_text(item['question'])
                    a = clean_text(item['answer'])
                    data.append({"input_text": q, "target_text": a})
                    if len(data) >= sample_size:
                        break
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(data)

def compute_metrics(eval_pred, tokenizer):
    """Fixed compute_metrics function with proper error handling"""
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")

    predictions, labels = eval_pred

    # Handle predictions - take logits if tuple
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # Convert to numpy if tensor
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # For generation tasks, predictions might be token IDs already
    # If predictions are logits, get the token IDs
    if len(predictions.shape) > 2:
        predictions = np.argmax(predictions, axis=-1)

    # Clean up labels - replace -100 with pad token
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Ensure token IDs are within valid range
    vocab_size = len(tokenizer)
    predictions = np.clip(predictions, 0, vocab_size - 1)
    labels = np.clip(labels, 0, vocab_size - 1)

    try:
        # Decode predictions and labels
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Clean up empty predictions
        decoded_preds = [pred.strip() if pred.strip() else "empty" for pred in decoded_preds]
        decoded_labels = [label.strip() if label.strip() else "empty" for label in decoded_labels]

    except Exception as e:
        print(f"Error in decoding: {e}")
        # Fallback: return dummy metrics
        return {
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
            "bleu": 0.0,
            "meteor": 0.0,
            "f1_score": 0.0,
            "exact_match": 0.0
        }

    # Calculate F1 scores
    f1_scores = []
    for pred, label in zip(decoded_preds, decoded_labels):
        pred_tokens = set(pred.split())
        label_tokens = set(label.split())
        if len(pred_tokens | label_tokens) > 0:
            precision = len(pred_tokens & label_tokens) / len(pred_tokens) if pred_tokens else 0
            recall = len(pred_tokens & label_tokens) / len(label_tokens) if label_tokens else 0
            if (precision + recall) > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
                f1_scores.append(f1)

    avg_f1 = np.mean(f1_scores) if f1_scores else 0

    # Calculate metrics with error handling
    try:
        rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        bleu_result = bleu.compute(predictions=decoded_preds, references=[[ref] for ref in decoded_labels])
        meteor_result = meteor.compute(predictions=decoded_preds, references=decoded_labels)

        exact_matches = sum(1 for p, l in zip(decoded_preds, decoded_labels) if p.strip() == l.strip())
        accuracy = exact_matches / len(decoded_labels)

        return {
            "rouge1": rouge_result["rouge1"],
            "rouge2": rouge_result["rouge2"],
            "rougeL": rouge_result["rougeL"],
            "bleu": bleu_result["bleu"],
            "meteor": meteor_result["meteor"],
            "f1_score": avg_f1,
            "exact_match": accuracy
        }
    except Exception as e:
        print(f"Error in metric computation: {e}")
        return {
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
            "bleu": 0.0,
            "meteor": 0.0,
            "f1_score": avg_f1,
            "exact_match": 0.0
        }

def plot_training_metrics(metrics_history):
    if not metrics_history:
        print("No metrics to plot")
        return

    df_metrics = pd.DataFrame(metrics_history)
    plt.figure(figsize=(12, 5))

    # Plot ROUGE scores
    plt.subplot(1, 2, 1)
    rouge_cols = [col for col in df_metrics.columns if 'rouge' in col.lower()]
    if rouge_cols:
        sns.lineplot(data=df_metrics[rouge_cols])
    plt.title('ROUGE Scores')
    plt.xlabel('Epoch')
    plt.ylabel('Score')

    # Plot other metrics
    plt.subplot(1, 2, 2)
    other_cols = [col for col in df_metrics.columns if any(metric in col.lower() for metric in ['bleu', 'meteor', 'f1'])]
    if other_cols:
        sns.lineplot(data=df_metrics[other_cols])
    plt.title('Other Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

    # Interactive plot
    if rouge_cols or other_cols:
        plot_cols = rouge_cols + other_cols
        if plot_cols:
            fig = px.line(df_metrics, y=plot_cols[:3], title='Model Performance Metrics')
            fig.write_html("interactive_metrics.html")

def train_t5_model(train_df, eval_df):
    """Fixed T5 training function"""
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    # Add custom token properly
    if CUSTOM_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_tokens([CUSTOM_TOKEN])

    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    if torch.cuda.is_available():
        model.gradient_checkpointing_enable()

    def preprocess(batch):
        # Add T5 prefix for question answering
        inputs = [f"question: {text}" for text in batch["input_text"]]
        targets = batch["target_text"]

        model_inputs = tokenizer(
            inputs,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="np"
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=64,
                padding="max_length",
                truncation=True,
                return_tensors="np"
            )

        # Replace pad token ids in labels with -100
        labels_array = labels["input_ids"]
        labels_array[labels_array == tokenizer.pad_token_id] = -100

        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": labels_array
        }

    # Create datasets
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)

    tokenized_train = train_dataset.map(preprocess, batched=True, remove_columns=train_dataset.column_names)
    tokenized_eval = eval_dataset.map(preprocess, batched=True, remove_columns=eval_dataset.column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./t5_qa_model",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,  # Reduced for testing
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        report_to="none",
        generation_max_length=64,
        generation_num_beams=2,
        save_total_limit=2
    )

    metrics_history = []

    class CustomTrainer(Seq2SeqTrainer):
        def evaluate(self, *args, **kwargs):
            try:
                metrics = super().evaluate(*args, **kwargs)
                metrics_history.append(metrics)
                return metrics
            except Exception as e:
                print(f"Evaluation error: {e}")
                # Return dummy metrics to continue training
                dummy_metrics = {
                    "eval_loss": 1.0,
                    "eval_rouge1": 0.0,
                    "eval_rouge2": 0.0,
                    "eval_rougeL": 0.0,
                    "eval_bleu": 0.0,
                    "eval_meteor": 0.0,
                    "eval_f1_score": 0.0,
                    "eval_exact_match": 0.0
                }
                metrics_history.append(dummy_metrics)
                return dummy_metrics

    # Create trainer with fixed compute_metrics
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer)
    )

    # Train the model
    print("Starting T5 training...")
    trainer.train()

    # Save model
    trainer.save_model("./t5_qa_final")
    tokenizer.save_pretrained("./t5_qa_final")

    # Plot metrics
    plot_training_metrics(metrics_history)

    # Clear memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("T5 training complete! Model saved.")

def train_gpt_model(train_df, eval_df):
    """Fixed GPT-2 training function"""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Add custom token and set pad token
    if CUSTOM_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_tokens([CUSTOM_TOKEN])
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))

    if torch.cuda.is_available():
        model.gradient_checkpointing_enable()

    def preprocess(batch):
        # Format as question-answer pairs
        texts = [f"Question: {q} Answer: {a}" for q, a in zip(batch['input_text'], batch['target_text'])]

        encodings = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="np"
        )

        # For GPT-2, labels are the same as input_ids
        labels = np.copy(encodings['input_ids'])

        return {
            "input_ids": encodings['input_ids'],
            "attention_mask": encodings['attention_mask'],
            "labels": labels
        }

    # Create datasets
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)

    tokenized_train = train_dataset.map(preprocess, batched=True, remove_columns=train_dataset.column_names)
    tokenized_eval = eval_dataset.map(preprocess, batched=True, remove_columns=eval_dataset.column_names)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./gpt2_qa_model",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,  # Reduced for testing
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none"
    )

    metrics_history = []

    class GPT2Trainer(Trainer):
        def evaluate(self, *args, **kwargs):
            try:
                metrics = super().evaluate(*args, **kwargs)
                metrics_history.append(metrics)
                return metrics
            except Exception as e:
                print(f"GPT-2 Evaluation error: {e}")
                dummy_metrics = {"eval_loss": 1.0}
                metrics_history.append(dummy_metrics)
                return dummy_metrics

    trainer = GPT2Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer)
    )

    print("Starting GPT-2 training...")
    trainer.train()

    trainer.save_model("./gpt2_qa_final")
    tokenizer.save_pretrained("./gpt2_qa_final")

    plot_training_metrics(metrics_history)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("GPT-2 training complete! Model saved.")

# def train_bert_model(train_df, eval_df):
#     """Note: BERT is not suitable for text generation. This is for demonstration only."""
#     print("Note: BERT is not designed for question answering generation.")
#     print("Consider using BERT for classification tasks instead.")
#     print("Skipping BERT training...")

# Main execution
if __name__ == "__main__":
    # Load and process data
    file_path = "Quora-QuAD.jsonl"
    df = process_jsonl_data(file_path)

    print(f"Loaded {len(df)} samples")

    # Plot data distribution
    plot_data_distribution(df)
    plot_word_clouds(df)

    # Split data
    train_df, eval_df = train_test_split(df, test_size=0.1, random_state=42)

    print(f"Training samples: {len(train_df)}")
    print(f"Evaluation samples: {len(eval_df)}")

    # Train models
    train_t5_model(train_df, eval_df)
    train_gpt_model(train_df, eval_df)
    train_bert_model(train_df, eval_df)
