import streamlit as st
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from rouge_score import rouge_scorer
import sacrebleu
import torch

# Initialize Streamlit page configuration
st.set_page_config(layout="wide")

# Define summarization models and their respective tokenizers
summarization_models = {
    "BERT": {"model": "mrm8488/bert-mini2bert-mini-finetuned-cnn_daily_mail-summarization", "tokenizer": "mrm8488/bert-mini2bert-mini-finetuned-cnn_daily_mail-summarization"},
    "PEGASUS": {"model": "google/pegasus-cnn_dailymail", "tokenizer": "google/pegasus-cnn_dailymail"},
    "T5": {"model": "BeenaSamuel/t5_cnn_daily_mail_abstractive_summarizer_v2", "tokenizer": "BeenaSamuel/t5_cnn_daily_mail_abstractive_summarizer_v2"},
    "BART": {"model": "facebook/bart-large-cnn", "tokenizer": "facebook/bart-large-cnn"},
    "RoBERTa": {"model": "google/roberta2roberta_L-24_cnn_daily_mail", "tokenizer": "google/roberta2roberta_L-24_cnn_daily_mail"},

}

# Function to load summarization model and tokenizer
def load_summarization_model(model_name):
    model = AutoModelForSeq2SeqLM.from_pretrained(summarization_models[model_name]["model"])
    tokenizer = AutoTokenizer.from_pretrained(summarization_models[model_name]["tokenizer"])
    return model, tokenizer

# Function to summarize text
def summarize_text(model, tokenizer, text):
    input_ids = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(input_ids, max_length=150, min_length=30, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Function to compute evaluation scores
def compute_scores(reference, summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference, summary)
    reference_tokens = [reference.split()]
    summary_tokens = summary.split()
    bleu_score = sacrebleu.corpus_bleu(summary_tokens, reference_tokens).score
    return rouge_scores, bleu_score

# Streamlit app
st.sidebar.subheader("Select your model")
choice = st.sidebar.selectbox("Summarization Model", list(summarization_models.keys()))

st.subheader(f"Summarizing Text Using {choice}")
input_text = st.text_area("Enter your text here")

if input_text:
    model, tokenizer = load_summarization_model(choice)

    custom_reference = st.text_area("Enter your custom reference summary")

    if st.button("Summarize Text"):
        summary = summarize_text(model, tokenizer, input_text)
        st.success(summary)

        # Compute evaluation scores based on custom reference summary
        rouge_scores, bleu_score = compute_scores(custom_reference, summary)
        st.write("ROUGE-1 F1 Score:", rouge_scores['rouge1'].fmeasure)
        st.write("ROUGE-L F1 Score:", rouge_scores['rougeL'].fmeasure)
        st.write("BLEU Score:", bleu_score)
