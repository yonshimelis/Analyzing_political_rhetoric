#My work in the APP analysis code 

import os 


os.chdir(r"C:\Users\shime\OneDrive\Documents\GitHub\Final-Project-GroupJavaWockeez\Code")

print(os.getcwd())
#%%
import requests
from bs4 import BeautifulSoup
import csv
import time
from pathlib import Path

BASE_URL = "https://www.presidency.ucsb.edu"
LIST_PATH = "/documents/app-categories/statements"
LIST_URL = BASE_URL + LIST_PATH
OUTFILE = "presidential_statements.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; Scraper/1.0)"}
TEMP_SAVE_EVERY = 100   # flush every N records
DELAY_BETWEEN_REQUESTS = 0.35

def fetch(url, timeout=15, tries=3):
    for attempt in range(tries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as e:
            print(f"Fetch error ({attempt+1}/{tries}) for {url}: {e}")
            time.sleep(1)
    return None

def extract_detail_content(detail_url):
    r = fetch(detail_url)
    if not r:
        return "", "", ""
    s = BeautifulSoup(r.text, "html.parser")
    content_node = s.select_one("div.field-docs-content")
    content = content_node.get_text("\n", strip=True) if content_node else ""

    # categories (if present)
    cats = s.select("div.group-meta a, div.field-ds-filed-under- a, .field-ds-filed-under a")
    categories = ", ".join([c.get_text(strip=True) for c in cats]) if cats else ""

    # citation
    cit = s.select_one(".field-prez-document-citation, .ucsbapp_citation")
    citation = cit.get_text(" ", strip=True) if cit else ""

    return content, categories, citation

def read_existing_urls(outfile):
    p = Path(outfile)
    if not p.exists():
        return set()
    try:
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return {row.get("url","").strip() for row in reader if row.get("url")}
    except Exception as e:
        print("Error reading existing CSV, will start fresh:", e)
        return set()

def scrape_statements(max_pages=None):
    existing_urls = read_existing_urls(OUTFILE)
    page = 0
    total_saved = 0

    # prepare CSV writer (append mode)
    headers = ["title", "url", "president", "date", "content", "categories", "citation"]
    outfile_path = Path(OUTFILE)
    write_header = not outfile_path.exists()

    csvfile = open(outfile_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csvfile, 
                            fieldnames=headers,
                            quoting = csv.QUOTE_ALL,
                            escapechar='\\')
    if write_header:
        writer.writeheader()

    try:
        while True:
            if max_pages is not None and page >= max_pages:
                print("Reached max_pages limit, stopping.")
                break

            page_url = f"{LIST_URL}?page={page}"
            resp = fetch(page_url)
            if not resp:
                print("Failed to fetch listing page:", page_url)
                break

            soup = BeautifulSoup(resp.text, "html.parser")

            # listing item containers for statements
            items = soup.select("div.views-row, div.node-teaser, div.node-documents.node-teaser")
            # filter duplicates and ensure items have a link
            items = [it for it in items if it.select_one("a[href*='/documents/']")]

            if not items:
                print(f"No items found on page {page}. Stopping.")
                break

            num_items = len(items)
            print(f"Scraping page {page+1}: {num_items} statements (Total so far: {total_saved + num_items})")

            for item in items:
                # Title + link
                title_a = item.select_one(".field-title a, h3 a, a[href*='/documents/']")
                if not title_a:
                    print("Skipping item (no title link). snippet:", item.get_text(" ", strip=True)[:150])
                    continue
                title = title_a.get_text(strip=True)
                href = title_a.get("href", "").strip()
                full_link = href if href.startswith("http") else BASE_URL + href

                # skip if already scraped
                if full_link in existing_urls:
                    # print("Skipping already-saved:", full_link)
                    continue

                # president (the "Related" link on the right column)
                pres_a = item.select_one(".col-sm-4 a, .views-field-field-president a, .field-title ~ .col-sm-4 a")
                president = pres_a.get_text(strip=True) if pres_a else ""

                # date
                date_span = item.select_one("span.date-display-single, .views-field-field-docs-date span, .views-field-created span")
                date = date_span.get("content", date_span.get_text(strip=True)) if date_span else ""

                # fetch detail content
                content, categories, citation = extract_detail_content(full_link)

                row = {
                    "title": title,
                    "url": full_link,
                    "president": president,
                    "date": date,
                    "content": content,
                    "categories": categories,
                    "citation": citation
                }

                writer.writerow(row)
                existing_urls.add(full_link)
                total_saved += 1

                if total_saved % TEMP_SAVE_EVERY == 0:
                    csvfile.flush()
                    print(f"Checkpoint: saved {total_saved} records so far.")

                time.sleep(DELAY_BETWEEN_REQUESTS)

            page += 1
            # small delay between pages
            time.sleep(0.5)

    finally:
        csvfile.close()

    print("Finished. Total new records saved:", total_saved)
    return total_saved

if __name__ == "__main__":
    # For a quick test set max_pages=2
    # For full run use max_pages=None
    scrape_statements(max_pages=None)

#%%#Original scraper had issues with formatting columns, so we reformat here

import pandas as pd

df_final = pd.read_csv("presidential_statements.csv", header=None)

df_final.columns = [
    "title",
    "url",
    "president",
    "date",
    "content",
    "categories",
    "citation"
]

df_final.to_csv("presidential_statements_scraped.csv", index=False)
#%%
#Read in the CSV file
import pandas as pd

dataset = pd.read_csv("presidential_statements_scraped.csv")

#Inspect the dataset
print(dataset.shape)

#Print the first few rows
print(dataset.head())

#Check columns
print(dataset.columns)
#%%
#Add party affiliation based on president name
#Make copy of df to keep original intact
dataset_raw = dataset.copy()

'''From this point on we will manipulate dataset and keep dataset_raw as original'''


#Check dataset again
print(dataset.columns)

#Check presidents in dataset
print(dataset['president'].unique())
print(len(dataset['president'].unique()))

#Add party affiliation
party_affiliation = { "Donald J. Trump (1st Term)" : "Republican",
                        "Donald J. Trump (2nd Term)" : "Republican",
                        "Joseph R. Biden, Jr.": "Democrat",
                        "Barack Obama": "Democrat",
                        "George W. Bush": "Republican",
                        "William J. Clinton": "Democrat",
                        "George Bush": "Republican",
                        "Ronald Reagan": "Republican",
                        "Jimmy Carter": "Democrat",
                        "Gerald R. Ford": "Republican",
                        "Richard Nixon": "Republican",
                        "Lyndon B. Johnson": "Democrat",
                        "John F. Kennedy": "Democrat",
                        "Dwight D. Eisenhower": "Republican",
                        "Harry S Truman": "Democrat",
                        "Franklin D. Roosevelt": "Democrat",
                        "Herbert Hoover": "Republican", 
                        "Calvin Coolidge": "Republican",
                        "Warren G. Harding": "Republican",
                        "Woodrow Wilson": "Democrat",
                        "William Howard Taft": "Republican",
                        "Theodore Roosevelt": "Republican",
                        "William McKinley": "Republican",
                        "Grover Cleveland": "Democrat",
                        "Benjamin Harrison": "Republican",
                        "Chester A. Arthur": "Republican",
                        "James A. Garfield": "Republican",
                        "Rutherford B. Hayes": "Republican",
                        "Ulysses S. Grant": "Republican",
                        "Andrew Johnson": "Democrat",
                        "Abraham Lincoln": "Republican",
                        "James Buchanan": "Democrat",
                        "Franklin Pierce": "Democrat",
                        "Millard Fillmore": "Whig",
                        "Zachary Taylor": "Whig",
                        "James K. Polk": "Democrat",
                        "John Tyler": "Whig",
                        "William Harrison": "Whig",
                        "Martin Van Buren": "Democrat",
                        "Andrew Jackson": "Democrat",
                        "John Quincy Adams": "National Republican",
                        "James Monroe": "Democrat-Republican",
                        "James Madison": "Democrat-Republican",
                        "Thomas Jefferson": "Democrat-Republican",
                        "John Adams": "Federalist",
                        "George Washington": "Federalist"
                        }

#Check if we have all presidents listed
print(len(party_affiliation))

#Map party affiliation to df
dataset['party'] = dataset['president'].map(party_affiliation)

#Check if it correctly mapped
print(dataset[['president', 'party']].drop_duplicates().sort_values(by='president'))

'''Prior to the two party system we know, there were other parties such as Whig, Federalist, National Republican, and Democrat-Republican. We will keep these as is for now.'''

#Remove gavin newsom speeches if any since he is not a president
dataset = dataset[dataset['president'] != 'Gavin Newsom']

#Check dataset again to see if drop worked
"Gavin Newsom" in dataset['president'].unique()
#%%
#Check value counts for each president in descending order

print(dataset['president'].value_counts().sort_values(ascending=False))

#%%

#Preprocess text

#Import necessary libraries
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords') # Download if needed
from nltk.tokenize import word_tokenize
nltk.download('punkt') #Download if needed
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet') #Download if needed

#Define stopwords
stopwords = set(stopwords.words('english'))

#Create function to preprocess text
def preprocess(text): 
    text = text.lower()  # Lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'&[a-z]+;', ' ', text)  # Remove HTML entities
    text = re.sub(r"[^a-z\s']", ' ', text)  # Remove punctuation and special characters
    text = re.sub(r'\s+', ' ',text).strip()  # Remove extra whitespace
    text = re.sub (r'\d+', '', text)  # Remove numbers
    tokens = word_tokenize(text)  # Tokenization
    tokens = [word for word in tokens if word not in stopwords]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    return ' '.join(tokens)  # Join tokens back to string

#Apply to text 
dataset['cleaned_content'] = dataset['content'].apply(preprocess)

#Check df 
print(dataset[['content', 'cleaned_content']].head())

#%%

# Topic modeling with LDA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

#Make new stopword list due to overlap in previous runs
stopwords = ['american', 'america', 'states', 'state', 'president']

#Vectorize statements 
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words=stopwords, ngram_range=(1,3))
X = vectorizer.fit_transform(dataset['cleaned_content'])

#Extract the topics 
lda = LatentDirichletAllocation(n_components=7, random_state=42)
lda.fit(X)

#Function for displaying the topics
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx}:")
        print("|".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

#Display topics
no_top_words = 10
feature_names = vectorizer.get_feature_names_out()
display_topics(lda, feature_names, no_top_words)

#Topic coherence evaluation
print(lda.perplexity(X))
print(lda.score(X))
#%%

#Attempting topic modeling with Gensim LDA
import gensim
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary

#Prepare data for Gensim
texts = [doc.split() for doc in dataset['cleaned_content']]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

#Build LDA model
lda_model = gensim.models.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=15, random_state=42)

#Display topics
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic {idx}: {topic}")

#Evaluate topic coherence
coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print(f"Coherence Score: {coherence_lda}")

#%%

#Topic modeling again this time with NMF
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

#Vectorize again using TF-IDF
vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
X_tfidf = vectorizer.fit_transform(dataset['cleaned_content'])

#Fit NMF
nmf = NMF(n_components=5, random_state=42)
nmf.fit(X_tfidf)

#Display topics
no_top_words = 10 
feature_names = vectorizer.get_feature_names_out()
display_topics(nmf, feature_names, no_top_words)

#%%
# Optimized sentiment analysis with CardiffNLP politics model
import re
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Model setup
model_name = "cardiffnlp/xlm-twitter-politics-sentiment"
device = 0 if torch.cuda.is_available() else -1

# Use fast tokenizer; align truncation/padding for batching
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

sentiment_classifier = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    framework="pt",
    device=device
)

# Lightweight tweet-style normalization expected by CardiffNLP models
def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text
    t = re.sub(r"https?://\S+", "<url>", t)
    t = re.sub(r"www\.\S+", "<url>", t)
    t = re.sub(r"@[A-Za-z0-9_]+", "<user>", t)
    t = re.sub(r"#[A-Za-z0-9_]+", "<hashtag>", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# Prepare batched inputs
texts = dataset['content'].fillna("").map(normalize_text).tolist()

# Choose batch size based on hardware
batch_sz = 32 if device == -1 else 64  # CPU vs GPU

# Batched inference with padding/truncation for throughput
results = sentiment_classifier(
    texts,
    batch_size=batch_sz,
    truncation=True,
    padding=True,
    max_length=512  # shorter seq length speeds inference, adjust if needed
)

# Assign results back
dataset['transformer_sentiment'] = [r['label'] for r in results]
dataset['transformer_sentiment_score'] = [r['score'] for r in results]

# Quick checks
print(dataset[['transformer_sentiment']].value_counts().head())
print(dataset[['content', 'transformer_sentiment', 'transformer_sentiment_score']].head())
print(dataset.columns)


#%%
# Optimized sentiment analysis with CardiffNLP politics model
import re
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Model setup
model_name = "cardiffnlp/xlm-twitter-politics-sentiment"
device = 0 if torch.cuda.is_available() else -1

# Use fast tokenizer; align truncation/padding for batching
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

sentiment_classifier = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    framework="pt",
    device=device
)

# Lightweight tweet-style normalization expected by CardiffNLP models
def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text
    t = re.sub(r"https?://\S+", "<url>", t)
    t = re.sub(r"www\.\S+", "<url>", t)
    t = re.sub(r"@[A-Za-z0-9_]+", "<user>", t)
    t = re.sub(r"#[A-Za-z0-9_]+", "<hashtag>", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# Prepare batched inputs
texts = dataset['content'].fillna("").map(normalize_text).tolist()

# Choose batch size based on hardware
batch_sz = 32 if device == -1 else 64  # CPU vs GPU

# Batched inference with padding/truncation for throughput
results = sentiment_classifier(
    texts,
    batch_size=batch_sz,
    truncation=True,
    padding=True,
    max_length=256  # shorter seq length speeds inference, adjust if needed
)

# Assign results back
dataset['transformer_sentiment'] = [r['label'] for r in results]
dataset['transformer_sentiment_score'] = [r['score'] for r in results]

# Quick checks
print(dataset[['transformer_sentiment']].value_counts().head())
print(dataset[['content', 'transformer_sentiment', 'transformer_sentiment_score']].head())
print(dataset.columns)

#%%

# Visualization of sentiment distribution in dataset
import matplotlib.pyplot as plt
import seaborn as sns
# Sentiment distribution plot
plt.figure(figsize=(8,6))
sns.countplot(data=dataset, x='transformer_sentiment', order=['Negative', 'Neutral', 'Positive'])
plt.title('Sentiment Distribution of Presidential Statements')
plt.xlabel('Sentiment')
plt.ylabel('Number of Statements')
plt.show()
# Aggregate sentiment by president
president_sentiment = dataset.pivot_table(index='president', 
                                         columns='transformer_sentiment', 
                                         aggfunc='size', 
                                         fill_value=0)
# Print sentiment counts for each president ordered by Negative sentiment
president_sentiment = president_sentiment.sort_values(by='Negative', ascending=False)
print(president_sentiment)

#%%
# #Doing another of the same pipeline but changing max length to 256 to observe any differences
# # Optimized sentiment analysis with CardiffNLP politics model
# import re
# import torch
# from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# # Model setup
# model_name = "cardiffnlp/xlm-twitter-politics-sentiment"
# device = 0 if torch.cuda.is_available() else -1

# # Use fast tokenizer; align truncation/padding for batching
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
# model = AutoModelForSequenceClassification.from_pretrained(model_name)

# sentiment_classifier = pipeline(
#     "sentiment-analysis",
#     model=model,
#     tokenizer=tokenizer,
#     framework="pt",
#     device=device
# )

# # Lightweight tweet-style normalization expected by CardiffNLP models
# def normalize_text(text: str) -> str:
#     if not isinstance(text, str):
#         return ""
#     t = text
#     t = re.sub(r"https?://\S+", "<url>", t)
#     t = re.sub(r"www\.\S+", "<url>", t)
#     t = re.sub(r"@[A-Za-z0-9_]+", "<user>", t)
#     t = re.sub(r"#[A-Za-z0-9_]+", "<hashtag>", t)
#     t = re.sub(r"\s+", " ", t).strip()
#     return t

# # Prepare batched inputs
# texts = dataset['content'].fillna("").map(normalize_text).tolist()

# # Choose batch size based on hardware
# batch_sz = 32 if device == -1 else 64  # CPU vs GPU

# # Batched inference with padding/truncation for throughput
# results = sentiment_classifier(
#     texts,
#     batch_size=batch_sz,
#     truncation=True,
#     padding=True,
#     max_length=256  # shorter seq length speeds inference, adjust if needed
# )

# # Assign results back
# dataset['transformer_sentiment_256'] = [r['label'] for r in results]
# dataset['transformer_sentiment_score_256'] = [r['score'] for r in results]

# # Quick checks
# print(dataset[['transformer_sentiment_256']].value_counts().head())
# print(dataset[['content', 'transformer_sentiment_256', 'transformer_sentiment_score_256']].head())
# print(dataset.columns)

#%%

# Picking a sample for inspection and comparing both max length results


sample_df = dataset.sample(n=20, random_state=42)


print(sample_df[['content', 'transformer_sentiment', 'transformer_sentiment_score', 'transformer_sentiment_256', 'transformer_sentiment_score_256']])


#Check for differences between the two max lengths


differences = dataset[dataset['transformer_sentiment'] != dataset['transformer_sentiment_256']]


print(f"Number of differences between max_length 512 and 256: {differences.shape[0]}")

print(differences[['content', 'transformer_sentiment', 'transformer_sentiment_score', 'transformer_sentiment_256', 'transformer_sentiment_score_256']])


#%%
from datasets import Dataset
from transformers import pipeline
import torch

# Use MUCH faster MNLI model
zero_shot_clf = pipeline(
    task="zero-shot-classification",
    model="valhalla/distilbart-mnli-12-1",   
    device=0 if torch.cuda.is_available() else -1,
)

# Convert your speeches to a HuggingFace Dataset
ds = Dataset.from_dict({"text": dataset["content"].fillna("").tolist()})


# -------------------------------------------------------------
# Helper function: runs zero-shot for ANY label set
# -------------------------------------------------------------
def run_zero_shot(ds, labels, template, multi_label=False):
    """
    ds: HuggingFace Dataset
    labels: list of candidate labels
    template: hypothesis template
    multi_label: True/False
    """

    def _apply(batch):
        # 1. TRUNCATE TEXT (GPU safe + massively faster)
        batch_texts = [t[:512] for t in batch["text"]]

        # 2. Run zero-shot
        out = zero_shot_clf(
            batch_texts,
            candidate_labels=labels,
            hypothesis_template=template,
            multi_label=multi_label,
        )

        # 3. Return DICT of lists — REQUIRED by Dataset.map
        return {
            "zs_labels": [o["labels"] for o in out],
            "zs_scores": [o["scores"] for o in out],
        }

    return ds.map(_apply, batched=True, batch_size=32)


# -------------------------------------------------------------
# Define label groups
# -------------------------------------------------------------
tone_labels = ["combative", "conciliatory", "neutral-ceremonial"]

emotion_labels = ["anger", "fear", "hope", "pride", "sadness", "empathy"]


# -------------------------------------------------------------
# TONE CLASSIFICATION (single-label)
# -------------------------------------------------------------
print("\nRunning zero-shot tone classification...")

tone_out = run_zero_shot(
    ds,
    labels=tone_labels,
    template="The statement uses {} rhetoric.",
    multi_label=False,
)

# pick final label with margin rule
def pick_single_label(labels, scores, margin=0.05):
    if len(scores) > 1 and scores[0] - scores[1] < margin:
        return "mixed"
    return labels[0]

dataset["tone"] = [
    pick_single_label(lbls, scrs)
    for lbls, scrs in zip(tone_out["zs_labels"], tone_out["zs_scores"])
]


# -------------------------------------------------------------
# EMOTION CLASSIFICATION (multi-label)
# -------------------------------------------------------------
print("\nRunning zero-shot emotion classification...")

emotion_out = run_zero_shot(
    ds,
    labels=emotion_labels,
    template="The statement expresses {}.",
    multi_label=True,
)

thr_emotion = 0.30
dataset["emotions"] = [
    [lab for lab, score in zip(lbls, scrs) if score >= thr_emotion]
    for lbls, scrs in zip(emotion_out["zs_labels"], emotion_out["zs_scores"])
]


# -------------------------------------------------------------
# SUMMARY
# -------------------------------------------------------------
print("\n=== Zero-Shot Completed ===")
print("Tone distribution:\n", dataset["tone"].value_counts())
print("\nTop emotions:\n", dataset["emotions"].explode().value_counts().head(10))


#%%

# Visualization emotion distribution for each president after 2000
import matplotlib.pyplot as plt
import seaborn as sns

# Filter dataset for presidents after 2000
recent_presidents = ["George W. Bush", "Barack Obama", "Donald J. Trump (1st Term)", "Donald J. Trump (2nd Term)", "Joseph R. Biden, Jr."]
recent_data = dataset[dataset['president'].isin(recent_presidents)]

# Emotion ratios by president 
emotion_counts = recent_data.explode('emotions').groupby(['president', 'emotions']).size().unstack(fill_value=0)
emotion_ratios = emotion_counts.div(emotion_counts.sum(axis=1), axis=0)
emotion_ratios.plot(kind='bar', stacked=True, figsize=(10,6))
plt.title('Emotion Distribution by President (Post-2000)')
plt.xlabel('President')
plt.ylabel('Proportion of Emotions')
plt.legend(title='Emotions', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

#%%
#Visualizing ratio of tone by president after 2000
# Tone ratios by president
tone_counts = recent_data.groupby(['president', 'tone']).size().unstack(fill_value=0)
tone_ratios = tone_counts.div(tone_counts.sum(axis=1), axis=0)
tone_ratios.plot(kind='bar', stacked=True, figsize=(10,6))
plt.title('Tone Distribution by President (Post-2000)')
plt.xlabel('President')
plt.ylabel('Proportion of Tones')
plt.legend(title='Tones', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

#%%
#Line chart with multiple lines showing emotion trends over time
import matplotlib.pyplot as plt
import pandas as pd
# Convert 'date' to datetime
dataset['date'] = pd.to_datetime(dataset['date'], errors='coerce')  
# Filter dataset for anything after 1980
filtered_data = dataset[dataset['date'].dt.year >= 1980]
# Explode emotions for counting
exploded_data = filtered_data.explode('emotions')
# Group by year and emotion, then count occurrences
emotion_trends = exploded_data.groupby([exploded_data['date'].dt.year, 'emotions']).size().unstack(fill_value=0)
# Plotting
plt.figure(figsize=(12, 6))
for emotion in emotion_trends.columns:
    plt.plot(emotion_trends.index, emotion_trends[emotion], marker='o', label=emotion)
plt.xlabel('Year')
plt.ylabel('Count')
plt.title('Emotion Trends Over Time')
plt.legend()
plt.show()

#%%
# Implement BOW and Logistic Regression Baseline for sentiment analysis

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

#Dataset view
#print(dataset.columns)
#print(dataset.head()) 

#Use the transformer sentiment as labels for supervised learning
X = dataset['cleaned_content']
y = dataset['transformer_sentiment']

#Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english', ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

#Train Logistic Regression model
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_tfidf, y_train)

#Predict on test set
y_pred = model.predict(X_test_tfidf)

#Evaluate model
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Derive feature importance from Logistic Regression model
import numpy as np
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_
for i, class_label in enumerate(model.classes_):
    top_positive_indices = np.argsort(coefficients[i])[-10:]
    top_negative_indices = np.argsort(coefficients[i])[:10]
    print(f"Top positive features for class {class_label}:")
    print([feature_names[j] for j in top_positive_indices])
    print(f"Top negative features for class {class_label}:")
    print([feature_names[j] for j in top_negative_indices])

#%%
#Make visualization of confusion matrix of logistic regression model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

#Predict on test set
y_pred = model.predict(X_test_tfidf)

#Evaluate model
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#%%

#Make more visualizations using labels derived from CardiffNLP transformer model
import matplotlib.pyplot as plt
import seaborn as sns 

# Value counts for positive or negative sentiment
sentiment_counts = dataset['transformer_sentiment'].value_counts() 

# Bar plot for sentiment distribution
plt.figure(figsize=(8,6))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Distribution of Sentiment Labels from CardiffNLP Model')
plt.show()

#Ordered bar plot for positive and negative sentiment for each president
president_sentiment = dataset.groupby(['president', 'transformer_sentiment']).size().unstack(fill_value=0)
president_sentiment = president_sentiment.reindex(president_sentiment.sum(axis=1).sort_values(ascending=False).index)
president_sentiment.plot(kind='bar', stacked=True, figsize=(12,8), colormap='viridis')
plt.xlabel('President')
plt.ylabel('Number of Statements')
plt.title('Sentiment Distribution by President')
plt.legend(title='Sentiment')
plt.show()

#%%
# This time doing a ratio of sentiment labels per president for the last 5 presidents
recent_presidents = ["Donald J. Trump (1st Term)", "Donald J. Trump (2nd Term)", "Joseph R. Biden, Jr.", "Barack Obama", "George W. Bush", "William J. Clinton", "George Bush"]
recent_data = dataset[dataset['president'].isin(recent_presidents)]

# Calculate sentiment ratios
sentiment_ratios = recent_data.groupby(['president', 'transformer_sentiment']).size().unstack(fill_value=0)
sentiment_ratios = sentiment_ratios.div(sentiment_ratios.sum(axis=1), axis=0) 

# Plot sentiment ratios
sentiment_ratios.plot(kind='bar', stacked=True, figsize=(10,6), colormap='viridis')
plt.xlabel('President')
plt.ylabel('Proportion of Statements')
plt.title('Sentiment Proportions by Recent Presidents')
plt.legend(title='Sentiment')
plt.show()

#%%
#Plot to show the presidens with the highest negative sentiment over total number of statements ratio
negative_ratios = sentiment_ratios['Negative'].sort_values(ascending=False)

# Plot negative sentiment ratios
plt.figure(figsize=(10,6))
sns.barplot(x=negative_ratios.index, y=negative_ratios.values, palette='viridis')
plt.xlabel('President')
plt.ylabel('Proportion of Negative Statements')
plt.title('Proportion of Negative Sentiment by President')
plt.xticks(rotation=45)
plt.show()

#%%
#Porportion of positive ratio by president ordered by highest to lowest
positive_ratios = sentiment_ratios['Positive'].sort_values(ascending=False)

# Plot positive sentiment ratios
plt.figure(figsize=(10,6))
sns.barplot(x=positive_ratios.index, y=positive_ratios.values, palette='viridis')
plt.xlabel('President')
plt.ylabel('Proportion of Positive Statements')
plt.title('Proportion of Positive Sentiment by President')
plt.xticks(rotation=45)
plt.show()


#%%
#Identify the most negative statements overall

print(dataset['transformer_sentiment'].value_counts())
negative_statements = dataset[dataset['transformer_sentiment'] == 'Negative']
negative_statements = negative_statements.sort_values(by='transformer_sentiment_score', ascending=False)
print(negative_statements[['president', 'date', 'title', 'transformer_sentiment_score', 'content']].head(10))

#%%
#sentiment over time visualization
import matplotlib.pyplot as plt
import pandas as pd

# Convert date column to datetime
dataset['date'] = pd.to_datetime(dataset['date'], errors='coerce')

# Extract year from date
dataset['year'] = dataset['date'].dt.year

# Group by year and sentiment, count occurrences
yearly_sentiment = dataset.groupby(['year', 'transformer_sentiment']).size().unstack(fill_value=0)

#Filter year to reasonable range
yearly_sentiment = yearly_sentiment[(yearly_sentiment.index >= 1980)]

# Plot sentiment trends over time
yearly_sentiment.plot(kind='line', figsize=(12,6), marker='o')
plt.xlabel('Year')
plt.ylabel('Number of Statements')
plt.title('Sentiment Trends Over Time')
plt.legend(title='Sentiment')
plt.show()