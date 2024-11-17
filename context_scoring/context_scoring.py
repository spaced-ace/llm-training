import re
import json
import tqdm
import datasets
import spacy
import argparse
import pandas as pd
from textstat import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from gensim.models import LdaModel

HUNGARIAN_VOWELS = 'aáeéiíoóöőuúüű'
VOWELFINDER=re.compile(f'[{HUNGARIAN_VOWELS}]', re.IGNORECASE)
NAMED_ENTITY_DENSITY_MAX = 0.30

def get_readability_score(context):
    return textstat.flesch_reading_ease(context)/100

def get_hungarian_readability_score(context, nlp_pipeline):
    doc = nlp_pipeline(context)
    word_count = len([token for token in doc if token.is_alpha])
    sentence_count = len(list(doc.sents))
    vowel_count = len(VOWELFINDER.findall(context.lower()))
    
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    avg_syllables_per_word = vowel_count / word_count if word_count > 0 else 0
    # Adapt Flesch Reading Ease for Hungarian
    # Note: These coefficients might need further adjustment based on Hungarian language studies
    readability = 1.5*206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
    # Normalize to 0-1 range
    return max(0, min(readability, 100)) / 100

def calculate_named_entity_density(context, nlp_pipeline):
    context_len = len(context.split())
    if context_len == 0:
        return 0
    doc = nlp_pipeline(context)
    named_entities = [ent.text for ent in doc.ents]
    return len(named_entities) / context_len


def detect_truncation(context):
    truncation_patterns = [
        r'\:.$'
        r'for example:$',
        r'such as:$',
        r'including:$',
        r'\.\.\.$',
        r'etc\.$'
    ]
    return any(re.search(pattern, context, re.IGNORECASE) for pattern in truncation_patterns)

def detect_truncation_hu(context):
    truncation_patterns = [
        r'\:.$'
        r'például:$',
        r'mint:$',
        r'beleértve:$',
        r'\.\.\.$',
        r'stb\.$'
    ]
    return any(re.search(pattern, context, re.IGNORECASE) for pattern in truncation_patterns)

def check_length(context, min_length=10, max_length=200):
    wordcount = len(list(filter(lambda x: not x.isspace(), context.split(' '))))
    return min_length <= wordcount <= max_length


def analyze_sentence_structure(context, nlp_pipeline):
    doc = nlp_pipeline(context)
    sentences = list(doc.sents)
    if len(sentences) == 0:
        return 0  # No valid sentences
    
    complete_sentences = sum(1 for sent in sentences if len(sent) > 3 and any(token.pos_ == "VERB" for token in sent))
    return complete_sentences / len(sentences)

def is_title_only(context, nlp_pipeline):
    doc = nlp_pipeline(context)
    return len(list(doc.sents)) == 1 and all(token.is_title for token in doc if token.is_alpha)

def check_coherence(context, nlp_pipeline, num_topics=3):
    if len(context) == 0:
        return 0
    sentences = [sent.text.split() for sent in nlp_pipeline(context).sents]
    dictionary = Dictionary(sentences)
    corpus = [dictionary.doc2bow(text) for text in sentences]
    if len(corpus) == 0:
        return 0
    
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=10)
    coherence_model = CoherenceModel(model=lda_model, texts=sentences, dictionary=dictionary, coherence='c_v')
    
    return coherence_model.get_coherence()

def measure_content_diversity(context) -> float:
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([context])
        unique_terms = (tfidf_matrix.toarray() > 0).sum()
        return unique_terms / len(context.split())
    # happens if context only contains stopwords
    except ValueError as e:
        print(e)
        return 0

def assess_context_quality(context, nlp_pipeline, lang: str):
    scores = {
        'length': 1 if check_length(context) else 0,
        'not_truncated': 0 if detect_truncation(context) else 1,
        'not_title_only': 0 if is_title_only(context, nlp_pipeline) else 1,
        'named_entity_density': calculate_named_entity_density(context, nlp_pipeline),
        'sentence_structure': analyze_sentence_structure(context, nlp_pipeline),
        'content_diversity': measure_content_diversity(context),
        'coherence': check_coherence(context, nlp_pipeline),
        'readability': (
            get_readability_score(context) if lang == 'en'
            else get_hungarian_readability_score(context, nlp_pipeline)
        )
    }

    if scores['length'] == 0:
        return 0, scores
    if scores['not_truncated'] == 0:
        return 0, scores
    if scores['not_title_only'] == 0:
        return 0, scores
    if scores['named_entity_density'] > NAMED_ENTITY_DENSITY_MAX:
        return 0, scores
    
    return sum([
        scores['length']*0.15,
        scores['sentence_structure']*0.15,
        scores['content_diversity']*0.2,
        scores['coherence']*0.2,
        scores['readability']*0.2,
        scores['named_entity_density']*0.1,
    ]), scores


def process_contexts(contexts: list[tuple[str, str]], nlp_pipeline):
    scores = [
        assess_context_quality(context, nlp_pipeline, lang)
        for lang, context in tqdm.tqdm(
            contexts, desc=f'Processing contexts', total=len(contexts)
        )]
    return scores

def read_input_data(dir_path:str) -> datasets.DatasetDict | datasets.Dataset:
    return datasets.load_from_disk(dir_path)

def read_labeled_data(json_path: str ) -> pd.DataFrame:
    json_questions = []
    with open(json_path, 'r') as f:
        json_questions = json.load(f)
    return pd.DataFrame.from_records(json_questions)

def atomize_scoring(scores: list[tuple])->pd.DataFrame:
    df = pd.DataFrame.from_records([s[1] for s in scores])
    df['score'] = [s[0] for s in scores]
    return df

def process_dataset(ds: datasets.Dataset, nlp_hu, nlp_en) -> datasets.Dataset:
    df = ds.to_pandas()
    df_en = df.loc[~df['url'].str.contains('hu.wikipedia')]
    df_hu = df.loc[df['url'].str.contains('hu.wikipedia')]
    en_scores = process_contexts([('en', context) for context in df_en['context']], nlp_en)
    hu_scores = process_contexts([('hu', context) for context in df_hu['context']], nlp_hu)
    en_scores = atomize_scoring(en_scores)
    hu_scores = atomize_scoring(hu_scores)
    hu_scores.index = df_hu.index
    en_scores.index = df_en.index
    df_en = pd.concat([df_en, en_scores], axis=1)
    df_hu = pd.concat([df_hu, hu_scores], axis=1)
    df = pd.concat([df_en, df_hu]).sort_index()
    return datasets.Dataset.from_pandas(df, preserve_index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str, help='The directory containing the generated questions in hf dataset format')
    parser.add_argument('output_dir', type=str, help='The directory where the scored contexts will be output to')
    args = parser.parse_args()
    nlp_en = spacy.load("en_core_web_sm")
    nlp_hu = spacy.load("hu_core_news_md")
    ds = read_input_data(args.dataset_path)
    scored = {}
    for subset in list(ds.keys()):
        print(f'Processing {subset}')
        sset = process_dataset(ds[subset], nlp_hu, nlp_en)
        scored[subset] = sset
    scored_ds = datasets.DatasetDict(scored)
    scored_ds.save_to_disk(args.output_dir)
