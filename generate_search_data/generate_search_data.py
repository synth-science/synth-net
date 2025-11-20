# %%
import io
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer, util
from cryptography.fernet import Fernet

# %%
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:    
    SCRIPT_DIR = Path.cwd()

# %%
config_path = f"{SCRIPT_DIR}/config.yaml"

with open(config_path) as stream:
    config = yaml.safe_load(stream)

# %%
meta_data_raw = pd.read_json(path_or_buf=config['meta_data_load_path'])

# %%
meta_data_columns = {
    "DOI": "meta_doi",
    "Name": "meta_instrument_name",
    "meta_language": "meta_language",
    "number_of_test_items_best_guess": "meta_item_count",
    "citation_count": "meta_citation_count",
    "number_of_factors_subscales": "meta_scale_count",
}

# Extract nested language data
meta_data_raw['meta_language'] = meta_data_raw['LanguagePresentList'].apply(
    lambda x: x.get('LanguagePresent', [[None]])[0][0] 
    if isinstance(x, dict) and 'LanguagePresent' in x 
    and len(x.get('LanguagePresent', [])) > 0 
    and len(x['LanguagePresent'][0]) > 0 
    else None
)

meta_data = (
    meta_data_raw
    .rename(columns=meta_data_columns)
    [list(meta_data_columns.values())]
)

# %%
pdf_data_raw = (
    pd.read_parquet(path=config["pdf_data_load_path"])
    .query("filename != '999900001_full_001 (2).pdf'")
)

# Extract instrument type
pdf_data_raw['meta_instrument_type'] = pdf_data_raw['cover_text'].str.findall(
    r'Instrument Type:\s*\n([^\n]+)'
)

# Extract test format
pdf_data_raw['meta_test_format'] = (
    pdf_data_raw['cover_text']
    .str.extract(r'(?s)Test Format:\s*\n(.*)', expand=False)
    .str.split('\nSource:')
    .str[0]
    .str.strip()
)

# Extract DOI
pdf_data_raw['meta_doi'] = (
    pdf_data_raw['cover_text']
    .str.extract(r'(?i)doi:\s*(https?://[^\s,\n]+)', expand=False)
    .str.replace(r'https?://(?:dx\.)?doi\.org/', '', regex=True)
)

# Extract source
pdf_data_raw['meta_source'] = (
    pdf_data_raw['cover_text']
    .str.extract(r'(?s)Source:\s*\n(.*)', expand=False)
    .str.split('\nPermissions:')
    .str[0]
    .str.strip()
)

# %%
def flatten(df: pd.DataFrame) -> pd.DataFrame:
    
    exploded_df = df.explode('survey_content')
    exploded_df = exploded_df[exploded_df['survey_content'].notna()]   
    normalized_df = pd.json_normalize(exploded_df['survey_content'])
    
    other_columns = exploded_df.drop(
        'survey_content', axis=1).reset_index(drop=True)
    result_df = pd.concat([other_columns, normalized_df], axis=1)

    return result_df

def add_fuzzy_similarity(df):
    scores = [
        fuzz.partial_ratio(row['item_text'], str(row['text']).lower())
        if pd.notna(row['item_text']) and pd.notna(row['text']) else None
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing fuzzy string similarity")
    ]
    return df.assign(fuzzy_similarity=scores)

data = (
    pdf_data_raw
    .merge(meta_data, on='meta_doi', how='inner')
    .query('meta_language == "English" or meta_language.isna()', engine='python')
    .pipe(flatten)
    .query('not construct_name.str.contains("(?i)demographics", na=False)', engine='python')
    .assign(
        meta_instrument_name=lambda x: x['meta_instrument_name'].str.title(),
        construct_name=lambda x: x['construct_name'].replace('<NONE>', pd.NA).str.title(),
        meta_scale_count=lambda x: x['meta_scale_count'].fillna(0).astype('Int32'),
        item_text=lambda x: (
            x['item_text'].astype(str)
            .str.lower()
            .str.replace(r'^[0-9]+', '', regex=True)
            .str.replace(r'[^\w\s]+$', '', regex=True)
            .str.strip()
        )
    )
    .rename(columns={'construct_name': 'scale_name'})
    .dropna(subset=['item_text'])
    .query('item_text != ""')
    .pipe(add_fuzzy_similarity)
)

# %%
item_model = SentenceTransformer(model_name_or_path=config['item_model_path'], device='cuda')
scale_model = SentenceTransformer(model_name_or_path=config['construct_model_path'], device='cuda')

# %%
embedding_data = data.copy(deep=True)

item_embeddings = item_model.encode(
    sentences=embedding_data['item_text'].to_list(), 
    convert_to_tensor=True, 
    show_progress_bar=True, 
    batch_size=512
).cpu().numpy()

embedding_data['item_embeddings'] = [x for x in item_embeddings]

embedding_data['scale_label'] = np.where(
    embedding_data['scale_name'].isna(), 
    embedding_data['meta_instrument_name'], 
    embedding_data['scale_name']
)

scale_embeddings = scale_model.encode(
    sentences=embedding_data['scale_label'].to_list(), 
    convert_to_tensor=True, 
    show_progress_bar=True, 
    batch_size=512
).cpu().numpy()

embedding_data['scale_embeddings'] = [x for x in scale_embeddings]

# %%
# compute centroids
def compute_aligned_centroid(item_embeddings, keying):

    item_embeddings_positive = item_embeddings[[x == "positive" for x in keying]]
    item_embeddings_negative = item_embeddings[[x == "negative" for x in keying]]

    if item_embeddings_positive.size == 0 or item_embeddings_negative.size == 0:
        return item_embeddings.mean(axis=0)
    
    item_centroid_positive = item_embeddings_positive.mean(axis=0)
    item_centroid_negative = item_embeddings_negative.mean(axis=0)

    cosine_similarities = util.cos_sim(item_embeddings, item_centroid_positive).numpy().squeeze()
    synthetic_is_negative = cosine_similarities < 0

    polarity_axis = item_centroid_positive - item_centroid_negative
    axis_magnitude = np.sqrt(np.sum(polarity_axis**2))

    if not np.isfinite(axis_magnitude) or axis_magnitude <= 0 or not any(synthetic_is_negative):
        return item_embeddings.mean(axis=0)

    polarity_unit_vector = polarity_axis / axis_magnitude
    reflection_plane_center = (item_centroid_positive + item_centroid_negative) / 2

    signed_distances_to_plane = np.dot(
        item_embeddings - reflection_plane_center, 
        polarity_unit_vector
    )

    items_to_align = np.array([x == "negative" for x in keying]) & synthetic_is_negative

    reflection_distances = np.where(items_to_align, signed_distances_to_plane, 0)

    item_embeddings_aligned = item_embeddings - 2 * np.outer(
        reflection_distances, 
        polarity_unit_vector
    )

    item_centroid_aligned = item_embeddings_aligned.mean(axis=0)

    return item_centroid_aligned

def compute_scale_centroid(embedding_list):
    valid = [e for e in embedding_list if e is not None]
    return np.mean(valid, axis=0) if valid else None

instrument_level = (
    embedding_data
    .groupby('meta_doi')
    [['item_embeddings', 'keying', 'scale_embeddings']]
    .agg(list)
    .reset_index()
    .assign(
        item_centroid=lambda df: df.apply(
            lambda row: compute_aligned_centroid(
                np.array(row['item_embeddings']), 
                row['keying']
            ), 
            axis=1
        ),
        scale_centroid=lambda df: df['scale_embeddings'].apply(
            lambda x: np.mean(x, axis=0)
        ),
        is_instrument=True,
    )
)

scale_level = (
    embedding_data
    .groupby(['meta_doi', 'scale_name'])
    [['item_embeddings', 'keying', 'scale_embeddings']]
    .agg(list)
    .reset_index()
    .assign(
        item_centroid=lambda df: df.apply(
            lambda row: compute_aligned_centroid(
                np.array(row['item_embeddings']), 
                row['keying']
            ), 
            axis=1
        ),
        scale_centroid=lambda df: df['scale_embeddings'].apply(
            lambda x: np.mean(x, axis=0)
        ),
        is_instrument=False,
    )
)

centroid_data = pd.concat([instrument_level, scale_level], axis=0, ignore_index=True)

# %%
def flag_item_count_deviation(df):
    
    valid = df['meta_item_count'].notna()
    actual_counts = df['item_text'].apply(len)
    
    return valid & (actual_counts != df['meta_item_count'])

def flag_scale_count_deviation(df):
    
    valid = ~df['meta_scale_count'].isna() & (df['meta_scale_count'] != 0)
    unique_scales = df['scale_name'].apply(lambda x: len(set(x)))

    return valid & (unique_scales != df['meta_scale_count'])

def flag_item_text_deviation(df, threshold=95):

    below_threshold = df['fuzzy_similarity'].apply(
        lambda x: any(val < threshold for val in x if pd.notna(val))
    )
    
    return below_threshold

warning_data = (
    data
    .groupby('meta_doi')
    .agg({
        'meta_item_count': 'first',
        'meta_scale_count': 'first',        
        'item_text': list,
        'scale_name': list,
        'fuzzy_similarity': list,
        'keying_correction': list,
    })
    .reset_index()
    .assign(
        warn_item_count_deviation=lambda df: df.pipe(flag_item_count_deviation),
        warn_scale_count_deviation=lambda df: df.pipe(flag_scale_count_deviation),
        warn_item_text_deviation=lambda df: df.pipe(flag_item_text_deviation),
        warn_keying_correction=lambda df: df['keying_correction'].apply(lambda x: any(x))
    )
)

# %%
# create centroid search data
centroid_columns = ["meta_doi", "scale_name", "is_instrument", "item_centroid", "scale_centroid"]
warning_columns = ["meta_doi", "warn_item_count_deviation", "warn_scale_count_deviation", "warn_item_text_deviation", "warn_keying_correction"]

centroid_search_data = (
    centroid_data[centroid_columns]
    .merge(
        right=data[['meta_doi', 'meta_instrument_name']].drop_duplicates(subset='meta_doi'),
        on='meta_doi',
        how='left'
    )
    .merge(
        right=warning_data[warning_columns],
        on='meta_doi',
        how='left'
    )
)
# %%
# create item search data
item_search_data = (
    embedding_data[
        ['meta_doi', 'meta_instrument_name', 'scale_name',
         'item_text', 'keying', 'fuzzy_similarity',
         'item_embeddings', 'scale_embeddings']
    ].copy()
    .merge(
        right=centroid_data[['meta_doi', 'item_centroid', 'scale_centroid']].drop_duplicates(subset='meta_doi'),
        on='meta_doi',
        how='left'
    )
)

# %%
# save centroid search data
centroid_search_data.to_parquet(path=config['save_centroid_search_data_path'], index=False)

# %%
# save item search data
item_search_data.to_parquet(path=config['save_item_search_data_path'], index=False)

# %%
# Summarize heuristic quality checks
def report_item_count_quality(grouped_data):
    item_count = grouped_data.agg({'meta_item_count': 'first', 'item_text': 'count'})
    deltas = (
        item_count
        .dropna()
        .astype(int)
        .query("meta_item_count != 0")
        .assign(delta=lambda df: df['item_text'] - df['meta_item_count'])
    )
    
    _print_quality_metrics(deltas, "item")


def report_scale_count_quality(grouped_data):
    scale_count = grouped_data.agg({
        'meta_scale_count': 'first',
        'scale_name': lambda x: len(set(x))
    })
    deltas = (
        scale_count
        .dropna()
        .astype(int)
        .query("meta_scale_count != 0")
        .assign(delta=lambda df: df['scale_name'] - df['meta_scale_count'])
    )
    
    _print_quality_metrics(deltas, "scale")


def _print_quality_metrics(deltas, metric_name):
    deviations = deltas.query("delta != 0")
    overextracted = deltas.query("delta > 0")
    underextracted = deltas.query("delta < 0")
    
    print(f"Deviations in {metric_name} counts: {deviations.shape[0]}")
    print(f"Median abs deviation: {deviations['delta'].abs().median()}")
    print(f"IQR abs deviation: {_calculate_iqr(deviations['delta'].abs())}")
    
    print(f"Overextraction, {metric_name} count: {overextracted.shape[0]}")
    print(f"Overextraction, Median abs deviation: {overextracted['delta'].abs().median()}")
    print(f"Overextraction, IQR abs deviation: {_calculate_iqr(overextracted['delta'].abs())}")
    
    print(f"Underextraction, {metric_name} count: {underextracted.shape[0]}")
    print(f"Underextraction, Median abs deviation: {underextracted['delta'].abs().median()}")
    print(f"Underextraction, IQR abs deviation: {_calculate_iqr(underextracted['delta'].abs())}")


def _calculate_iqr(series):
    return series.quantile(0.75) - series.quantile(0.25)

grouped_data = data.groupby("meta_doi")

print(f"Instruments in total: {centroid_search_data.groupby("meta_doi").size().shape[0]}")
print(f"Scales in total: {centroid_search_data.groupby(['meta_doi', 'scale_name'], dropna=True).size().shape[0]}")

report_item_count_quality(grouped_data)
report_scale_count_quality(grouped_data)

# %%
print("Done!")