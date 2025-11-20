# -*- coding: utf-8 -*-
import os
import json
import yaml
import random
import time
import pandas as pd
from tqdm import tqdm

from common import setup_logger, timeout_method, timeit, TimeoutError # type: ignore
from document import APAPsycTestDocument # type: ignore
from vl_extractor import VisionLanguageModelExtractor # type: ignore
from api_extractor import LanguageModelExtractorAPI # type: ignore

logger = setup_logger('main', log_level='DEBUG')

def apply_config_overrides(config, overrides):

    for override in overrides:
        if '=' not in override:
            raise ValueError(f"Invalid override format: {override}. Expected key=value")
        
        key_path, value = override.split('=', 1)
        keys = key_path.split('.')
        
        try:
            if '.' not in value:
                value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                if value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
        
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
        logger.info(f"Config override applied: {key_path} = {value}")

def extract(config):
    """Main extraction routine that processes documents."""    
    input_files = os.listdir(path=config['input_dir'])
    random.seed(config['seed'])
    random.shuffle(input_files)

    vl_extractor = VisionLanguageModelExtractor(
        model_path=config['vl_extractor'].get('model_path'),
        timeout_seconds=config['vl_extractor'].get('timeout_seconds'),
    )

    api_extractor = LanguageModelExtractorAPI(
        model_name = config['api_extractor'].get('model_name'),
        options = config['api_extractor'].get('options'),
        max_reprompt_attempts = config['api_extractor'].get('max_reprompt_attempts'),
        timeout_seconds = config['api_extractor'].get('timeout_seconds'),
    )

    if config['test_mode']:
        pass

    elif os.path.exists(config['save_parquet_path']) and not config['overwrite_save_files']:
        df = pd.read_parquet(config['save_parquet_path'])
        records = df.to_dict(orient="records")
        filenames = [x for x in input_files if x not in df.filename.tolist()]

        logger.info(
            f"{'#' * 10} Resuming extraction with {len(records)} completed and {len(filenames)} remaining records! {'#' * 10}"
        )

    elif config['do_sample']:
        all_files = random.sample(population=input_files, k=config['n_samples'])

    else:
        filenames = input_files
        records = []

    logger.info(
        f"{'#' * 10} Beginning extraction of {len(filenames)} records! {'#' * 10}"
    )

    if config['test_files_first']:
        filenames.sort(key=lambda x: (x not in config['test_files'], x))

    for filename in tqdm(filenames):

        if not filename.endswith('.pdf'):
            continue

        start_time = time.time()
        logger.info(f"Processing {filename}...")

        pdf_path = os.path.join(config['input_dir'], filename)

        document = APAPsycTestDocument(
            pdf_path=pdf_path
        )

        parsed_transcript = vl_extractor.transcribe(document)
        transcript = parsed_transcript['transcript_content']

        parsed_survey = api_extractor.process_survey(transcript)
        end_time = time.time()
    
        record = {
            'filename': filename,
            'cover_text': document.cover_text,
            'text': document.text,
            'duration': end_time - start_time,
            **parsed_transcript,
            **parsed_survey
        }

        records.append(record)
    
        try:
            pd.DataFrame(records).to_parquet(config['save_parquet_path'])
        except Exception as e:
            logger.error(f"Failed saving records as .parquet: {e}")

    logger.info(f"Finished processing {filename} in {end_time - start_time} seconds!")

    logger.info(f"{'#' * 10} Finished extraction of {len(records)} records! {'#' * 10}")

    try:
        pd.DataFrame(records).to_parquet(config['save_parquet_path'])
    except Exception as e:
        logger.error(f"Failed saving records as .parquet!")

def fix_missing_transcripts(config):

    logger.info("Starting fix_missing_transcripts mode")

    if not os.path.exists(config['save_parquet_path']):
        logger.error(f"Output file {config['save_parquet_path']} not found!")
        return

    logger.info(f"Loading records from {config['save_parquet_path']}")
    df = pd.read_parquet(config['save_parquet_path'])
    
    error_mask = df['transcript_has_error'] == True
    error_records = df[error_mask]
    
    if len(error_records) == 0:
        logger.info("No records with transcript errors found!")
        return
    
    logger.info(f"Found {len(error_records)} records with transcript errors to fix")
    
    vl_extractor = VisionLanguageModelExtractor(
        model_path=config['vl_extractor'].get('model_path'),
        timeout_seconds=config['vl_extractor'].get('timeout_seconds'),
    )
    
    fixed_count = 0
    for idx, record in tqdm(error_records.iterrows(), total=len(error_records), desc="Fixing transcripts"):
        filename = record['filename']
        logger.info(f"Attempting to fix transcript for {filename}...")
        
        try:
            pdf_path = os.path.join(config['input_dir'], filename)
            
            if not os.path.exists(pdf_path):
                logger.warning(f"PDF file not found: {pdf_path}")
                continue
            
            document = APAPsycTestDocument(
                pdf_path=pdf_path,
                max_megapixels=config.get('max_megapixels', 1.8)
            )
            
            parsed_transcript = vl_extractor.transcribe(document)
            
            if not parsed_transcript['transcript_has_error']:
                df.loc[idx, 'transcript_content'] = parsed_transcript['transcript_content']
                df.loc[idx, 'transcript_duration_seconds'] = parsed_transcript['transcript_duration_seconds']
                df.loc[idx, 'transcript_length'] = parsed_transcript['transcript_length']
                df.loc[idx, 'transcript_has_error'] = False
                df.loc[idx, 'transcript_error'] = None
                
                fixed_count += 1
                logger.info(f"Successfully fixed transcript for {filename}")
                
                logger.info(f"Saving progress... Fixed {fixed_count} transcripts so far")
                df.to_parquet(config['save_parquet_path'])
            else:
                logger.warning(f"Transcript still has error for {filename}: {parsed_transcript['transcript_error']}")
                df.loc[idx, 'transcript_error'] = parsed_transcript['transcript_error']
                
        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
            df.loc[idx, 'transcript_error'] = f"Fix attempt failed: {str(e)}"
    
    logger.info(f"Saving final results... Fixed {fixed_count} out of {len(error_records)} transcript errors")
    df.to_parquet(config['save_parquet_path'])
    
    remaining_errors = df['transcript_has_error'].sum()
    logger.info(f"{'#' * 10} Fix transcript summary {'#' * 10}")
    logger.info(f"Total records: {len(df)}")
    logger.info(f"Fixed transcripts: {fixed_count}")
    logger.info(f"Remaining transcript errors: {remaining_errors}")
    logger.info(f"Success rate: {fixed_count / len(error_records) * 100:.1f}%")

    
def fix_missing_surveys(config, force=False):

    logger.info("Starting fix_missing_surveys mode")

    if not os.path.exists(config['save_parquet_path']):
        logger.error(f"Output file {config['save_parquet_path']} not found!")
        return

    logger.info(f"Loading records from {config['save_parquet_path']}")
    df = pd.read_parquet(config['save_parquet_path'])
    
    if force:
        error_mask = df['survey_has_error'] == True
        logger.info("Force mode: fixing all records with survey_has_error=True")
    else:
        error_mask = (df['survey_has_error'] == True) & (df['survey_content'].isna())
        logger.info("Standard mode: fixing records with survey_has_error=True AND survey_content=None")
    
    error_records = df[error_mask]
    
    if len(error_records) == 0:
        if force:
            logger.info("No records with survey errors found!")
        else:
            logger.info("No records with survey errors and missing survey content found!")
        return
    
    if force:
        logger.info(f"Found {len(error_records)} records with survey errors to fix (force mode)")
    else:
        logger.info(f"Found {len(error_records)} records with survey errors and missing content to fix")
    
    api_extractor = LanguageModelExtractorAPI(
        model_name=config['api_extractor'].get('model_name'),
        options=config['api_extractor'].get('options'),
        max_reprompt_attempts=config['api_extractor'].get('max_reprompt_attempts'),
        timeout_seconds=config['api_extractor'].get('timeout_seconds'),
    )
    
    fixed_count = 0
    processed_count = 0
    for idx, record in tqdm(error_records.iterrows(), total=len(error_records), desc="Fixing surveys"):
        filename = record['filename']
        logger.info(f"Attempting to fix survey for {filename}...")
        
        try:
            if record['transcript_has_error'] or pd.isna(record['transcript_content']) or not record['transcript_content']:
                logger.warning(f"Cannot fix survey for {filename}: no valid transcript available")
                processed_count += 1
                continue
            
            transcript = record['transcript_content']
            parsed_survey = api_extractor.process_survey(transcript)
            
            for key, value in parsed_survey.items():
                df.at[idx, key] = value
            
            if not parsed_survey['survey_has_error']:
                fixed_count += 1
                logger.info(f"Successfully fixed survey for {filename}")
            else:
                logger.warning(f"Survey still has error for {filename}: {parsed_survey['survey_error']}")
            
            processed_count += 1
            
            logger.info(f"Saving progress... Processed {processed_count} records, {fixed_count} successful fixes so far")
            df.to_parquet(config['save_parquet_path'])
                
        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
    
    logger.info(f"Saving final results... Processed {processed_count} records, fixed {fixed_count} surveys")
    df.to_parquet(config['save_parquet_path'])
    
    remaining_errors = df['survey_has_error'].sum()
    logger.info(f"{'#' * 10} Fix surveys summary {'#' * 10}")
    logger.info(f"Total records: {len(df)}")
    logger.info(f"Records processed: {processed_count}")
    logger.info(f"Fixed surveys: {fixed_count}")
    logger.info(f"Remaining survey errors: {remaining_errors}")
    if processed_count > 0:
        logger.info(f"Success rate: {fixed_count / processed_count * 100:.1f}%")
    else:
        logger.info("Success rate: N/A (no records processed)")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Process documents with different modes.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        python main.py --mode extract
        python main.py --mode fix_missing_transcripts
        python main.py --mode fix_missing_surveys
        python main.py --mode fix_missing_surveys --force
        python main.py --mode extract --set api_extractor.model_name=qwen3:7b --set seed=42
        python main.py --mode extract --set api_extractor.options.temperature=0.8 --set n_samples=5
        """
    )

    parser.add_argument(
        '--mode',
        required=True,
        choices=['extract', 'fix_missing_transcripts', 'fix_missing_surveys'],
        help='Operation mode (required)'
    )
    
    parser.add_argument(
        '--set',
        action='append',
        dest='config_overrides',
        help='Override config values (format: key.subkey=value). Can be used multiple times.'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force mode: for fix_missing_surveys, fix all records with survey_has_error=True regardless of survey_content'
    )
    
    args = parser.parse_args()
    
    config_path = "./config.yaml"
    with open(config_path) as stream:
        config = yaml.safe_load(stream)
    
    if args.config_overrides:
        logger.info(f"Applying {len(args.config_overrides)} config override(s)")
        apply_config_overrides(config, args.config_overrides)
    
    if args.mode == 'extract':
        extract(config)
    elif args.mode == 'fix_missing_transcripts':
        fix_missing_transcripts(config)
    elif args.mode == 'fix_missing_surveys':
        fix_missing_surveys(config, force=args.force)

if __name__ == "__main__":
    main()