import streamlit as st
from pathlib import Path
from typing import Dict, List
import pandas as pd

@st.cache_data
def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load CSV results file with caching.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame with results
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is malformed
    """
    try:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Validate required columns
        required_columns = {'dialect', 'wer', 'cer', 'bleu'}
        if not required_columns.issubset(df.columns):
            raise ValueError(
                f"CSV file must contain columns: {required_columns}. "
                f"Found: {set(df.columns)}"
            )
        
        return df
        
    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV file is empty: {csv_path}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Failed to parse CSV file: {e}")
    except Exception as e:
        raise IOError(f"Failed to load CSV file: {e}") from e


def get_available_results(results_base_path: str = "results/metrics") -> Dict[str, List[Dict[str, str]]]:
    """
    Get list of all available result files in the results directory, grouped by model.
    
    Args:
        results_base_path: Base path to results directory
        
    Returns:
        Dictionary mapping model names to list of result file info dicts
    """
    results = {}
    results_path = Path(results_base_path)
    
    if not results_path.exists():
        st.warning(f"Results directory not found: {results_path}")
        return results
    
    try:
        # Iterate through timestamp directories
        for timestamp_dir in sorted(results_path.iterdir(), reverse=True):
            if not timestamp_dir.is_dir():
                continue
            
            # Find all CSV files in this directory
            for csv_file in timestamp_dir.glob("*_results.csv"):
                # Extract model name from filename (e.g., "whisper-small_results.csv" -> "whisper-small")
                model_name = csv_file.stem.replace("_results", "")
                
                # Check for corresponding JSON file
                json_file = csv_file.with_suffix(".json")
                
                result_info = {
                    'model_name': model_name,
                    'csv_path': str(csv_file),
                    'json_path': str(json_file) if json_file.exists() else None,
                    'timestamp': timestamp_dir.name
                }
                
                # Group by model name
                if model_name not in results:
                    results[model_name] = []
                results[model_name].append(result_info)
        
        return results
        
    except Exception as e:
        st.error(f"Error scanning results directory: {e}")
        return {}


def combine_model_results(result_files: List[Dict[str, str]]) -> pd.DataFrame:
    """
    Combine multiple result files for the SAME model into a single DataFrame.
    This is used when a single model has multiple timestamp entries.
    
    Args:
        result_files: List of dictionaries with 'model_name' and 'csv_path'
        
    Returns:
        Combined DataFrame with model column added
        
    Raises:
        ValueError: If no valid results could be loaded
    """
    combined_data = []
    failed_models = []
    
    for result_file in result_files:
        model_name = result_file['model_name']
        csv_path = result_file['csv_path']
        
        try:
            df = load_data(csv_path)
            df['model'] = model_name
            combined_data.append(df)
            
        except (FileNotFoundError, ValueError, IOError) as e:
            failed_models.append(model_name)
            st.warning(f"Failed to load results for {model_name}: {e}")
            continue
    
    if not combined_data:
        raise ValueError(
            f"No valid results could be loaded. "
            f"Failed models: {', '.join(failed_models) if failed_models else 'None'}"
        )
    
    if failed_models:
        st.info(f"Loaded {len(combined_data)} model(s). Failed: {len(failed_models)}")
    
    # Combine all DataFrames
    combined_df = pd.concat(combined_data, ignore_index=True)
    
    # Reorder columns to have model first
    cols = ['model'] + [col for col in combined_df.columns if col != 'model']
    combined_df = combined_df[cols]
    
    return combined_df


def combine_multiple_models(
    selected_models: List[str],
    available_models: Dict[str, List[Dict[str, str]]]
) -> pd.DataFrame:
    """
    Combine data from multiple models into a single DataFrame for comparison.
    
    Args:
        selected_models: List of model names to load
        available_models: Dictionary mapping model names to their result files
        
    Returns:
        Combined DataFrame with 'model' column
        
    Raises:
        ValueError: If no valid models could be loaded
    """
    all_model_data = []
    failed_models = []
    
    # Define required columns for validation
    REQUIRED_COLUMNS = {'dialect', 'wer', 'cer', 'bleu'}
    
    for model_name in selected_models:
        if model_name not in available_models:
            st.warning(f"⚠️ Model '{model_name}' not found in available results")
            failed_models.append(model_name)
            continue
        
        model_files = available_models[model_name]
        
        try:
            # Use only most recent evaluation if multiple exist
            if len(model_files) > 1:
                st.info(
                    f"ℹ️ Model '{model_name}' has {len(model_files)} evaluation runs. "
                    f"Using most recent: {model_files[0]['timestamp']}"
                )
                # Keep only the first (most recent) file
                model_files = [model_files[0]]
            
            # Load the data
            df = load_data(model_files[0]['csv_path'])
            df['model'] = model_name
            
            # Validate schema
            if not REQUIRED_COLUMNS.issubset(df.columns):
                missing_cols = REQUIRED_COLUMNS - set(df.columns)
                st.warning(
                    f"⚠️ Model '{model_name}' missing required columns: {missing_cols}. Skipping."
                )
                failed_models.append(model_name)
                continue
            
            all_model_data.append(df)
                
        except (FileNotFoundError, ValueError, IOError) as e:
            st.warning(f"❌ Failed to load data for model '{model_name}': {e}")
            failed_models.append(model_name)
            continue
    
    if not all_model_data:
        # Improved error message with actionable guidance
        raise ValueError(
            f"Failed to load any of the {len(selected_models)} selected model(s). "
            f"Failed models: {', '.join(failed_models)}. "
            f"Ensure evaluation results exist in results/metrics/ directory."
        )
    
    if failed_models:
        st.info(
            f"✅ Successfully loaded {len(all_model_data)} model(s). "
            f"⚠️ Failed: {len(failed_models)} ({', '.join(failed_models)})"
        )
    
    # Combine all model DataFrames
    final_df = pd.concat(all_model_data, ignore_index=True)
    
    # Ensure model column is first
    cols = ['model'] + [col for col in final_df.columns if col != 'model']
    final_df = final_df[cols]
    
    return final_df