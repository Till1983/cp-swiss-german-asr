import streamlit as st
from pathlib import Path
from typing  import Dict, List
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


def get_available_results(results_base_path: str = "results/metrics") -> List[Dict[str, str]]:
    """
    Get list of all available result files in the results directory.
    
    Args:
        results_base_path: Base path to results directory
        
    Returns:
        List of dictionaries with 'model_name', 'csv_path', 'json_path', 'timestamp'
    """
    results = []
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
                
                results.append({
                    'model_name': model_name,
                    'csv_path': str(csv_file),
                    'json_path': str(json_file) if json_file.exists() else None,
                    'timestamp': timestamp_dir.name
                })
        
        return results
        
    except Exception as e:
        st.error(f"Error scanning results directory: {e}")
        return []


def combine_model_results(result_files: List[Dict[str, str]]) -> pd.DataFrame:
    """
    Combine multiple model results into a single DataFrame for comparison.
    
    Args:
        result_files: List of dictionaries with 'model_name' and 'csv_path'
        
    Returns:
        Combined DataFrame with model_name column added
        
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