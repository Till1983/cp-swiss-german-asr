import streamlit as st
from pathlib import Path
from typing  import Dict, List
import pandas as pd

@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load a CSV results file with caching.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing the results
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        pd.errors.EmptyDataError: If the file is empty
    """
    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        df = pd.read_csv(path)
        if df.empty:
            raise pd.errors.EmptyDataError(f"File is empty: {file_path}")
        
        return df
    except Exception as e:
        st.error(f"Error loading {file_path}: {str(e)}")
        raise


def get_available_results(results_dir: str = "results/metrics") -> Dict[str, List[Path]]:
    """
    Get all available results files grouped by model name.
    
    Args:
        results_dir: Path to the results directory
        
    Returns:
        Dictionary mapping model names to list of result file paths
    """
    results_path = Path(results_dir)
    model_files = {}
    
    try:
        for csv_file in results_path.rglob("*_results.csv"):
            model_name = csv_file.stem.replace("_results", "")
            if model_name not in model_files:
                model_files[model_name] = []
            model_files[model_name].append(csv_file)
    except Exception as e:
        st.warning(f"Error scanning results directory: {str(e)}")
        return {}
    
    return model_files


def combine_model_results(file_paths: List[Path]) -> pd.DataFrame:
    """
    Combine multiple result files into a single DataFrame.
    
    Args:
        file_paths: List of paths to CSV files
        
    Returns:
        Combined DataFrame with all results
    """
    dfs = []
    
    for file_path in file_paths:
        try:
            df = load_data(str(file_path))
            df['source_file'] = file_path.parent.name
            dfs.append(df)
        except Exception as e:
            st.warning(f"Skipping {file_path.name}: {str(e)}")
            continue
    
    if not dfs:
        return pd.DataFrame()
    
    return pd.concat(dfs, ignore_index=True)