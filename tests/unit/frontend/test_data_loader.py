"""
Unit tests for src.frontend.utils.data_loader module.

Tests data loading, caching, and result file aggregation functionality.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch
from src.frontend.utils.data_loader import (
    load_data,
    get_available_results,
    combine_model_results,
    combine_multiple_models
)


class TestLoadData:
    """Tests for load_data function."""
    
    def test_load_valid_csv(self):
        """Test loading a valid CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("dialect,wer,cer,bleu\n")
            f.write("Zurich,0.15,0.08,0.65\n")
            f.write("Bern,0.18,0.10,0.63\n")
            temp_path = f.name
        
        try:
            with patch('src.frontend.utils.data_loader.st.cache_data', lambda x: x):
                df = load_data(temp_path)
                assert len(df) == 2
                assert list(df.columns) == ['dialect', 'wer', 'cer', 'bleu']
                assert df.iloc[0]['dialect'] == 'Zurich'
        finally:
            Path(temp_path).unlink()
    
    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        with patch('src.frontend.utils.data_loader.st.cache_data', lambda x: x):
            with pytest.raises((FileNotFoundError, IOError)):
                load_data("/nonexistent/path/file.csv")
    
    def test_load_malformed_csv(self):
        """Test loading a CSV with missing required columns."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1,col2\n")
            f.write("val1,val2\n")
            temp_path = f.name
        
        try:
            with patch('src.frontend.utils.data_loader.st.cache_data', lambda x: x):
                with pytest.raises((ValueError, IOError)):
                    load_data(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_load_empty_csv(self):
        """Test loading an empty CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            with patch('src.frontend.utils.data_loader.st.cache_data', lambda x: x):
                with pytest.raises(ValueError, match="CSV file is empty"):
                    load_data(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_load_parser_error_csv(self):
        """Test loading a CSV with parser errors."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Write malformed CSV that will cause a parser error
            f.write('dialect,wer,cer,bleu\n')
            f.write('"Zurich,0.15,0.08,0.65\n')  # Unclosed quote
            temp_path = f.name
        
        try:
            with patch('src.frontend.utils.data_loader.st.cache_data', lambda x: x):
                # ParserError gets wrapped in IOError
                with pytest.raises((ValueError, IOError)):
                    load_data(temp_path)
        finally:
            Path(temp_path).unlink()


class TestGetAvailableResults:
    """Tests for get_available_results function."""
    
    def test_get_available_results_empty_directory(self):
        """Test with empty results directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('src.frontend.utils.data_loader.st'):
                results = get_available_results(str(tmpdir))
                assert results == {}
    
    def test_get_available_results_with_files(self):
        """Test retrieving available result files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create directory structure
            results_dir = Path(tmpdir) / "20251229_100000"
            results_dir.mkdir(parents=True)
            
            # Create result files
            (results_dir / "whisper-small_results.csv").write_text(
                "dialect,wer,cer,bleu\nZurich,0.15,0.08,0.65\n"
            )
            (results_dir / "whisper-small_results.json").write_text('{"config": "test"}')
            
            with patch('src.frontend.utils.data_loader.st'):
                results = get_available_results(tmpdir)
                assert 'whisper-small' in results


class TestCombineModelResults:
    """Tests for combine_model_results function."""
    
    def test_combine_single_result_file(self):
        """Test combining results from a single file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("dialect,wer,cer,bleu\n")
            f.write("Zurich,0.15,0.08,0.65\n")
            temp_path = f.name
        
        try:
            with patch('src.frontend.utils.data_loader.st'):
                with patch('src.frontend.utils.data_loader.st.cache_data', lambda x: x):
                    result_files = [{'model_name': 'whisper-small', 'csv_path': temp_path}]
                    df = combine_model_results(result_files)
                    assert 'model' in df.columns
                    assert df.iloc[0]['model'] == 'whisper-small'
        finally:
            Path(temp_path).unlink()
    
    def test_combine_multiple_result_files(self):
        """Test combining results from multiple files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f1:
            f1.write("dialect,wer,cer,bleu\n")
            f1.write("Zurich,0.15,0.08,0.65\n")
            temp_path1 = f1.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f2:
            f2.write("dialect,wer,cer,bleu\n")
            f2.write("Bern,0.18,0.10,0.63\n")
            temp_path2 = f2.name
        
        try:
            with patch('src.frontend.utils.data_loader.st'):
                with patch('src.frontend.utils.data_loader.st.cache_data', lambda x: x):
                    result_files = [
                        {'model_name': 'model-a', 'csv_path': temp_path1},
                        {'model_name': 'model-b', 'csv_path': temp_path2}
                    ]
                    df = combine_model_results(result_files)
                    assert len(df) == 2
                    assert 'model' in df.columns
        finally:
            Path(temp_path1).unlink()
            Path(temp_path2).unlink()
    
    def test_combine_with_invalid_file(self):
        """Test combining when one file doesn't exist."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("dialect,wer,cer,bleu\n")
            f.write("Zurich,0.15,0.08,0.65\n")
            temp_path = f.name
        
        try:
            with patch('src.frontend.utils.data_loader.st'):
                with patch('src.frontend.utils.data_loader.st.cache_data', lambda x: x):
                    result_files = [
                        {'model_name': 'valid-model', 'csv_path': temp_path},
                        {'model_name': 'invalid-model', 'csv_path': '/nonexistent/path.csv'}
                    ]
                    df = combine_model_results(result_files)
                    assert len(df) == 1
                    assert df.iloc[0]['model'] == 'valid-model'
        finally:
            Path(temp_path).unlink()


class TestCombineMultipleModels:
    """Tests for combine_multiple_models function."""
    
    def test_combine_single_model(self):
        """Test combining a single model's results."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("dialect,wer,cer,bleu\n")
            f.write("Zurich,0.15,0.08,0.65\n")
            temp_path = f.name
        
        try:
            available_models = {
                'whisper-small': [
                    {
                        'model_name': 'whisper-small',
                        'csv_path': temp_path,
                        'json_path': None,
                        'timestamp': '20251229_100000'
                    }
                ]
            }
            
            with patch('src.frontend.utils.data_loader.st'):
                with patch('src.frontend.utils.data_loader.st.cache_data', lambda x: x):
                    df = combine_multiple_models(['whisper-small'], available_models)
                    assert 'model' in df.columns
                    assert df.iloc[0]['model'] == 'whisper-small'
        finally:
            Path(temp_path).unlink()
    
    def test_combine_multiple_models(self):
        """Test combining multiple models' results."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f1:
            f1.write("dialect,wer,cer,bleu\n")
            f1.write("Zurich,0.15,0.08,0.65\n")
            temp_path1 = f1.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f2:
            f2.write("dialect,wer,cer,bleu\n")
            f2.write("Bern,0.18,0.10,0.63\n")
            temp_path2 = f2.name
        
        try:
            available_models = {
                'model-a': [{'model_name': 'model-a', 'csv_path': temp_path1, 'json_path': None, 'timestamp': '20251229_100000'}],
                'model-b': [{'model_name': 'model-b', 'csv_path': temp_path2, 'json_path': None, 'timestamp': '20251229_100000'}]
            }
            
            with patch('src.frontend.utils.data_loader.st'):
                with patch('src.frontend.utils.data_loader.st.cache_data', lambda x: x):
                    df = combine_multiple_models(['model-a', 'model-b'], available_models)
                    assert 'model' in df.columns
                    assert len(df) == 2
        finally:
            Path(temp_path1).unlink()
            Path(temp_path2).unlink()
    
    def test_combine_with_missing_model(self):
        """Test combining when a model is not found."""
        available_models = {}
        
        with patch('src.frontend.utils.data_loader.st'):
            with pytest.raises(ValueError):
                combine_multiple_models(['nonexistent-model'], available_models)
    
    def test_combine_with_multiple_timestamps_uses_most_recent(self):
        """Test that when a model has multiple evaluations, the most recent is used."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f1:
            f1.write("dialect,wer,cer,bleu\n")
            f1.write("Zurich,0.15,0.08,0.65\n")
            temp_path1 = f1.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f2:
            f2.write("dialect,wer,cer,bleu\n")
            f2.write("Zurich,0.20,0.12,0.60\n")
            temp_path2 = f2.name
        
        try:
            available_models = {
                'whisper-small': [
                    {'model_name': 'whisper-small', 'csv_path': temp_path1, 'json_path': None, 'timestamp': '20251229_100000'},
                    {'model_name': 'whisper-small', 'csv_path': temp_path2, 'json_path': None, 'timestamp': '20251228_100000'}
                ]
            }
            
            with patch('src.frontend.utils.data_loader.st'):
                with patch('src.frontend.utils.data_loader.st.cache_data', lambda x: x):
                    df = combine_multiple_models(['whisper-small'], available_models)
                    # Should use most recent (first in list)
                    assert df.iloc[0]['wer'] == 0.15
        finally:
            Path(temp_path1).unlink()
            Path(temp_path2).unlink()
    
    def test_combine_with_missing_required_columns_skips_model(self):
        """Test that models with missing required columns are skipped."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f1:
            f1.write("dialect,wer,cer\n")  # Missing 'bleu'
            f1.write("Zurich,0.15,0.08\n")
            temp_path1 = f1.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f2:
            f2.write("dialect,wer,cer,bleu\n")
            f2.write("Bern,0.18,0.10,0.63\n")
            temp_path2 = f2.name
        
        try:
            available_models = {
                'incomplete-model': [{'model_name': 'incomplete-model', 'csv_path': temp_path1, 'json_path': None, 'timestamp': '20251229_100000'}],
                'complete-model': [{'model_name': 'complete-model', 'csv_path': temp_path2, 'json_path': None, 'timestamp': '20251229_100000'}]
            }
            
            with patch('src.frontend.utils.data_loader.st'):
                with patch('src.frontend.utils.data_loader.st.cache_data', lambda x: x):
                    df = combine_multiple_models(['incomplete-model', 'complete-model'], available_models)
                    # Should only have complete-model
                    assert len(df) == 1
                    assert df.iloc[0]['model'] == 'complete-model'
        finally:
            Path(temp_path1).unlink()
            Path(temp_path2).unlink()
    
    def test_combine_with_file_load_error(self):
        """Test combining when a model file cannot be loaded."""
        available_models = {
            'broken-model': [
                {'model_name': 'broken-model', 'csv_path': '/nonexistent/path.csv', 'json_path': None, 'timestamp': '20251229_100000'}
            ]
        }
        
        with patch('src.frontend.utils.data_loader.st'):
            with patch('src.frontend.utils.data_loader.st.cache_data', lambda x: x):
                with pytest.raises(ValueError, match="Failed to load any of the"):
                    combine_multiple_models(['broken-model'], available_models)
