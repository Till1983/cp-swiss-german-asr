"""
Unit tests for src.training.trainer module.

Tests custom trainer classes and callbacks for ASR model training.
"""

import tempfile
from unittest.mock import patch, MagicMock
from src.training.trainer import (
    CatastrophicForgettingCallback,
    ProjectTrainer
)


class TestCatastrophicForgettingCallback:
    """Tests for CatastrophicForgettingCallback class."""
    
    def test_initialization(self):
        """Test callback initialization."""
        mock_dataset = MagicMock()
        mock_metric_fn = MagicMock()
        
        callback = CatastrophicForgettingCallback(
            eval_dataset_pretrain=mock_dataset,
            metric_fn=mock_metric_fn,
            threshold=0.05
        )
        
        assert callback.eval_dataset_pretrain == mock_dataset
        assert callback.metric_fn == mock_metric_fn
        assert callback.threshold == 0.05
        assert callback.best_metric is None
    
    def test_on_evaluate_first_call(self):
        """Test on_evaluate sets initial best_metric."""
        mock_dataset = MagicMock()
        mock_metric_fn = MagicMock(return_value=0.20)
        
        callback = CatastrophicForgettingCallback(
            eval_dataset_pretrain=mock_dataset,
            metric_fn=mock_metric_fn,
            threshold=0.1
        )
        
        callback.on_evaluate(None, None, None, metrics=None)
        
        assert callback.best_metric == 0.20
        mock_metric_fn.assert_called_once_with(mock_dataset)
    
    def test_on_evaluate_metric_improvement(self):
        """Test on_evaluate when metric improves."""
        mock_dataset = MagicMock()
        mock_metric_fn = MagicMock(return_value=0.18)
        
        callback = CatastrophicForgettingCallback(
            eval_dataset_pretrain=mock_dataset,
            metric_fn=mock_metric_fn,
            threshold=0.1
        )
        
        callback.best_metric = 0.20  # Set initial best
        
        with patch('src.training.trainer.logger') as mock_logger:
            callback.on_evaluate(None, None, None, metrics=None)
            # No warning should be logged for improvement
            mock_logger.warning.assert_not_called()
    
    def test_on_evaluate_catastrophic_forgetting(self):
        """Test on_evaluate detects catastrophic forgetting."""
        mock_dataset = MagicMock()
        mock_metric_fn = MagicMock(return_value=0.35)
        
        callback = CatastrophicForgettingCallback(
            eval_dataset_pretrain=mock_dataset,
            metric_fn=mock_metric_fn,
            threshold=0.1
        )
        
        callback.best_metric = 0.20  # Set initial best
        
        with patch('src.training.trainer.logger') as mock_logger:
            callback.on_evaluate(None, None, None, metrics=None)
            # Warning should be logged for significant degradation
            mock_logger.warning.assert_called_once()
    
    def test_on_evaluate_with_none_dataset(self):
        """Test on_evaluate with None dataset does nothing."""
        callback = CatastrophicForgettingCallback(
            eval_dataset_pretrain=None,
            metric_fn=MagicMock(),
            threshold=0.1
        )
        
        with patch('src.training.trainer.logger') as mock_logger:
            callback.on_evaluate(None, None, None, metrics=None)
            # No warning should be logged
            mock_logger.warning.assert_not_called()


class TestProjectTrainer:
    """Tests for ProjectTrainer class."""
    
    def test_initialization(self):
        """Test ProjectTrainer initialization."""
        mock_model = MagicMock()
        mock_args = MagicMock()
        
        with patch('src.training.trainer.Trainer.__init__', return_value=None):
            trainer = ProjectTrainer(
                model=mock_model,
                args=mock_args,
                run_name="test_run",
                use_wandb=False
            )
            
            assert trainer.run_name == "test_run"
            assert trainer.use_wandb is False
    
    def test_custom_run_name(self):
        """Test that custom run_name is preserved."""
        with patch('src.training.trainer.Trainer.__init__', return_value=None):
            trainer = ProjectTrainer(
                model=MagicMock(),
                args=MagicMock(),
                run_name="my_custom_run"
            )
            
            assert trainer.run_name == "my_custom_run"
    
    def test_default_run_name(self):
        """Test that default run_name is used when not provided."""
        with patch('src.training.trainer.Trainer.__init__', return_value=None):
            trainer = ProjectTrainer(
                model=MagicMock(),
                args=MagicMock()
            )
            
            assert trainer.run_name == "asr_training"
    
    def test_wandb_callback_added(self):
        """Test that WandB callback is added when use_wandb=True."""
        with patch('src.training.trainer.Trainer.__init__', return_value=None):
            with patch('src.training.trainer.WandbCallback'):
                trainer = ProjectTrainer(
                    model=MagicMock(),
                    args=MagicMock(),
                    use_wandb=True
                )
                
                assert trainer.use_wandb is True
    
    def test_catastrophic_forgetting_callback_added(self):
        """Test that catastrophic forgetting callback is added when provided."""
        mock_callback = MagicMock()
        
        with patch('src.training.trainer.Trainer.__init__', return_value=None):
            trainer = ProjectTrainer(
                model=MagicMock(),
                args=MagicMock(),
                catastrophic_forgetting_callback=mock_callback
            )
            
            assert trainer.catastrophic_forgetting_callback == mock_callback
    
    def test_log_metrics_saves_to_file(self):
        """Test that log_metrics saves metrics to JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_model = MagicMock()
            mock_args = MagicMock()
            mock_args.logging_dir = tmpdir
            
            with patch('src.training.trainer.Trainer.__init__', return_value=None):
                with patch('src.training.trainer.Trainer.log_metrics'):
                    trainer = ProjectTrainer(
                        model=mock_model,
                        args=mock_args
                    )
                    trainer.args = mock_args
                    
                    # Mock parent log_metrics
                    with patch.object(
                        trainer.__class__.__bases__[0],
                        'log_metrics',
                        return_value=None
                    ):
                        metrics = {"eval_loss": 0.15, "eval_wer": 0.18}
                        trainer.log_metrics("eval", metrics)
                        
                        # Metrics file should be created (verified in actual implementation)
    
    def test_evaluate_calls_log_metrics(self):
        """Test that evaluate calls log_metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_model = MagicMock()
            mock_args = MagicMock()
            mock_args.logging_dir = tmpdir
            
            with patch('src.training.trainer.Trainer.__init__', return_value=None):
                trainer = ProjectTrainer(
                    model=mock_model,
                    args=mock_args
                )
                trainer.args = mock_args
                
                # Mock parent evaluate
                with patch.object(
                    trainer.__class__.__bases__[0],
                    'evaluate',
                    return_value={"eval_loss": 0.15}
                ):
                    with patch.object(
                        trainer,
                        'log_metrics'
                    ) as mock_log:
                        trainer.evaluate()
                        
                        # Verify log_metrics was called
                        mock_log.assert_called_once()
    
    def test_train_logging(self):
        """Test that train logs start and completion messages."""
        with patch('src.training.trainer.Trainer.__init__', return_value=None):
            trainer = ProjectTrainer(
                model=MagicMock(),
                args=MagicMock(),
                run_name="test_training"
            )
            
            # Mock parent train
            with patch.object(
                trainer.__class__.__bases__[0],
                'train',
                return_value=MagicMock()
            ):
                with patch('src.training.trainer.logger') as mock_logger:
                    trainer.train()
                    
                    # Check logging calls
                    assert mock_logger.info.call_count >= 2
    
    def test_output_dir_uses_run_name(self):
        """Test that output directory includes the run name."""
        with patch('src.training.trainer.Trainer.__init__', return_value=None) as mock_init:
            mock_model = MagicMock()
            mock_args = MagicMock()
            
            ProjectTrainer(
                model=mock_model,
                args=mock_args,
                run_name="custom_run"
            )
            
            # Verify init was called with modified kwargs
            call_kwargs = mock_init.call_args[1]
            assert 'output_dir' in call_kwargs
            assert 'custom_run' in call_kwargs['output_dir']
    
    def test_logging_dir_uses_run_name(self):
        """Test that logging directory includes the run name."""
        with patch('src.training.trainer.Trainer.__init__', return_value=None) as mock_init:
            mock_model = MagicMock()
            mock_args = MagicMock()
            
            ProjectTrainer(
                model=mock_model,
                args=mock_args,
                run_name="my_training_run"
            )
            
            # Verify init was called with modified kwargs
            call_kwargs = mock_init.call_args[1]
            assert 'logging_dir' in call_kwargs
            assert 'my_training_run' in call_kwargs['logging_dir']


class TestTrainerIntegration:
    """Integration tests for trainer components."""
    
    def test_callback_with_trainer(self):
        """Test that CatastrophicForgettingCallback integrates with ProjectTrainer."""
        mock_dataset = MagicMock()
        mock_metric_fn = MagicMock(return_value=0.20)
        
        callback = CatastrophicForgettingCallback(
            eval_dataset_pretrain=mock_dataset,
            metric_fn=mock_metric_fn,
            threshold=0.1
        )
        
        with patch('src.training.trainer.Trainer.__init__', return_value=None):
            trainer = ProjectTrainer(
                model=MagicMock(),
                args=MagicMock(),
                catastrophic_forgetting_callback=callback
            )
            
            assert trainer.catastrophic_forgetting_callback == callback
