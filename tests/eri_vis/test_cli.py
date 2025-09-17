"""
Tests for ERI CLI interface.

This module tests the command-line interface for ERI visualization,
including argument parsing, data loading, processing, and visualization
generation with various configurations.
"""

import argparse
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

# Import the CLI module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.plot_eri import ERICLI, ERICLIError
from eri_vis.dataset import ERITimelineDataset


class TestERICLI:
    """Test cases for ERICLI class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cli = ERICLI()
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def create_sample_csv(self, filename: str = "test_data.csv") -> Path:
        """Create a sample CSV file for testing."""
        csv_path = self.temp_dir / filename

        # Create sample data
        data = []
        methods = ['Scratch_T2', 'sgd', 'ewc_on']
        splits = ['T2_shortcut_normal', 'T2_shortcut_masked']
        seeds = [42, 43]
        epochs = [0.0, 1.0, 2.0, 3.0, 4.0]

        for method in methods:
            for split in splits:
                for seed in seeds:
                    for epoch in epochs:
                        # Create realistic accuracy progression
                        if method == 'Scratch_T2':
                            base_acc = 0.1 if 'normal' in split else 0.05
                            acc = base_acc + epoch * 0.15
                        else:
                            base_acc = 0.2 if 'normal' in split else 0.1
                            acc = base_acc + epoch * 0.1

                        acc = min(acc, 0.95)  # Cap at 95%

                        data.append({
                            'method': method,
                            'seed': seed,
                            'epoch_eff': epoch,
                            'split': split,
                            'acc': acc
                        })

        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_create_parser(self):
        """Test argument parser creation."""
        parser = self.cli.create_parser()

        assert isinstance(parser, argparse.ArgumentParser)

        # Test required arguments
        with pytest.raises(SystemExit):
            parser.parse_args([])  # Missing required args

        # Test minimal valid arguments
        args = parser.parse_args(['--csv', 'test.csv', '--outdir', 'output/'])
        assert args.csv == 'test.csv'
        assert args.outdir == 'output/'
        assert args.tau == 0.6  # Default value
        assert args.smooth == 3  # Default value

    def test_validate_args_valid(self):
        """Test argument validation with valid arguments."""
        parser = self.cli.create_parser()
        args = parser.parse_args([
            '--csv', 'test.csv',
            '--outdir', 'output/',
            '--tau', '0.65',
            '--smooth', '5'
        ])

        # Should not raise any exception
        self.cli.validate_args(args)

    def test_validate_args_invalid_tau(self):
        """Test argument validation with invalid tau value."""
        parser = self.cli.create_parser()
        args = parser.parse_args([
            '--csv', 'test.csv',
            '--outdir', 'output/',
            '--tau', '1.5'  # Invalid: > 1.0
        ])

        with pytest.raises(ERICLIError, match="Tau value must be between 0.0 and 1.0"):
            self.cli.validate_args(args)

    def test_validate_args_invalid_smooth(self):
        """Test argument validation with invalid smoothing window."""
        parser = self.cli.create_parser()
        args = parser.parse_args([
            '--csv', 'test.csv',
            '--outdir', 'output/',
            '--smooth', '0'  # Invalid: < 1
        ])

        with pytest.raises(ERICLIError, match="Smoothing window must be >= 1"):
            self.cli.validate_args(args)

    def test_validate_args_invalid_tau_grid(self):
        """Test argument validation with invalid tau grid."""
        parser = self.cli.create_parser()
        args = parser.parse_args([
            '--csv', 'test.csv',
            '--outdir', 'output/',
            '--tau-grid', '0.5', '1.2', '0.7'  # Invalid: 1.2 > 1.0
        ])

        with pytest.raises(ERICLIError, match="All tau grid values must be between 0.0 and 1.0"):
            self.cli.validate_args(args)

    def test_validate_args_conflicting_flags(self):
        """Test argument validation with conflicting quiet/verbose flags."""
        parser = self.cli.create_parser()
        args = parser.parse_args([
            '--csv', 'test.csv',
            '--outdir', 'output/',
            '--quiet',
            '--verbose'
        ])

        with pytest.raises(ERICLIError, match="Cannot specify both --quiet and --verbose"):
            self.cli.validate_args(args)

    def test_find_csv_files_single_file(self):
        """Test finding single CSV file."""
        csv_path = self.create_sample_csv()

        files = self.cli.find_csv_files(str(csv_path))

        assert len(files) == 1
        assert files[0] == csv_path

    def test_find_csv_files_glob_pattern(self):
        """Test finding CSV files with glob pattern."""
        # Create multiple CSV files
        csv1 = self.create_sample_csv("data1.csv")
        csv2 = self.create_sample_csv("data2.csv")

        # Create non-CSV file (should be ignored)
        (self.temp_dir / "data3.txt").write_text("not a csv")

        pattern = str(self.temp_dir / "data*.csv")
        files = self.cli.find_csv_files(pattern)

        assert len(files) == 2
        assert csv1 in files
        assert csv2 in files

    def test_find_csv_files_not_found(self):
        """Test error when CSV file not found."""
        with pytest.raises(ERICLIError, match="CSV file not found"):
            self.cli.find_csv_files("nonexistent.csv")

    def test_find_csv_files_no_matches(self):
        """Test error when glob pattern matches no files."""
        pattern = str(self.temp_dir / "*.csv")

        with pytest.raises(ERICLIError, match="No CSV files found matching pattern"):
            self.cli.find_csv_files(pattern)

    def test_load_datasets_success(self):
        """Test successful dataset loading."""
        csv_path = self.create_sample_csv()

        datasets = self.cli.load_datasets([csv_path])

        assert len(datasets) == 1
        file_path, dataset = datasets[0]
        assert file_path == csv_path
        assert isinstance(dataset, ERITimelineDataset)
        assert len(dataset.methods) == 3  # Scratch_T2, sgd, ewc_on
        assert len(dataset.splits) == 2   # shortcut_normal, shortcut_masked

    def test_load_datasets_invalid_csv(self):
        """Test dataset loading with invalid CSV."""
        # Create invalid CSV
        invalid_csv = self.temp_dir / "invalid.csv"
        invalid_csv.write_text("invalid,csv,content\n1,2")  # Malformed

        datasets = self.cli.load_datasets([invalid_csv])

        # Should return empty list (errors logged but not raised)
        assert len(datasets) == 0

    def test_filter_methods_valid(self):
        """Test method filtering with valid methods."""
        csv_path = self.create_sample_csv()
        datasets = self.cli.load_datasets([csv_path])
        _, dataset = datasets[0]

        filtered = self.cli.filter_methods(dataset, ['sgd', 'ewc_on'])

        assert len(filtered.methods) == 2
        assert 'sgd' in filtered.methods
        assert 'ewc_on' in filtered.methods
        assert 'Scratch_T2' not in filtered.methods

    def test_filter_methods_none(self):
        """Test method filtering with None (no filtering)."""
        csv_path = self.create_sample_csv()
        datasets = self.cli.load_datasets([csv_path])
        _, dataset = datasets[0]

        filtered = self.cli.filter_methods(dataset, None)

        assert filtered == dataset  # Should be unchanged

    def test_filter_methods_invalid(self):
        """Test method filtering with invalid methods."""
        csv_path = self.create_sample_csv()
        datasets = self.cli.load_datasets([csv_path])
        _, dataset = datasets[0]

        with pytest.raises(ERICLIError, match="None of the requested methods found"):
            self.cli.filter_methods(dataset, ['nonexistent_method'])

    @patch('tools.plot_eri.ERIDynamicsPlotter')
    @patch('tools.plot_eri.ERITimelineProcessor')
    def test_generate_dynamics_plot_success(self, mock_processor_class, mock_plotter_class):
        """Test successful dynamics plot generation."""
        # Set up mocks
        mock_processor = Mock()
        mock_processor.compute_accuracy_curves.return_value = {
            'sgd_T2_shortcut_normal': Mock(),
            'sgd_T2_shortcut_masked': Mock()
        }
        mock_processor.compute_adaptation_delays.return_value = {'sgd': 1.5}
        mock_processor.compute_performance_deficits.return_value = {'sgd': Mock()}
        mock_processor.compute_sfr_relative.return_value = {'sgd': Mock()}
        mock_processor.tau = 0.6

        mock_plotter = Mock()
        mock_fig = Mock()
        mock_plotter.create_dynamics_figure.return_value = mock_fig

        self.cli.processor = mock_processor
        self.cli.dynamics_plotter = mock_plotter

        # Create test dataset
        csv_path = self.create_sample_csv()
        datasets = self.cli.load_datasets([csv_path])
        _, dataset = datasets[0]

        # Generate plot
        output_dir = self.temp_dir / "output"
        output_dir.mkdir()

        result = self.cli.generate_dynamics_plot(dataset, output_dir)

        assert result is not None
        assert result.name == "fig_eri_dynamics.pdf"
        mock_plotter.create_dynamics_figure.assert_called_once()
        mock_plotter.save_figure.assert_called_once()

    @patch('tools.plot_eri.ERIHeatmapPlotter')
    @patch('tools.plot_eri.ERITimelineProcessor')
    def test_generate_heatmap_success(self, mock_processor_class, mock_plotter_class):
        """Test successful heatmap generation."""
        # Set up mocks
        mock_processor = Mock()
        mock_processor.compute_accuracy_curves.return_value = {
            'Scratch_T2_T2_shortcut_normal': Mock(),
            'sgd_T2_shortcut_normal': Mock()
        }

        mock_plotter = Mock()
        mock_fig = Mock()
        mock_plotter.create_method_comparison_heatmap.return_value = mock_fig

        self.cli.processor = mock_processor
        self.cli.heatmap_plotter = mock_plotter

        # Create test dataset
        csv_path = self.create_sample_csv()
        datasets = self.cli.load_datasets([csv_path])
        _, dataset = datasets[0]

        # Generate heatmap
        output_dir = self.temp_dir / "output"
        output_dir.mkdir()

        tau_grid = [0.5, 0.6, 0.7]
        result = self.cli.generate_heatmap(dataset, output_dir, tau_grid)

        assert result is not None
        assert result.name == "fig_ad_tau_heatmap.pdf"
        mock_plotter.create_method_comparison_heatmap.assert_called_once()
        mock_plotter.save_heatmap.assert_called_once()

    def test_setup_components(self):
        """Test component setup with user parameters."""
        parser = self.cli.create_parser()
        args = parser.parse_args([
            '--csv', 'test.csv',
            '--outdir', 'output/',
            '--tau', '0.65',
            '--smooth', '5',
            '--dpi', '150'
        ])

        self.cli.setup_components(args)

        assert self.cli.processor is not None
        assert self.cli.processor.tau == 0.65
        assert self.cli.processor.smoothing_window == 5
        assert self.cli.dynamics_plotter is not None
        assert self.cli.heatmap_plotter is not None

    @patch('tools.plot_eri.ERICLI.generate_dynamics_plot')
    @patch('tools.plot_eri.ERICLI.generate_heatmap')
    def test_process_single_dataset(self, mock_gen_heatmap, mock_gen_dynamics):
        """Test processing single dataset."""
        # Set up mocks
        mock_gen_dynamics.return_value = Path("dynamics.pdf")
        mock_gen_heatmap.return_value = Path("heatmap.pdf")

        # Create test data
        csv_path = self.create_sample_csv()
        datasets = self.cli.load_datasets([csv_path])
        _, dataset = datasets[0]

        # Set up args
        parser = self.cli.create_parser()
        args = parser.parse_args([
            '--csv', str(csv_path),
            '--outdir', str(self.temp_dir),
            '--tau-grid', '0.5', '0.6', '0.7'
        ])

        output_dir = self.temp_dir / "output"
        output_dir.mkdir()

        results = self.cli.process_single_dataset(csv_path, dataset, args, output_dir)

        assert 'dynamics' in results
        assert 'heatmap' in results
        assert results['dynamics'] == Path("dynamics.pdf")
        assert results['heatmap'] == Path("heatmap.pdf")

    def test_run_success_single_file(self):
        """Test successful CLI run with single file."""
        csv_path = self.create_sample_csv()
        output_dir = self.temp_dir / "output"

        with patch.object(self.cli, 'generate_dynamics_plot') as mock_dynamics:
            mock_dynamics.return_value = output_dir / "dynamics.pdf"

            args = [
                '--csv', str(csv_path),
                '--outdir', str(output_dir),
                '--quiet'  # Suppress logging for test
            ]

            exit_code = self.cli.run(args)

            assert exit_code == 0
            assert output_dir.exists()

    def test_run_file_not_found(self):
        """Test CLI run with non-existent file."""
        args = [
            '--csv', 'nonexistent.csv',
            '--outdir', str(self.temp_dir),
            '--quiet'
        ]

        exit_code = self.cli.run(args)

        assert exit_code == 1

    def test_run_invalid_arguments(self):
        """Test CLI run with invalid arguments."""
        args = [
            '--csv', 'test.csv',
            '--outdir', 'output/',
            '--tau', '2.0',  # Invalid tau value
            '--quiet'
        ]

        exit_code = self.cli.run(args)

        assert exit_code == 1

    def test_run_keyboard_interrupt(self):
        """Test CLI run with keyboard interrupt."""
        csv_path = self.create_sample_csv()

        with patch.object(self.cli, 'load_datasets') as mock_load:
            mock_load.side_effect = KeyboardInterrupt()

            args = [
                '--csv', str(csv_path),
                '--outdir', str(self.temp_dir),
                '--quiet'
            ]

            exit_code = self.cli.run(args)

            assert exit_code == 1

    def test_report_results(self):
        """Test results reporting."""
        results = {
            Path("data1.csv"): {
                'dynamics': Path("dynamics1.pdf"),
                'heatmap': Path("heatmap1.pdf")
            },
            Path("data2.csv"): {
                'dynamics': Path("dynamics2.pdf"),
                'heatmap': None  # Failed
            }
        }

        # Should not raise any exception
        self.cli.report_results(results)


class TestCLIIntegration:
    """Integration tests for CLI functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def create_realistic_csv(self) -> Path:
        """Create a realistic CSV file for integration testing."""
        csv_path = self.temp_dir / "realistic_data.csv"

        # Create more realistic data with proper progression
        data = []
        methods = ['Scratch_T2', 'sgd', 'ewc_on', 'derpp']
        splits = ['T1_all', 'T2_shortcut_normal', 'T2_shortcut_masked', 'T2_nonshortcut_normal']
        seeds = [42, 43, 44]
        epochs = np.linspace(0, 10, 21)  # 0 to 10 epochs

        for method in methods:
            for split in splits:
                for seed in seeds:
                    # Add some noise for realism
                    np.random.seed(seed)
                    noise = np.random.normal(0, 0.02, len(epochs))

                    for i, epoch in enumerate(epochs):
                        # Create realistic accuracy progression
                        if split == 'T1_all':
                            # High accuracy on T1
                            base_acc = 0.85
                            acc = base_acc + noise[i]
                        elif split == 'T2_shortcut_normal':
                            if method == 'Scratch_T2':
                                # Scratch learns shortcut quickly
                                acc = 0.1 + 0.7 * (1 - np.exp(-epoch/2)) + noise[i]
                            else:
                                # CL methods learn more slowly
                                acc = 0.2 + 0.5 * (1 - np.exp(-epoch/4)) + noise[i]
                        elif split == 'T2_shortcut_masked':
                            # Lower accuracy when shortcut is masked
                            if method == 'Scratch_T2':
                                acc = 0.05 + 0.1 * (1 - np.exp(-epoch/3)) + noise[i]
                            else:
                                acc = 0.1 + 0.2 * (1 - np.exp(-epoch/5)) + noise[i]
                        else:  # T2_nonshortcut_normal
                            # Moderate accuracy on non-shortcut samples
                            acc = 0.3 + 0.4 * (1 - np.exp(-epoch/3)) + noise[i]

                        acc = np.clip(acc, 0.0, 1.0)

                        data.append({
                            'method': method,
                            'seed': seed,
                            'epoch_eff': epoch,
                            'split': split,
                            'acc': acc
                        })

        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_end_to_end_single_file(self):
        """Test end-to-end CLI execution with single file."""
        csv_path = self.create_realistic_csv()
        output_dir = self.temp_dir / "output"

        cli = ERICLI()

        args = [
            '--csv', str(csv_path),
            '--outdir', str(output_dir),
            '--tau', '0.6',
            '--smooth', '3',
            '--tau-grid', '0.5', '0.6', '0.7',
            '--quiet'
        ]

        exit_code = cli.run(args)

        assert exit_code == 0
        assert output_dir.exists()

        # Check that output files were created
        output_files = list(output_dir.glob("*.pdf"))
        assert len(output_files) >= 1  # At least dynamics plot

    def test_end_to_end_batch_processing(self):
        """Test end-to-end CLI execution with batch processing."""
        # Create multiple CSV files
        csv1 = self.create_realistic_csv()
        csv2_path = self.temp_dir / "data2.csv"

        # Copy and modify for second file
        df = pd.read_csv(csv1)
        df['seed'] = df['seed'] + 10  # Different seeds
        df.to_csv(csv2_path, index=False)

        output_dir = self.temp_dir / "output"

        cli = ERICLI()

        args = [
            '--csv', str(self.temp_dir / "*.csv"),
            '--outdir', str(output_dir),
            '--batch-summary',
            '--quiet'
        ]

        exit_code = cli.run(args)

        assert exit_code == 0
        assert output_dir.exists()

        # Check that output files were created for both datasets
        output_files = list(output_dir.glob("*.pdf"))
        assert len(output_files) >= 2  # At least one per dataset

    def test_method_filtering(self):
        """Test CLI with method filtering."""
        csv_path = self.create_realistic_csv()
        output_dir = self.temp_dir / "output"

        cli = ERICLI()

        args = [
            '--csv', str(csv_path),
            '--outdir', str(output_dir),
            '--methods', 'sgd', 'ewc_on',  # Filter to specific methods
            '--quiet'
        ]

        exit_code = cli.run(args)

        assert exit_code == 0

    def test_different_output_formats(self):
        """Test CLI with different output formats."""
        csv_path = self.create_realistic_csv()
        output_dir = self.temp_dir / "output"

        cli = ERICLI()

        # Test PNG format
        args = [
            '--csv', str(csv_path),
            '--outdir', str(output_dir),
            '--format', 'png',
            '--quiet'
        ]

        exit_code = cli.run(args)

        assert exit_code == 0

        # Check that PNG files were created
        png_files = list(output_dir.glob("*.png"))
        assert len(png_files) >= 1


if __name__ == '__main__':
    pytest.main([__file__])
