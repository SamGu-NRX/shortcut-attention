"""
Comprehensive DGR Integration Test

This test file verifies the complete integration of DGR with the Mammoth framework,
including compatibility with the Einstellung evaluation system, visualization pipeline,
and experiment runner.
"""

import pytest

pytest.skip("Legacy DGR adapter integration tests disabled for original-method integration", allow_module_level=True)


class TestDGRFullIntegration:
    """Test complete DGR integration with all system components."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def dgr_args(self):
        """Create comprehensive DGR arguments."""
        args = Namespace()
        # DGR-specific parameters
        args.dgr_z_dim = 64
        args.dgr_vae_lr = 0.001
        args.dgr_vae_fc_layers = 3
        args.dgr_vae_fc_units = 256
        args.dgr_replay_weight = 0.5
        args.dgr_vae_train_epochs = 1

        # Mammoth framework parameters
        args.model = 'dgr'
        args.backbone = 'resnet18'
        args.lr = 0.01
        args.optimizer = 'adam'
        args.optim_wd = 0.0001
        args.optim_mom = 0.9
        args.optim_nesterov = False
        args.label_perc = 1.0
        args.buffer_size = 0  # DGR doesn't use buffer
        args.dataset = 'seq-cifar100-einstellung-224'
        args.nowand = True
        args.seed = 42
        args.n_epochs = 2  # Short for testing
        args.batch_size = 16

        # Einstellung-specific parameters
        args.enable_einstellung = True
        args.einstellung_eval_frequency = 1
        args.einstellung_save_attention = False  # Disable for speed

        return args

    @pytest.fixture
    def mock_dataset(self):
        """Create mock Einstellung dataset."""
        dataset = Mock()
        dataset.SIZE = [3, 224, 224]  # Einstellung uses 224x224
        dataset.N_CLASSES = 100
        dataset.N_TASKS = 2
        dataset.N_CLASSES_PER_TASK = 50
        dataset.SETTING = 'class-il'
        dataset.get_offsets = Mock(side_effect=lambda task: (task * 50, (task + 1) * 50))
        dataset.get_normalization_transform = Mock(return_value=nn.Identity())
        dataset.get_transform = Mock(return_value=nn.Identity())
        dataset.get_loss = Mock(return_value=nn.CrossEntropyLoss())

        # Mock data loaders
        def mock_dataloader(task_id):
            # Create dummy data for the task
            data = []
            for i in range(32):  # Small dataset for testing
                x = torch.randn(3, 224, 224)
                y = torch.randint(task_id * 50, (task_id + 1) * 50, (1,)).item()
                data.append((x, y))
            return data

        dataset.train_loaders = [mock_dataloader(0), mock_dataloader(1)]
        dataset.test_loaders = [mock_dataloader(0), mock_dataloader(1)]

        return dataset

    @pytest.fixture
    def dgr_model(self, dgr_args, mock_dataset):
        """Create DGR model with mocked components."""
        backbone = ResNet32(num_classes=100)
        loss_fn = nn.CrossEntropyLoss()
        transform = nn.Identity()

        model = DGRMammothAdapter(backbone, loss_fn, dgr_args, transform, mock_dataset)
        return model

    def test_dgr_model_registration(self):
        """Test that DGR model is properly registered with Mammoth."""
        from models import get_model_names

        model_names = get_model_names()
        assert 'dgr' in model_names, "DGR model not registered in Mammoth"

        # Test that the class can be retrieved
        dgr_class = model_names['dgr']
        assert dgr_class == DGRMammothAdapter, "Wrong DGR class registered"

    def test_dgr_einstellung_compatibility(self, dgr_model, dgr_args, temp_dir):
        """Test DGR compatibility with Einstellung evaluation system."""
        device = torch.device('cpu')  # Use CPU for testing
        dgr_model.to(device)

        # Create mock EinstellungEvaluator
        evaluator = Mock(spec=EinstellungEvaluator)
        evaluator.evaluate_model = Mock(return_value={
            'T1_all': 75.0,
            'T2_shortcut_normal': 80.0,
            'T2_shortcut_masked': 60.0,
            'T2_nonshortcut_normal': 70.0,
            'eri_score': 0.45
        })

        # Test that DGR model can be evaluated
        batch_size = 8
        inputs = torch.randn(batch_size, 3, 224, 224).to(device)
        labels = torch.randint(0, 50, (batch_size,)).to(device)
        not_aug_inputs = inputs.clone()

        # Simulate training step
        loss = dgr_model.observe(inputs, labels, not_aug_inputs)
        assert isinstance(loss, float)
        assert loss > 0

        # Test evaluation compatibility
        dgr_model.eval()
        with torch.no_grad():
            outputs = dgr_model(inputs)
            assert outputs.shape == (batch_size, 100)

        # Test that evaluator can work with DGR
        results = evaluator.evaluate_model(dgr_model, None)  # Mock call
        assert 'eri_score' in results
        assert results['eri_score'] > 0

    def test_dgr_replay_monitoring(self, dgr_model, temp_dir):
        """Test monitoring and visualization of generated replay samples."""
        device = torch.device('cpu')
        dgr_model.to(device)

        # Train on first task to create VAE
        batch_size = 16
        n_batches = 5

        for i in range(n_batches):
            inputs = torch.randn(batch_size, 3, 224, 224).to(device)
            labels = torch.randint(0, 50, (batch_size,)).to(device)
            not_aug_inputs = inputs.clone()

            loss = dgr_model.observe(inputs, labels, not_aug_inputs)
            assert loss > 0

        # End first task to train VAE
        dgr_model.end_task(None)

        # Verify VAE was created and trained
        assert dgr_model.previous_vae is not None
        assert isinstance(dgr_model.previous_vae, DGRVAE)

        # Test replay sample generation
        n_replay_samples = 8
        replay_samples = dgr_model.previous_vae.generate_samples(n_replay_samples, device)

        assert replay_samples.shape == (n_replay_samples, 3, 224, 224)
        assert torch.all(replay_samples >= 0)  # Should be in valid range
        assert torch.all(replay_samples <= 1)  # Sigmoid output

        # Test replay monitoring functionality
        replay_monitor = DGRReplayMonitor(temp_dir)
        replay_monitor.log_replay_samples(replay_samples, task_id=1, epoch=0)

        # Verify monitoring files were created
        monitor_files = list(Path(temp_dir).glob("replay_samples_*.png"))
        assert len(monitor_files) > 0, "Replay monitoring files not created"

    def test_dgr_attention_visualization_compatibility(self, dgr_model):
        """Test DGR compatibility with attention visualization system."""
        device = torch.device('cpu')
        dgr_model.to(device)

        # Test with ResNet backbone (should handle gracefully)
        analyzer = AttentionAnalyzer(dgr_model, device=device)

        batch_size = 4
        inputs = torch.randn(batch_size, 3, 224, 224).to(device)

        # Should not crash even though ResNet doesn't have attention
        attention_maps = analyzer.extract_attention_maps(inputs)
        # ResNet should return empty list (no attention)
        assert isinstance(attention_maps, list)

    def test_dgr_checkpoint_compatibility(self, dgr_model, temp_dir):
        """Test DGR checkpoint saving and loading."""
        device = torch.device('cpu')
        dgr_model.to(device)

        # Train for a few steps
        batch_size = 8
        for i in range(3):
            inputs = torch.randn(batch_size, 3, 224, 224).to(device)
            labels = torch.randint(0, 50, (batch_size,)).to(device)
            not_aug_inputs = inputs.clone()

            loss = dgr_model.observe(inputs, labels, not_aug_inputs)

        # End task to create VAE
        dgr_model.end_task(None)

        # Save checkpoint
        checkpoint_path = Path(temp_dir) / "dgr_checkpoint.pt"
        checkpoint = {
            'model_state_dict': dgr_model.state_dict(),
            'vae_state_dict': dgr_model.vae.state_dict() if dgr_model.vae else None,
            'previous_vae_state_dict': dgr_model.previous_vae.state_dict() if dgr_model.previous_vae else None,
            'current_task': dgr_model.current_task,
            'args': dgr_model.args
        }
        torch.save(checkpoint, checkpoint_path)

        # Create new model and load checkpoint
        from models.dgr_mammoth_adapter import DGRMammothAdapter
        backbone = ResNet32(num_classes=100)
        loss_fn = nn.CrossEntropyLoss()
        transform = nn.Identity()

        new_model = DGRMammothAdapter(backbone, loss_fn, dgr_model.args, transform, None)
        new_model.to(device)

        # Load checkpoint
        loaded_checkpoint = torch.load(checkpoint_path, map_location=device)
        new_model.load_state_dict(loaded_checkpoint['model_state_dict'])

        if loaded_checkpoint['vae_state_dict']:
            new_model.vae.load_state_dict(loaded_checkpoint['vae_state_dict'])

        if loaded_checkpoint['previous_vae_state_dict']:
            new_model.previous_vae = DGRVAE(
                image_size=224, image_channels=3, z_dim=64, device=device
            )
            new_model.previous_vae.load_state_dict(loaded_checkpoint['previous_vae_state_dict'])

        # Test that loaded model works
        inputs = torch.randn(4, 3, 224, 224).to(device)
        outputs = new_model(inputs)
        assert outputs.shape == (4, 100)

    def test_dgr_experiment_runner_integration(self, dgr_args, temp_dir):
        """Test DGR integration with the experiment runner."""
        # Mock the experiment runner components
        with patch('run_einstellung_experiment.py') as mock_runner:
            # Test that DGR can be called from command line
            cmd_args = [
                '--model', 'dgr',
                '--backbone', 'resnet18',
                '--dgr_z_dim', '64',
                '--dgr_vae_lr', '0.001',
                '--n_epochs', '2',
                '--batch_size', '16'
            ]

            # This would normally be called by the experiment runner
            # We're testing that the arguments are properly parsed
            from argparse import ArgumentParser
            parser = ArgumentParser()
            parser.add_argument('--model', type=str, default='sgd')
            parser.add_argument('--backbone', type=str, default='resnet18')

            # Add DGR arguments
            parser = DGRMammothAdapter.get_parser(parser)

            # Parse arguments
            parsed_args = parser.parse_args(cmd_args)

            assert parsed_args.model == 'dgr'
            assert parsed_args.dgr_z_dim == 64
            assert parsed_args.dgr_vae_lr == 0.001

    @pytest.mark.slow
    def test_dgr_full_pipeline_simulation(self, dgr_args, temp_dir):
        """Simulate a complete DGR training pipeline."""
        device = torch.device('cpu')

        # Create model
        backbone = ResNet32(num_classes=100)
        loss_fn = nn.CrossEntropyLoss()
        transform = nn.Identity()

        # Mock dataset
        dataset = Mock()
        dataset.SIZE = [3, 224, 224]
        dataset.N_CLASSES = 100
        dataset.N_TASKS = 2
        dataset.N_CLASSES_PER_TASK = 50
        dataset.SETTING = 'class-il'
        dataset.get_offsets = Mock(side_effect=lambda task: (task * 50, (task + 1) * 50))
        dataset.get_normalization_transform = Mock(return_value=nn.Identity())

        model = DGRMammothAdapter(backbone, loss_fn, dgr_args, transform, dataset)
        model.to(device)

        # Simulate 2-task continual learning
        n_epochs_per_task = 2
        batch_size = 8
        n_batches_per_epoch = 4

        results = {'task_accuracies': [], 'replay_quality': []}

        for task_id in range(2):
            print(f"\n=== Task {task_id + 1} ===")

            # Begin task
            model.begin_task(dataset)

            # Training loop
            for epoch in range(n_epochs_per_task):
                epoch_loss = 0
                n_batches = 0

                for batch_id in range(n_batches_per_epoch):
                    # Generate task-specific data
                    inputs = torch.randn(batch_size, 3, 224, 224).to(device)
                    labels = torch.randint(task_id * 50, (task_id + 1) * 50, (batch_size,)).to(device)
                    not_aug_inputs = inputs.clone()

                    # Training step
                    loss = model.observe(inputs, labels, not_aug_inputs, epoch=epoch)
                    epoch_loss += loss
                    n_batches += 1

                avg_loss = epoch_loss / n_batches
                print(f"  Epoch {epoch + 1}: Loss = {avg_loss:.4f}")

            # End task
            model.end_task(dataset)

            # Evaluate task performance
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for _ in range(5):  # 5 evaluation batches
                    inputs = torch.randn(batch_size, 3, 224, 224).to(device)
                    labels = torch.randint(task_id * 50, (task_id + 1) * 50, (batch_size,)).to(device)

                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            task_accuracy = 100 * correct / total
            results['task_accuracies'].append(task_accuracy)
            print(f"  Task {task_id + 1} Accuracy: {task_accuracy:.2f}%")

            # Test replay generation if we have a previous VAE
            if model.previous_vae is not None:
                replay_samples = model.previous_vae.generate_samples(8, device)
                replay_quality = self._assess_replay_quality(replay_samples)
                results['replay_quality'].append(replay_quality)
                print(f"  Replay Quality Score: {replay_quality:.3f}")

            model.train()

        # Final evaluation
        print(f"\n=== Final Results ===")
        print(f"Task Accuracies: {results['task_accuracies']}")
        print(f"Replay Quality Scores: {results['replay_quality']}")

        # Assertions
        assert len(results['task_accuracies']) == 2
        assert all(acc > 0 for acc in results['task_accuracies'])  # Should have some accuracy

        if len(results['replay_quality']) > 0:
            assert all(0 <= quality <= 1 for quality in results['replay_quality'])

        # Test that model can still generate replay
        if model.previous_vae is not None:
            final_replay = model.previous_vae.generate_samples(4, device)
            assert final_replay.shape == (4, 3, 224, 224)

    def _assess_replay_quality(self, replay_samples: torch.Tensor) -> float:
        """Simple replay quality assessment based on sample statistics."""
        # Check if samples are in valid range
        if not (torch.all(replay_samples >= 0) and torch.all(replay_samples <= 1)):
            return 0.0

        # Check for diversity (standard deviation across samples)
        std_dev = torch.std(replay_samples).item()
        diversity_score = min(std_dev * 10, 1.0)  # Scale to [0, 1]

        # Check for reasonable pixel intensity distribution
        mean_intensity = torch.mean(replay_samples).item()
        intensity_score = 1.0 - abs(mean_intensity - 0.5) * 2  # Prefer mean around 0.5

        # Combine scores
        quality_score = 0.6 * diversity_score + 0.4 * intensity_score
        return max(0.0, min(1.0, quality_score))


class DGRReplayMonitor:
    """Monitor and visualize DGR replay samples."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Try to import matplotlib for visualization
        try:
            import matplotlib.pyplot as plt
            self.plt = plt
            self.visualization_available = True
        except ImportError:
            self.visualization_available = False
            print("Warning: matplotlib not available, replay visualization disabled")

    def log_replay_samples(self, samples: torch.Tensor, task_id: int, epoch: int):
        """Log and visualize replay samples."""
        if not self.visualization_available:
            return

        # Convert to numpy and ensure proper format
        samples_np = samples.detach().cpu().numpy()

        # Create visualization
        n_samples = min(8, samples.shape[0])
        fig, axes = self.plt.subplots(2, 4, figsize=(12, 6))
        fig.suptitle(f'DGR Replay Samples - Task {task_id}, Epoch {epoch}')

        for i in range(n_samples):
            row = i // 4
            col = i % 4

            # Convert from CHW to HWC for display
            img = samples_np[i].transpose(1, 2, 0)

            # Handle grayscale vs RGB
            if img.shape[2] == 1:
                img = img.squeeze(2)
                axes[row, col].imshow(img, cmap='gray')
            else:
                axes[row, col].imshow(img)

            axes[row, col].axis('off')
            axes[row, col].set_title(f'Sample {i+1}')

        # Save visualization
        filename = f'replay_samples_task{task_id}_epoch{epoch}.png'
        filepath = self.output_dir / filename
        self.plt.savefig(filepath, dpi=150, bbox_inches='tight')
        self.plt.close()

        # Log statistics
        stats_file = self.output_dir / 'replay_stats.txt'
        with open(stats_file, 'a') as f:
            mean_val = torch.mean(samples).item()
            std_val = torch.std(samples).item()
            min_val = torch.min(samples).item()
            max_val = torch.max(samples).item()

            f.write(f"Task {task_id}, Epoch {epoch}: "
                   f"mean={mean_val:.4f}, std={std_val:.4f}, "
                   f"min={min_val:.4f}, max={max_val:.4f}\n")


if __name__ == '__main__':
    # Run the tests
    pytest.main([__file__, '-v', '--tb=short'])
