"""Extensions for attention visualization support in continual learning models."""

import os
import torch
from utils.attention_visualization import AttentionAnalyzer, visualize_attention_map

class ContinualAttentionAnalyzer(AttentionAnalyzer):
    """Adapter for analyzing attention in continual learning models."""
    
    def extract_attention_maps(self, inputs: torch.Tensor) -> dict:
        """Extract attention maps using backbone's native attention extraction.
        
        Args:
            inputs: Input tensor of shape [B, C, H, W]
            
        Returns:
            Dictionary with attention maps from each transformer block
        """
        with torch.no_grad():
            self.model.eval()
            # Forward pass with attention extraction
            self.model.net(inputs.to(self.device), returnt='attention')
            
            # Get attention maps from backbone
            attn_maps = self.model.get_attention_maps()
            
            # Format attention maps into expected dictionary structure
            if attn_maps is not None:
                return {f'block_{i}': maps.cpu() for i, maps in enumerate(attn_maps)}
            return {}

def visualize_class_attention(model, dataset, save_dir: str, num_samples: int = 5):
    """Visualize attention maps for specific classes.
    
    Args:
        model: The continual learning model
        dataset: The dataset containing class samples
        save_dir: Directory to save visualizations
        num_samples: Number of samples per class to visualize
    """
    analyzer = ContinualAttentionAnalyzer(model, device=model.device)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Get samples from each class
    for task_id in range(dataset.N_TASKS):
        train_loader, _ = dataset.get_data_loaders()
        class_samples = {cls: [] for cls in dataset.get_task_class_names(task_id)}
        
        # Collect samples
        for images, labels, _ in train_loader:
            for img, lbl in zip(images, labels):
                cls_name = dataset.class_names[lbl]
                if cls_name in class_samples and len(class_samples[cls_name]) < num_samples:
                    class_samples[cls_name].append(img)
                    
            # Check if we have enough samples
            if all(len(samples) >= num_samples for samples in class_samples.values()):
                break
                
        # Analyze attention for each class
        for cls_name, samples in class_samples.items():
            samples_batch = torch.stack(samples)
            attn_maps = analyzer.extract_attention_maps(samples_batch)
            
            if attn_maps:  # Only visualize if we got attention maps
                for i, (layer_name, attn_map) in enumerate(attn_maps.items()):
                    for sample_idx in range(len(samples)):
                        save_path = f"{save_dir}/task{task_id}_{cls_name}_{sample_idx}_{layer_name}.png"
                        visualize_attention_map(
                            attention_map=attn_map[sample_idx:sample_idx+1],
                            input_image=samples_batch[sample_idx:sample_idx+1],
                            layer_name=f"{layer_name} - {cls_name}",
                            save_path=save_path
                        )
