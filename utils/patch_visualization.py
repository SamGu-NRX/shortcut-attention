def visualize_patch_grid(image, patch_size=16, save_path=None):
    """
    Visualizes the patch grid overlaid on an image to show patch indices.
    
    Args:
        image: Input image tensor [C, H, W]
        patch_size: Size of patches
        save_path: Where to save the visualization
    """
    # Convert tensor to numpy for visualization
    img_np = image.permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2615])
    img_np = np.clip(img_np * std + mean, 0, 1)
    
    h, w = img_np.shape[0], img_np.shape[1]
    num_patches_h, num_patches_w = h // patch_size, w // patch_size
    
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img_np)
    
    # Draw grid lines
    for i in range(num_patches_h + 1):
        ax.axhline(i * patch_size, color='white', linewidth=0.5)
    for j in range(num_patches_w + 1):
        ax.axvline(j * patch_size, color='white', linewidth=0.5)
    
    # Add patch indices
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            patch_idx = i * num_patches_w + j
            ax.text(j * patch_size + patch_size//2, i * patch_size + patch_size//2, 
                    str(patch_idx), color='white', ha='center', va='center',
                    fontsize=8, fontweight='bold')
    
    ax.set_title("Image with Patch Grid and Indices")
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()