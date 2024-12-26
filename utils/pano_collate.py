import torch

def pano_collate_fn(batch):
    """
    Custom collate function for PanimeDataset that merges 
    all text fields into `pano_prompt`.
    """
    images = []
    pano_prompts = []

    for item in batch:
        # Convert 'image' from NumPy to a torch.Tensor
        images.append(torch.from_numpy(item['image']))
        
        # The single merged textual prompt for this sample
        pano_prompts.append(item['pano_prompt'])

    # Stack only images (which share the same shape) into [B, C, H, W]
    images = torch.stack(images, dim=0)

    return {
        'image': images,
        'pano_prompt': pano_prompts
    }
