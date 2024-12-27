import torch

def pano_collate_fn(batch):
    """
    Custom collate function for PanimeDataset that merges 
    all text fields into `pano_prompt`.
    """
    images = []
    pano_prompts = []
    panos = [] 

    for item in batch:
        # Convert 'image' from NumPy to a torch.Tensor
        images.append(torch.from_numpy(item['image']).unsqueeze(1))
        panos.append(torch.from_numpy(item['pano']))
        
        # The single merged textual prompt for this sample
        pano_prompts.append(item['pano_prompt'])

    # Stack only images (which share the same shape) into [B, C, H, W]
    images = torch.stack(images, dim=0)
    panos = torch.stack(panos, dim=0)

    return {
        'images': images,
        'pano': panos,
        'pano_prompt': pano_prompts
    }
