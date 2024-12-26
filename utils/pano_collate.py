import torch

def pano_collate_fn(batch):
    """
    Custom collate function for panorama datasets that have variable-length fields.
    """
    # Separate out each field in the batch
    images = [torch.from_numpy(item['image']) for item in batch]
    prompts = [item['prompt'] for item in batch]
    moods = [item['mood'] for item in batch]
    tags = [item['tags'] for item in batch]
    neg_tags = [item['negative_tags'] for item in batch]
    lightings = [item['lighting'] for item in batch]
    
    # Stack only the images (which all have the same shape)
    images = torch.stack(images, dim=0)

    # Re-package everything into a batch dictionary
    batch_dict = {
        'image': images,           # [B, C, H, W]
        'prompt': prompts,         # list of strings
        'mood': moods,             # list of strings
        'tags': tags,              # list of lists of strings
        'negative_tags': neg_tags, # list of lists of strings
        'lighting': lightings      # list of strings
    }
    return batch_dict
