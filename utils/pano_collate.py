import torch

def pano_collate_fn(batch, pad_token=""):
    """
    Custom collate function that:
      - Stacks images into a tensor [B, C, H, W].
      - Pads 'tags' and 'negative_tags' to the same length within a batch.
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

    # ---------------------------
    # 1. Pad the 'tags' field
    # ---------------------------
    max_tags_len = max(len(t_list) for t_list in tags) if len(tags) > 0 else 0
    padded_tags = []
    for t_list in tags:
        pad_count = max_tags_len - len(t_list)
        # Optionally truncate if t_list is *longer* than max_tags_len
        # if len(t_list) > max_tags_len: t_list = t_list[:max_tags_len]
        padded_tags.append(t_list + [pad_token] * pad_count)

    # ---------------------------
    # 2. Pad the 'negative_tags' field
    # ---------------------------
    max_neg_tags_len = max(len(nt_list) for nt_list in neg_tags) if len(neg_tags) > 0 else 0
    padded_neg_tags = []
    for nt_list in neg_tags:
        pad_count = max_neg_tags_len - len(nt_list)
        padded_neg_tags.append(nt_list + [pad_token] * pad_count)

    # Re-package everything into a batch dictionary
    batch_dict = {
        'image': images,                      # torch.Tensor, shape [B, C, H, W]
        'prompt': prompts,                    # list of strings (unchanged)
        'mood': moods,                        # list of strings
        'tags': padded_tags,                  # list of lists, now all same length
        'negative_tags': padded_neg_tags,     # list of lists, now all same length
        'lighting': lightings                 # list of strings
    }

    return batch_dict
