import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerLoRA, CLIPVisionTowerVPT, CLIPVisionTowerContextMoe
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

def build_vision_tower(vision_tower_cfg, anchor_tower=False, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    vfm_tuning_type = getattr(vision_tower_cfg, 'vfm_tuning_type', False)

    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if anchor_tower:
            print('building anchor vision tower')
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        if vfm_tuning_type == 'lora':
            return CLIPVisionTowerLoRA(vision_tower, args=vision_tower_cfg, **kwargs)
        if vfm_tuning_type == 'vpt':
            return CLIPVisionTowerVPT(vision_tower, args=vision_tower_cfg, **kwargs)
        if 'context_moe' in vfm_tuning_type:
            return CLIPVisionTowerContextMoe(vision_tower, args=vision_tower_cfg, **kwargs)

        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')

def build_context_embedding_model(config):
    context_embedding_model = getattr(config, 'context_embedding_model', None)
    if context_embedding_model is None:
        raise ValueError("context_embedding_model must be specified in the config.")
    
    context_embedding_tokenizer = AutoTokenizer.from_pretrained(context_embedding_model)
    context_embedding_model = AutoModel.from_pretrained(context_embedding_model)
    context_embedding_model.eval()

    return context_embedding_tokenizer, context_embedding_model


