import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
import deepspeed
from transformers.activations import QuickGELUActivation

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)

class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        # self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

class LoRALinear(nn.Module):
    def __init__(self, original_layer, alpha=1, rank=4, drop_rate=0.):
        super().__init__()
        in_features = original_layer.in_features
        out_features = original_layer.out_features

        self.lora_dropout = nn.Dropout(drop_rate)
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        self.scaling = alpha / rank

        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)

        self.original_layer = original_layer
        self.original_layer.requires_grad_(False)

    def forward(self, x):
        out = self.original_layer(x)
        lora_out = self.lora_up(self.lora_down(self.lora_dropout(x))) * self.scaling
        return out + lora_out


class CLIPVisionTowerLoRA(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        self.args = args
        super().__init__(vision_tower, args, delay_load=delay_load)

    def load_model(self, device_map=None):
        super().load_model(device_map)
        self._apply_lora_to_vision_tower()

    def _apply_lora_to_vision_tower(self):
        self.lora_rank = getattr(self.args, 'vfm_lora_rank', 4)
        self.lora_alpha = getattr(self.args, 'vfm_lora_alpha', 16)
        self.lora_dropout = getattr(self.args, 'vfm_lora_dropout', 0.0)
        print(self.lora_dropout)
        # apply LoRA to attention q/k/v/proj in all transformer blocks
        for i, layer in enumerate(self.vision_tower.vision_model.encoder.layers):
            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
                proj = getattr(layer.self_attn, proj_name)
                setattr(layer.self_attn, proj_name, LoRALinear(
                    proj, alpha=self.lora_alpha, rank=self.lora_rank, drop_rate=self.lora_dropout
                ))

        print(f"[INFO] Applied LoRA to vision tower with rank={self.lora_rank}, alpha={self.lora_alpha}")

class CLIPVisionTowerVPT(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        self.args = args
        self.vpt_num_prompts = getattr(args, 'vpt_num_prompts', 10)
        self.vpt_prompt_dim = None  # to be initialized after loading vision model
        super().__init__(vision_tower, args, delay_load=delay_load)

    def load_model(self, device_map=None):
        super().load_model(device_map)
        self._apply_vpt_to_vision_tower()

    def _apply_vpt_to_vision_tower(self):
        embed_dim = self.vision_tower.config.hidden_size
        self.vpt_prompt_dim = embed_dim

        self.prompt_embeddings = nn.ParameterList([
            nn.Parameter(torch.randn(1, self.vpt_num_prompts, embed_dim))
            for _ in range(len(self.vision_tower.vision_model.encoder.layers))
        ])

        print(f"[INFO] Applied VPT-deep with {self.vpt_num_prompts} prompts per layer, dim={embed_dim}")

        for i, layer in enumerate(self.vision_tower.vision_model.encoder.layers):
            original_forward = layer.forward

            def make_vpt_forward(i, original_forward):
                def vpt_forward(layer_self, hidden_states, *args, **kwargs):
                    B = hidden_states.size(0)
                    prompts = self.prompt_embeddings[i].expand(B, -1, -1)  # [B, P, D]
                    hidden_states = torch.cat([hidden_states[:, :1], prompts, hidden_states[:, 1:]], dim=1)  
                    outputs = original_forward(hidden_states, *args, **kwargs)
                    hidden_states = outputs[0]  # 

                    # remove prompt tokens from output
                    hidden_states = hidden_states[:, [0] + list(range(self.vpt_num_prompts + 1, hidden_states.size(1)))]
                    
                    if len(outputs) > 1:
                        return (hidden_states,) + outputs[1:]
                    else:
                        return (hidden_states,)
                return vpt_forward.__get__(layer, layer.__class__)

            layer.forward = make_vpt_forward(i, original_forward)

    def forward(self, images):
        if isinstance(images, list):
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features


class CLIPVisionTowerContextMoe(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        self.args = args
        self.context_embedding_dim = getattr(args, 'context_embedding_dim', 768)
        self.num_experts = getattr(args, 'moe_num_experts', 4)
        self.moe_start_layer = getattr(args, 'moe_start_layer', 22)
        super().__init__(vision_tower, args, delay_load=delay_load)

    def load_model(self, device_map=None):
        super().load_model(device_map)
        self._apply_moe_to_vision_tower()

    def _apply_moe_to_vision_tower(self):

        for i, layer in enumerate(self.vision_tower.vision_model.encoder.layers):
            rank0_print('current layer:', i)
            num_layers = len(self.vision_tower.vision_model.encoder.layers)
            if i >= self.moe_start_layer and i < num_layers - 1:
            #if i in [num_layers - 2, num_layers - 3]:
                rank0_print('applying MoE to layer', i)
                expert_ffns = nn.ModuleList()
                for _ in range(self.num_experts):
                    expert = nn.Sequential(
                        nn.Linear(layer.mlp.fc1.in_features, layer.mlp.fc1.out_features),
                        QuickGELUActivation(),
                        #nn.GELU(),
                        nn.Linear(layer.mlp.fc2.in_features, layer.mlp.fc2.out_features)
                    )
                    with torch.no_grad():
                        with deepspeed.zero.GatheredParameters(
                            [layer.mlp.fc1.weight, layer.mlp.fc1.bias,
                             layer.mlp.fc2.weight, layer.mlp.fc2.bias],
                            modifier_rank=0
                        ):
                            expert[0].weight.copy_(layer.mlp.fc1.weight.detach().clone())
                            expert[0].bias.copy_(layer.mlp.fc1.bias.detach().clone())
                            expert[2].weight.copy_(layer.mlp.fc2.weight.detach().clone())
                            expert[2].bias.copy_(layer.mlp.fc2.bias.detach().clone())
                    expert_ffns.append(expert)

                moe_layer = ContextualMoE(
                    expert_ffns,
                    context_dim=self.context_embedding_dim,
                    token_dim=layer.mlp.fc1.in_features,
                ).to(self.device)
                layer.mlp = moe_layer

                # Wrap original forward to inject context_embeddings
                orig_forward = layer.forward

                def make_forward(orig_forward):
                    def new_forward(self_layer, hidden_states, *args, **kwargs):
                        # Extract context_embeddings if available
                        context_embeddings = kwargs.pop("context_embeddings", None)

                        # forward pass through attention
                        residual = hidden_states
                        hidden_states = self_layer.self_attn(self_layer.layer_norm1(hidden_states), **kwargs)[0]
                        hidden_states = residual + hidden_states

                        # forward pass through MoE FFN with context
                        residual = hidden_states
                        mlp_output = self_layer.mlp(self_layer.layer_norm2(hidden_states), context_embeddings=context_embeddings)
                        mlp_out_feature = mlp_output[0]
                        gate_probs = mlp_output[1]  # [B, num_experts]
                        hidden_states = residual + mlp_out_feature

                        return (hidden_states, gate_probs)

                    return new_forward.__get__(layer, layer.__class__)

                layer.forward = make_forward(orig_forward)

        rank0_print(f"[INFO] Applied MoE to vision tower with {self.num_experts} FFN experts per block.")

    def forward(self, images, context_embeddings):
        pixel_values = images.to(device=self.device, dtype=self.dtype)

        # Forward through embeddings
        encoder = self.vision_tower.vision_model
        hidden_states = encoder.embeddings(pixel_values)
        hidden_states = encoder.pre_layrnorm(hidden_states)

        all_hidden_states = []
        all_gate_probs = []

        # Forward each encoder block manually to pass context_embeddings
        for i, layer in enumerate(encoder.encoder.layers):
            num_layers = len(encoder.encoder.layers)
            if i >= self.moe_start_layer and i < num_layers - 1:
            #if i in [num_layers - 2, num_layers - 3]:
                layer_outputs = layer(
                    hidden_states,
                    context_embeddings=context_embeddings  # 🔁 context passed manually
                )
                # print(type(layer_outputs))
                # print(len(layer_outputs))
                hidden_states = layer_outputs[0]
                gate_probs = layer_outputs[1]  # [B, num_experts]
                all_hidden_states.append(hidden_states)
                all_gate_probs.append(gate_probs)
            else:
                layer_outputs = layer(hidden_states,
                                      attention_mask=None,
                                      causal_attention_mask=None)
                hidden_states = layer_outputs[0]
                all_hidden_states.append(hidden_states)

        # Create the final output object similar to CLIPVisionModel output
        class Output:
            def __init__(self, hidden_states):
                self.hidden_states = hidden_states

        return self.feature_select(Output(all_hidden_states)).to(images.dtype), all_gate_probs

class ResidualBlock(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 2
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = QuickGELUActivation()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.to(self.fc1.weight.dtype)
        residual = x
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        x = self.ln(x + residual)
        return x


class ContextualMoE(nn.Module):
    def __init__(self, experts, context_dim, token_dim=None, attn_heads=4):
        """
        experts: list of expert modules, each maps [B, N, D] -> [B, N, D]
        context_dim: 文本特征维度
        token_dim: 图像 token 特征维度
        """
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.num_experts = len(self.experts)

        assert token_dim is not None, "必须指定 token_dim (hidden_states.shape[-1])"
        self.token_dim = token_dim
        self.context_dim = context_dim

        # 残差增强模块
        self.image_resblock = ResidualBlock(token_dim)
        self.text_resblock = ResidualBlock(context_dim)

        # cross-attention：文本为 query，拼接后为 key/value
        self.cross_attn = nn.MultiheadAttention(embed_dim=context_dim,
                                                kdim=token_dim + context_dim,
                                                vdim=token_dim + context_dim,
                                                num_heads=attn_heads,
                                                batch_first=True)
        
        # gating 网络（输入为任务向量）
        self.gate = nn.Linear(context_dim, self.num_experts)

    def forward(self, hidden_states, *args, context_embeddings=None, **kwargs):
        """
        hidden_states: [B, N, D] 图像特征
        context_embeddings: [B, C] 文本特征
        """
        B, N, D = hidden_states.shape
        #C = context_embeddings.shape[-1]
        #device = hidden_states.device

        # === 1. 残差增强 ===
        img_feat = self.image_resblock(hidden_states)      # [B, N, D]
        txt_feat = self.text_resblock(context_embeddings)  # [B, C]

        # # === 2. 归一化 ===
        # img_feat = F.normalize(img_feat, dim=-1)
        # txt_feat = F.normalize(txt_feat, dim=-1)

        # === 3. 文本特征扩展并拼接 ===
        txt_expand = txt_feat.unsqueeze(1).expand(-1, N, -1)        # [B, N, C]
        fused = torch.cat([img_feat, txt_expand], dim=-1)           # [B, N, D+C]

        # === 4. Cross-Attention：文本特征为 Query，全图拼接特征为 Key/Value ===
        q = txt_feat.unsqueeze(1)                                   # [B, 1, C]

        attn_out, _ = self.cross_attn(q, fused, fused)            # [B, 1, C]
        task_vec = attn_out.squeeze(1)                              # [B, C]

        # === 5. 路由 gating（全图共享同一个）===
        gate_logits = self.gate(task_vec)                           # [B, E]
        gate_probs = F.softmax(gate_logits, dim=-1)                 # [B, E]
        gate_probs_expand = gate_probs.unsqueeze(1).expand(-1, N, -1)  # [B, N, E]

        # === 6. 专家计算 ===
        expert_outputs = []
        for expert in self.experts:
            out = expert(hidden_states)  # [B, N, D]
            expert_outputs.append(out.unsqueeze(2))  # [B, N, 1, D]

        expert_outputs = torch.cat(expert_outputs, dim=2)           # [B, N, E, D]

        # === 7. 加权融合 ===
        outputs = torch.sum(gate_probs_expand.unsqueeze(-1) * expert_outputs, dim=2)  # [B, N, D]

        return outputs, gate_logits
