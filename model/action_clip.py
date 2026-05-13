import wandb
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPModel, Dinov2Model, AutoModel, Siglip2TextModel, Siglip2VisionModel
from model.processor import ContrastiveProcessorWrapper, VisionActionGroundedContrastiveProcessorWrapper

from utils import printable_params, check_tensor, DotDict

@printable_params
class ActionCLIP(nn.Module):
    def __init__(self, cfg, device, dtype, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        attn_implementation = "flash_attention_2" if self.dtype in [torch.float16, torch.bfloat16] else None
        if cfg.model.clip.get("use_siglip2", False):
            self.clip = AutoModel.from_pretrained("google/siglip2-base-patch16-224",
                                                  attn_implementation=attn_implementation,
                                                  device_map=device,
                                                  torch_dtype=self.dtype)
        else:
            self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32",
                                                  attn_implementation=None,
                                                  device_map=device,
                                                  torch_dtype=self.dtype)

        self.processor = ContrastiveProcessorWrapper(
            cfg=self.cfg.model.processor,
            horizon=self.cfg.environment.kwargs.horizon,
            action_dimension=self.cfg.environment.action_dimension,
            target_vla=self.cfg.model.clip.get("target_vla", "openvla"),
            dtype=self.dtype,
            device=device,
            custom_prompt_template=self.cfg.model.clip.get("custom_prompt_template", True),
            dino_version=self.cfg.model.clip.get("dino_version", "v2"),
            vision_backbone=self.cfg.model.clip.get("vision_backbone", "dino"),
            text_backbone=self.cfg.model.clip.get("text_backbone", "clip")
        )

    @torch.no_grad()
    def process_batch(self, batch):
        return self.processor.process_batch(batch)

    def forward(self, input_ids, attention_mask=None, pixel_values=None, **kwargs):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        return self.clip(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, return_loss=True, **kwargs)

    @torch.no_grad()
    def score(self, trajectory, text, frames, instruction):
        batch = {
            "action": trajectory,
            "instruction": [instruction],
            "frames": frames,
            "captions": [text]
        }
        model_inputs = self.process_batch(batch)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask", None)
        pixel_values = model_inputs["pixel_values"]
        output = self.clip(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, return_loss=True)
        return output.logits_per_text


class ActionEncoder(nn.Module):
    def __init__(self, cfg, device, dtype, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.repeat_actions = cfg.model.get("repeat_actions", 1)
        self.action_dim = cfg.environment.action_dimension * self.repeat_actions
        self.horizon = cfg.environment.kwargs.get("horizon", 1)

        # Improved action encoder with intermediate layer
        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_dim, self.cfg.model.clip.dim // 2, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(self.cfg.model.clip.dim // 2, self.cfg.model.clip.dim, device=device, dtype=dtype)
        )

        # position embedding
        self.position_embedding_type = self.cfg.model.clip.get("position_embedding", "learned")
        if self.position_embedding_type == "learned":
            self.position_embedding = nn.Parameter(torch.zeros(1, self.horizon, self.cfg.model.clip.dim, device=device, dtype=dtype))

        self.transformer = nn.TransformerEncoderLayer(
            self.cfg.model.clip.dim,
            nhead=4,
            dropout=0.1,
            dim_feedforward=self.cfg.model.clip.dim*4,
            device=device,
            dtype=dtype,
            batch_first=True
        )

    def forward(self, actions):
        actions = actions.to(dtype=self.dtype, device=self.device)
        actions = actions.squeeze(1)
        actions = rearrange(actions, 'b (h a) -> b h a', h=self.horizon)
        # (batch_size, horizon, action_dim)

        if self.repeat_actions > 1:
            actions = actions.repeat_interleave(self.repeat_actions, dim=1)
        # action: (batch_size, seq_len, action_dim)
        # action_embedding: (batch_size, seq_len, embed_dim)
        action_embedding = self.action_encoder(actions)
        if self.position_embedding_type == "learned":
            action_embedding = action_embedding + self.position_embedding[:, :action_embedding.size(1), :]
        # transformer: (batch_size, seq_len, embed_dim)
        transformer_output = self.transformer(action_embedding) # already normalized inside
        return transformer_output

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1, ffn_expansion=4, device=None, dtype=None):
        super().__init__()

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            dim, num_heads, batch_first=True, dropout=dropout,
            device=device, dtype=dtype
        )

        # Feed-forward network
        self.ffn = FeedForwardNetwork(
            dim, hidden_dim=dim * ffn_expansion, dropout=dropout,
            device=device, dtype=dtype
        )

        # Norm layers (applied before each sublayer)
        self.query_norm = nn.LayerNorm(dim, device=device, dtype=dtype)
        self.key_norm   = nn.LayerNorm(dim, device=device, dtype=dtype)
        self.value_norm = nn.LayerNorm(dim, device=device, dtype=dtype)
        self.ffn_norm   = nn.LayerNorm(dim, device=device, dtype=dtype)

        # Dropouts
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_dropout  = nn.Dropout(dropout)

        # Residual Scales
        self.attn_scale = 0.5
        self.ffn_scale = 0.8

    def forward(self, query, key, value, attn_mask=None):
        # === Pre-Norm Attention ===
        q = self.query_norm(query)
        k = self.key_norm(key)
        v = self.value_norm(value)

        attn_out, attn_weights = self.attention(q, k, v, attn_mask=attn_mask)
        query = query + self.attn_scale * self.attn_dropout(attn_out)  # residual connection

        # === Pre-Norm Feed-Forward ===
        ffn_in = self.ffn_norm(query)
        ffn_out = self.ffn(ffn_in)
        output = query + self.ffn_scale * self.ffn_dropout(ffn_out)  # residual connection

        return output

class FiLMBlock(nn.Module):
    """Feature-wise Linear Modulation block for conditioning vision features with action embeddings."""

    def __init__(self, dim, hidden_dim=None, dropout=0.1, device=None, dtype=None):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = dim * 2

        # Generate gamma (scale) and beta (shift) parameters from conditioning input
        self.film_generator = nn.Sequential(
            nn.Linear(dim, hidden_dim, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim * 2, device=device, dtype=dtype)  # outputs both gamma and beta
        )

        # Feed-forward network applied after FiLM
        self.ffn = FeedForwardNetwork(
            dim, hidden_dim=dim * 4, dropout=dropout,
            device=device, dtype=dtype
        )

        # Norm layers
        self.feature_norm = nn.LayerNorm(dim, device=device, dtype=dtype)
        self.ffn_norm = nn.LayerNorm(dim, device=device, dtype=dtype)

        # Dropouts
        self.film_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)

        # Residual scales
        self.film_scale = 0.5
        self.ffn_scale = 0.8

    def forward(self, features, conditioning):
        """
        Args:
            features: [batch, seq_len, dim] - features to be modulated (e.g., vision)
            conditioning: [batch, dim] or [batch, seq_len_cond, dim] - conditioning input (e.g., action)
                         If 2D, will be broadcast across sequence dimension
                         If 3D, can use mean pooling or last token

        Returns:
            modulated_features: [batch, seq_len, dim]
        """
        # Handle conditioning input
        if conditioning.dim() == 3:
            # If conditioning is a sequence, pool it (mean pooling)
            conditioning = conditioning.mean(dim=1)  # [batch, dim]

        # === FiLM Modulation ===
        # Normalize features
        normed_features = self.feature_norm(features)

        # Generate gamma (scale) and beta (shift)
        film_params = self.film_generator(conditioning)  # [batch, dim * 2]
        gamma, beta = film_params.chunk(2, dim=-1)  # each [batch, dim]

        # QUICK FIX: Offset gamma to center at 1.0
        gamma = gamma + 1.0  # Shifts distribution from ~0 to ~1

        # Apply FiLM: gamma * x + beta (broadcast across sequence dimension)
        gamma = gamma.unsqueeze(1)  # [batch, 1, dim]
        beta = beta.unsqueeze(1)    # [batch, 1, dim]

        with torch.no_grad():
            wandb.log({
                "film/beta_mean": beta.mean().item(),
                "film/gamma_mean": gamma.mean().item(),
            })

        film_out = gamma * normed_features + beta
        features = features + self.film_scale * self.film_dropout(film_out)

        # === Feed-Forward Network ===
        ffn_in = self.ffn_norm(features)
        ffn_out = self.ffn(ffn_in)
        output = features + self.ffn_scale * self.ffn_dropout(ffn_out)

        return output

class FeedForwardNetwork(nn.Module):
    """Standard transformer feed-forward network with GELU activation"""

    def __init__(self, input_dim, hidden_dim, dropout=0.1, device=None, dtype=None):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim, device=device, dtype=dtype)
        self.linear2 = nn.Linear(hidden_dim, input_dim, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()

    def forward(self, x):
        # x -> linear1 -> GELU -> dropout -> linear2
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class SimpleProjection(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1, bias=True, device=None, dtype=None):
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim, device=device, dtype=dtype),  # Pre-norm
            nn.Linear(input_dim, output_dim * 2, bias=bias, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim, bias=bias, device=device, dtype=dtype),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.projection:
            if isinstance(module, nn.Linear):
                # Xavier initialization with smaller gain
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.projection(x)

class ResidualProjection(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1, bias=True,
                 num_layers=2, device=None, dtype=None):
        super().__init__()
        hidden_dim = max(input_dim, output_dim) * 2  # Wider bottleneck

        layers = [nn.LayerNorm(input_dim, device=device, dtype=dtype)]

        # First layer
        layers.extend([
            nn.Linear(input_dim, hidden_dim, bias=bias, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Dropout(dropout)
        ])

        # Additional hidden layers
        for _ in range(num_layers - 1):
            layers.extend([
                nn.LayerNorm(hidden_dim, device=device, dtype=dtype),
                nn.Linear(hidden_dim, hidden_dim, bias=bias, device=device, dtype=dtype),
                nn.SiLU(),
                nn.Dropout(dropout)
            ])

        # Output projection (with pre-norm)
        layers.extend([
            nn.LayerNorm(hidden_dim, device=device, dtype=dtype),
            nn.Linear(hidden_dim, output_dim, bias=bias, device=device, dtype=dtype)
        ])

        self.projection = nn.Sequential(*layers)
        self.residual_scale = nn.Parameter(torch.ones(1, dtype=dtype, device=device) * (1.0 / torch.sqrt(torch.tensor(num_layers, dtype=dtype, device=device))), requires_grad=True)
        self.skip_connection = nn.Linear(input_dim, output_dim, bias=bias,
                                        device=device, dtype=dtype) if input_dim != output_dim else nn.Identity()
        self._init_weights()

    def _init_weights(self):
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        if isinstance(self.skip_connection, nn.Linear):
            nn.init.xavier_uniform_(self.skip_connection.weight, gain=1.0)
            if self.skip_connection.bias is not None:
                nn.init.constant_(self.skip_connection.bias, 0)

    def forward(self, x):
        return self.residual_scale * self.projection(x) + self.skip_connection(x)  # Scaled residual

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim, device, dtype):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1, device=device, dtype=dtype)

    def forward(self, sequence_output):
        # sequence_output: [batch_size, seq_len, hidden_dim]

        # Compute attention scores
        attention_scores = self.attention(sequence_output)  # [batch, seq_len, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch, seq_len, 1]

        # Weighted sum
        pooled_output = torch.sum(attention_weights * sequence_output, dim=1)  # [batch, hidden_dim]
        return pooled_output

class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sequence_output):
        return sequence_output.mean(dim=1)

class MaskedAttentionPooling(nn.Module):
    def __init__(self, hidden_dim, device, dtype):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1, device=device, dtype=dtype)

    def forward(self, sequence_output, attention_mask):
        # sequence_output: [batch_size, seq_len, hidden_dim]
        # attention_mask: [batch_size, seq_len] (1 for real tokens, 0 for padding)

        attention_scores = self.attention(sequence_output)  # [batch, seq_len, 1]

        # Mask out padding tokens
        mask = attention_mask.unsqueeze(-1)  # [batch, seq_len, 1]
        attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(attention_scores, dim=1)
        pooled_output = torch.sum(attention_weights * sequence_output, dim=1)

        return pooled_output

class MultiLayerCrossAttention(nn.Module):
    """Stacked bidirectional cross-attention between vision and action embeddings."""

    def __init__(self, embed_dim, num_layers=4, num_heads=8, dropout=0.1, device=None, dtype=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        ffn_expansion = 4

        # Stacked cross-attention layers
        self.vision_to_action_layers = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, dropout, ffn_expansion, device, dtype)
            for _ in range(num_layers)
        ])
        self.action_to_vision_layers = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, dropout, ffn_expansion, device, dtype)
            for _ in range(num_layers)
        ])

    def forward(self, vision_embedding, action_embedding):
        # Vision: [batch, embed_dim] → [batch, 1, embed_dim]
        vision_hidden = vision_embedding.unsqueeze(1)
        action_hidden = action_embedding  # [batch, seq_len, embed_dim]

        for layer_idx in range(self.num_layers):
            # Bidirectional cross-attention
            vision_hidden = self.vision_to_action_layers[layer_idx](
                query=vision_hidden, key=action_hidden, value=action_hidden
            )
            action_hidden = self.action_to_vision_layers[layer_idx](
                query=action_hidden, key=vision_hidden, value=vision_hidden
            )

        return vision_hidden, action_hidden

class MultiLayerFiLM(nn.Module):
    """Stacked FiLM layers for action-conditioning vision features."""

    def __init__(self, embed_dim, num_layers=4, dropout=0.1, device=None, dtype=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # Stacked FiLM layers
        self.film_layers = nn.ModuleList([
            FiLMBlock(embed_dim, hidden_dim=embed_dim * 2, dropout=dropout, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])

    def forward(self, vision_embedding, action_embedding):
        """
        Args:
            vision_embedding: [batch, embed_dim] or [batch, 1, embed_dim]
            action_embedding: [batch, seq_len, embed_dim]

        Returns:
            conditioned_vision: same shape as vision_embedding
        """
        # Ensure vision is 3D
        if vision_embedding.dim() == 2:
            vision_hidden = vision_embedding.unsqueeze(1)  # [batch, 1, embed_dim]
            squeeze_output = True
        else:
            vision_hidden = vision_embedding
            squeeze_output = False

        # Apply FiLM layers sequentially
        for layer in self.film_layers:
            vision_hidden = layer(vision_hidden, action_embedding)

        # Return in original shape
        if squeeze_output:
            return vision_hidden.squeeze(1)  # [batch, embed_dim]
        return vision_hidden

class AttentionBasedFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1, device=None, dtype=None):
        super().__init__()
        self.embed_dim = embed_dim

        # Query generator instead of fixed parameter
        self.query_generator = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim, device=device, dtype=dtype),  # vision + action -> query
            nn.LayerNorm(embed_dim, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim, device=device, dtype=dtype)
        )

        self.fusion_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
            device=device, dtype=dtype
        )

        self.output_proj = nn.Linear(embed_dim, embed_dim, device=device, dtype=dtype)

    def forward(self, vision_features, action_features):
        # Generate sample-specific fusion query
        input_concat = torch.cat([vision_features, action_features], dim=-1)
        fusion_query = self.query_generator(input_concat).unsqueeze(1)  # [batch, 1, embed_dim]

        # Stack modalities as key-value pairs
        modality_features = torch.stack([vision_features, action_features], dim=1)

        # Apply attention with sample-specific queries
        fused_output, attention_weights = self.fusion_attention(
            fusion_query,      # Sample-specific queries
            modality_features, # Keys
            modality_features  # Values
        )

        # Project
        fused_output = fused_output.squeeze(1)
        fused_output = self.output_proj(fused_output) # no norm here since handled by contrastive loss

        return fused_output, attention_weights.squeeze(1)

class GatedFusion(nn.Module):
    """
    Gated fusion with proper normalization for stable training.
    """
    def __init__(self, embed_dim, device=None, dtype=None, use_layernorm=True, vector_fusion=False):
        super().__init__()

        # Option 1: Input normalization (recommended)
        self.norm_vision = nn.LayerNorm(embed_dim, device=device, dtype=dtype) if use_layernorm else nn.Identity()
        self.norm_action = nn.LayerNorm(embed_dim, device=device, dtype=dtype) if use_layernorm else nn.Identity()
        self.vector_fusion = vector_fusion
        # Gate network
        if self.vector_fusion:
            self.gate = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim, device=device, dtype=dtype),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim, device=device, dtype=dtype),
                nn.Sigmoid()
            )
        else:
            self.gate = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim, device=device, dtype=dtype),
                nn.ReLU(),
                nn.Linear(embed_dim, 2, device=device, dtype=dtype),
                nn.Softmax(dim=-1)
        )

        # Output projection
        self.output_proj = ResidualProjection(
            input_dim=embed_dim,
            output_dim=embed_dim,
            dropout=0.1,
            device=device,
            dtype=dtype
        )

        # Option 2: Output normalization (also recommended)
        # self.norm_output = nn.LayerNorm(embed_dim, device=device, dtype=dtype) if use_layernorm else nn.Identity()

    def forward(self, vision_features, action_features):
        # Normalize inputs before gating
        vision_norm = self.norm_vision(vision_features)
        action_norm = self.norm_action(action_features)

        # Compute gates on normalized features
        gates = self.gate(torch.cat([vision_norm, action_norm], dim=-1))  # [batch, 2]

        # Weighted sum (using normalized features)
        if self.vector_fusion:
            fused = gates * vision_norm + (1 - gates) * action_norm
        else:
            fused = gates[:, 0:1] * vision_norm + gates[:, 1:2] * action_norm

        # Project and normalize output
        fused = self.output_proj(fused)
        # fused = self.norm_output(fused)

        return fused, gates

class MLPFusion(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, dropout=0.1, device=None, dtype=None):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim * 2

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim, device=device, dtype=dtype)
        )

    def forward(self, vision_features, action_features):
        fused = self.mlp(torch.cat([vision_features, action_features], dim=-1))
        return fused, None  # no attention weights

class HadamardFusion(nn.Module):
    def __init__(self, embed_dim, device=None, dtype=None):
        super().__init__()
        self.output_proj = nn.Linear(embed_dim, embed_dim, device=device, dtype=dtype)

    def forward(self, vision_features, action_features):
        fused = vision_features * action_features  # element-wise product
        fused = self.output_proj(fused + vision_features + action_features)
        return fused, None

class NaiveFusion(nn.Module):
    def forward(self, vision_features, action_features):
        fused = 0.5 * vision_features + 0.5 * action_features
        return fused, None

@printable_params
class VisionActionGroundedCLIP(nn.Module):
    def __init__(self, cfg, device, dtype, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.device = device
        self.dtype = dtype

        # Explicit dimensions instead of lazy layers
        self.repeat_actions = cfg.model.get("repeat_actions", 1)
        self.action_dim = cfg.environment.action_dimension * self.repeat_actions
        self.embed_dim = cfg.model.clip.dim
        self.text_backbone = cfg.model.clip.get("text_backbone", "clip")
        self.text_dim = 512  if self.text_backbone == "clip" else 768 # CLIP text model output dim
        self.dino_version = cfg.model.clip.get("dino_version", "v2")
        self.vision_backbone = cfg.model.clip.get("vision_backbone", "dino")

        if self.vision_backbone == "dino":
            self.vision_dim = 384 if self.dino_version == "v2" else 768 # dinov2-small output dim
        elif self.vision_backbone == "siglip2":
            self.vision_dim = 768

        self.fusion_order = self.cfg.model.clip.get("fusion_order", "vision_action")
        self.freeze = self.cfg.model.clip.get("freeze", "pretrained_encoders")
        # attn_implementation = "flash_attention_2" if self.dtype in [torch.float16, torch.bfloat16] else None

        # text encoder
        if self.text_backbone == "clip":
            clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32",
                                                attn_implementation=None,
                                                device_map="auto",
                                                torch_dtype=self.dtype)
            self.text_encoder = clip.text_model
        elif self.text_backbone == "siglip2":
            self.text_encoder = Siglip2TextModel.from_pretrained("google/siglip2-base-patch16-224", device_map="auto", torch_dtype=self.dtype)
        else:
            raise ValueError(f"Unsupported text backbone: {self.cfg.model.clip.get('text_backbone', 'clip')}")

        self.text_encoder.requires_grad_(not self.freeze in ["pretrained_encoders", "language_encoder"])
        self.processor = VisionActionGroundedContrastiveProcessorWrapper(
            cfg=self.cfg.model.processor,
            horizon=self.cfg.environment.kwargs.horizon,
            action_dimension=self.action_dim,
            target_vla=self.cfg.model.clip.get("target_vla", "openvla"),
            text_backbone=self.text_backbone,
            dtype=self.dtype,
            device=device,
            custom_prompt_template=self.cfg.model.clip.get("custom_prompt_template", True)
        )

        logit_factor = cfg.model.clip.get("logit_scale_factor", 0.07)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / logit_factor, device=device, dtype=dtype)))

        self.diversity_weight = cfg.model.clip.get("diversity_weight", 0.01)
        # self.temperature = nn.Parameter(torch.log(torch.tensor(0.5, device=device, dtype=dtype)), requires_grad=True)
        # self.logit_scale_fixed = torch.ones([]) * torch.log(torch.tensor(1 / logit_factor, device=device, dtype=dtype)) # fixed temperature

        projection_type = cfg.model.clip.get("projection_type", "simple")  # simple, residual, transformer

        if projection_type == "simple":
            self.vision_proj = SimpleProjection(
                self.vision_dim, self.embed_dim, device=device, dtype=dtype
            )
            self.text_proj = SimpleProjection(
                self.text_dim, self.embed_dim, device=device, dtype=dtype
            )
        elif projection_type == "residual":
            self.vision_proj = ResidualProjection(
                self.vision_dim, self.embed_dim, device=device, dtype=dtype
            )
            self.text_proj = ResidualProjection(
                self.text_dim, self.embed_dim, device=device, dtype=dtype
            )
        print(f"Using projection method: {projection_type}")

        # action encoder
        self.action_encoder = ActionEncoder(cfg, device, dtype)
        # vision encoder

        if self.vision_backbone == "dino":
            if self.dino_version == "v2":
                self.vision_encoder = Dinov2Model.from_pretrained("facebook/dinov2-small", device_map="auto", torch_dtype=self.dtype)
            elif self.dino_version == "v3":
                self.vision_encoder = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m", device_map="auto", torch_dtype=self.dtype)
            else:
                raise ValueError(f"Unsupported DINO version: {self.dino_version}")
        elif self.vision_backbone == "siglip2":
            siglip2 = AutoModel.from_pretrained("google/siglip2-base-patch16-224", device_map="auto", torch_dtype=self.dtype).eval()
            self.vision_encoder = siglip2.vision_model

        self.vision_encoder.requires_grad_(not self.freeze in ["pretrained_encoders", "vision_encoder"])


        self.num_cross_attention_layers = cfg.model.clip.get("num_cross_attention_layers", 4)
        self.shared_projection = cfg.model.clip.get("shared_projection", "cross_attention")
        if self.shared_projection == "cross_attention":
            self.vision_action_crossattention = MultiLayerCrossAttention(
                self.embed_dim, num_layers=self.num_cross_attention_layers, num_heads=8, dropout=0.1, device=device, dtype=dtype
            )
        elif self.shared_projection == "skip":
            self.vision_action_crossattention = lambda x, y: (x, y)
        elif self.shared_projection == "film":
            self.vision_action_crossattention = MultiLayerFiLM(
                self.embed_dim, num_layers=self.num_cross_attention_layers, dropout=0.1, device=device, dtype=dtype
            )
        else:
            raise ValueError(f"Unsupported shared_projection method: {self.shared_projection}")
        print(f"Using shared_projection Method:  {self.shared_projection}")

        self.fusion_method = cfg.model.clip.get("fusion_method", "attention")  # attention, gated, mlp, hadamard
        if self.shared_projection == "film":
            self.fusion_method = "identity"  # no fusion needed after FiLM

        if self.fusion_method == "gated":
            self.vector_fusion = cfg.model.clip.get("vector_fusion", False)
            self.fusion_module = GatedFusion(
                embed_dim=self.embed_dim, vector_fusion=self.vector_fusion, device=device, dtype=dtype
            )
        elif self.fusion_method == "attention":
            self.fusion_module = AttentionBasedFusion(
                embed_dim=self.embed_dim, num_heads=8, dropout=0.1, device=device, dtype=dtype
            )
        elif self.fusion_method == "mlp":
            self.fusion_module = MLPFusion(
                embed_dim=self.embed_dim, hidden_dim=self.embed_dim * 2, dropout=0.1, device=device, dtype=dtype
            )
        elif self.fusion_method == "hadamard":
            self.fusion_module = HadamardFusion(
                embed_dim=self.embed_dim, device=device, dtype=dtype
            )
        elif self.fusion_method == "naive":
            self.fusion_module = NaiveFusion()
        elif self.fusion_method == "identity":
            self.fusion_module = nn.Identity()
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion_method}")
        print(f"Using Fusion Method:  {self.fusion_method}")

        # Layer norms for stability
        self.vision_norm = nn.LayerNorm(self.embed_dim, device=device, dtype=dtype)

        # self.output_norm = nn.LayerNorm(self.embed_dim, device=device, dtype=dtype)
        self.pooling_method = cfg.model.clip.get("pooling_method", "attention")
        if self.pooling_method == "attention":
            self.vision_pool = AttentionPooling(self.embed_dim, device=device, dtype=dtype)
            self.action_pool = AttentionPooling(self.embed_dim, device=device, dtype=dtype)
            self.text_pool = MaskedAttentionPooling(self.embed_dim, device=device, dtype=dtype)
        elif self.pooling_method == "mean":
            self.vision_pool = MeanPooling()
            self.action_pool = MeanPooling()
            self.text_pool = MeanPooling()
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling_method}")
        print(f"Using Pooling Method:  {self.pooling_method}")

        self.final_projection_text = ResidualProjection(self.embed_dim, self.embed_dim, bias=False, device=device, dtype=dtype)
        self.final_projection_action_vision = ResidualProjection(self.embed_dim, self.embed_dim, bias=False, device=device, dtype=dtype)

        self.final_text_norm = nn.LayerNorm(self.embed_dim, device=device, dtype=dtype)
        self.final_action_vision_norm = nn.LayerNorm(self.embed_dim, device=device, dtype=dtype)

        self.loss_type = cfg.model.clip.get("loss_type", "contrastive")
        if self.loss_type not in ["contrastive", "barlow_twins", "hybrid", "siglip"]:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
        if self.loss_type == "hybrid" or self.loss_type == "siglip":
            self.init_siglip_bias()
        print(f"Using Loss Type:  {self.loss_type}")

    def forward(self, input_ids, pixel_values, attention_mask = None, **kwargs):
        # text encoder
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        text_embeddings = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        text_embeddings = self.text_proj(text_embeddings)

        # action encoder
        action = kwargs["action"]
        action_embedding = self.action_encoder(action)
        # vision encoder
        vision_embedding = self.vision_encoder(pixel_values=pixel_values).pooler_output
        vision_embedding = self.vision_proj(vision_embedding)

        if self.shared_projection == "film":
            final_vision_attn = self.vision_action_crossattention(vision_embedding, action_embedding) # FiLM conditions vision on action
            final_action_attn = action_embedding # unchanged
        else:
            final_vision_attn, final_action_attn = self.vision_action_crossattention(vision_embedding, action_embedding)

        wandb.log({"debug/pooled_text_norm": text_embeddings.norm(p=2, dim=-1).mean().item()})
        action_pooled = self.action_pool(final_action_attn)
        if final_vision_attn.ndim == 3:
            vision_pooled = self.vision_pool(final_vision_attn)
        else:   # vision_embedding is already pooled
            vision_pooled = final_vision_attn

        with torch.no_grad():
            wandb.log({"debug/pooled_vision_norm": vision_embedding.norm(p=2, dim=-1).mean().item()})
            wandb.log({"debug/pooled_action_norm": action_pooled.norm(p=2, dim=-1).mean().item()})
            wandb.log({
                "debug/pooled_action_std": action_pooled.std().item(),
                "debug/pooled_vision_std": vision_embedding.std().item()
            })

        if self.shared_projection == "film":
            fused_embeddings = vision_pooled
        else:
            fused_embeddings, _ = self.fusion_module(vision_pooled, action_pooled)

        with torch.no_grad():
            wandb.log({
                "debug/pooled_text_std": text_embeddings.std().item(),
                "debug/pre_proj_fused_std": fused_embeddings.std().item()
            })

        loss, logits, similarities = self.calculate_loss(
            text_embeddings, fused_embeddings
        )

        return DotDict({
            "loss": loss,
            "logits": logits,
            "similarities": similarities,
            "text_embeddings": text_embeddings,
            "action_vision_embeddings": fused_embeddings
        })

    def calculate_loss(self, text_embeddings, action_vision_embeddings):
        if self.loss_type == "contrastive":
            return self.contrastive(text_embeddings, action_vision_embeddings)
        elif self.loss_type == "barlow_twins":
            return self.barlow_twins_loss(text_embeddings, action_vision_embeddings)
        elif self.loss_type == "siglip":
            return self.siglip_loss(text_embeddings, action_vision_embeddings)
        elif self.loss_type == "hybrid":
            return self.hybrid_loss(text_embeddings, action_vision_embeddings)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

    def contrastive(self, text_embeddings, action_vision_embeddings):
        text_embeddings = self.safe_normalize(text_embeddings, dim=-1, eps=1e-6)
        action_vision_embeddings = self.safe_normalize(action_vision_embeddings, dim=-1, eps=1e-6)

        logit_scale = self.logit_scale.exp() # .clamp(max=200)

        similarities = text_embeddings @ action_vision_embeddings.T
        logits = similarities * logit_scale

        # Ground truth: diagonal is positive match
        labels = torch.arange(text_embeddings.size(0)).to(self.device)

        loss_I = nn.functional.cross_entropy(logits.T, labels, label_smoothing=self.cfg.model.get("label_smoothing", 0.0))
        loss_T = nn.functional.cross_entropy(logits, labels, label_smoothing=self.cfg.model.get("label_smoothing", 0.0))

        contrastive_loss = (loss_I + loss_T)/2.0

        with torch.no_grad():
            wandb.log({"debug/contrastive_loss": contrastive_loss.item()})

        if self.cfg.model.clip.get("use_hard_negative_mining", False):
            hard_neg_loss = self.hard_negative_mining_loss(logits, logit_scale)
            alpha = self.cfg.model.clip.get("hard_neg_weight", 0.5)
            contrastive_loss = (1 - alpha) * contrastive_loss + alpha * hard_neg_loss
            with torch.no_grad():
                wandb.log({"debug/hard_negative_loss": hard_neg_loss.item()})

        diversity_loss = self.diversity_weight * self.diversity_regularization(text_embeddings, action_vision_embeddings)

        with torch.no_grad():
            wandb.log({"debug/weighted_diversity_loss": diversity_loss.item()})

        total_loss = contrastive_loss + diversity_loss

        return total_loss, logits, similarities

    def barlow_twins_loss(self, text_embeddings, action_vision_embeddings):
        """
        Barlow Twins loss for aligning text and action-vision embeddings.
        Maximizes diagonal of cross-correlation matrix (agreement) while
        minimizing off-diagonal (redundancy reduction).
        """
        # Normalize embeddings
        text_embeddings = self.safe_normalize(text_embeddings, dim=-1, eps=1e-6)
        action_vision_embeddings = self.safe_normalize(action_vision_embeddings, dim=-1, eps=1e-6)

        batch_size = text_embeddings.size(0)

        # Compute cross-correlation matrix between the two modalities
        # C has shape [embedding_dim, embedding_dim]
        c = (text_embeddings.T @ action_vision_embeddings) / batch_size

        # Loss: encourage diagonal to be 1, off-diagonal to be 0
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self._off_diagonal(c).pow_(2).sum()

        # Lambda weight for off-diagonal (typically 0.005 - 0.05)
        lambda_param = self.cfg.model.get("barlow_lambda", 0.005)
        barlow_loss = on_diag + lambda_param * off_diag

        # For logging: compute similarities for compatibility
        similarities = text_embeddings @ action_vision_embeddings.T
        logits = similarities  # No temperature scaling in Barlow Twins

        return barlow_loss, logits, similarities


    def siglip_loss(self, text_embeddings, action_vision_embeddings):
        """
        SigLIP loss using sigmoid instead of softmax for contrastive learning.
        Treats each pair independently rather than competing across batch.
        """
        # Normalize embeddings
        text_embeddings = self.safe_normalize(text_embeddings, dim=-1, eps=1e-6)
        action_vision_embeddings = self.safe_normalize(action_vision_embeddings, dim=-1, eps=1e-6)

        batch_size = text_embeddings.size(0)

        # Compute similarity matrix
        logit_scale = self.logit_scale.exp().clamp(max=100)  # Prevent overflow
        raw_similarities = text_embeddings @ action_vision_embeddings.T
        similarities = raw_similarities * logit_scale

        # Add bias term (if it exists) *before* calculating loss
        if hasattr(self, 'siglip_bias'):
            logits = similarities + self.siglip_bias
        else:
            logits = similarities

        # Create labels: 1 for diagonal (positive pairs), -1 for off-diagonal (negatives)
        labels = 2 * torch.eye(batch_size, device=self.device) - 1

        # SigLIP uses binary cross entropy with logits for each pair independently
        # Loss = -log(sigmoid(sim)) for positives, -log(sigmoid(-sim)) for negatives
        # This simplifies to: log(1 + exp(-labels * similarities))
        siglip_loss = -F.logsigmoid(labels * logits).sum(dim=-1).mean() # pylint: disable=not-callable
        with torch.no_grad():
            wandb.log({"debug/siglip_loss": siglip_loss.item()})

        if self.cfg.model.clip.get("use_hard_negative_mining", False):
            hard_neg_loss = self.hard_negative_mining_loss(logits, logit_scale)
            alpha = self.cfg.model.clip.get("hard_neg_weight", 0.5)
            siglip_loss = (1 - alpha) * siglip_loss + alpha * hard_neg_loss
            with torch.no_grad():
                wandb.log({"debug/hard_negative_loss": hard_neg_loss.item()})

        diversity_loss = self.diversity_weight * self.diversity_regularization(text_embeddings, action_vision_embeddings)

        with torch.no_grad():
            wandb.log({"debug/weighted_diversity_loss": diversity_loss.item()})
        total_loss = siglip_loss + diversity_loss

        with torch.no_grad():
            pos_mask = torch.eye(batch_size, device=self.device).bool()
            neg_mask = ~pos_mask

            pos_sims_raw = raw_similarities[pos_mask]
            neg_sims_raw = raw_similarities[neg_mask]
            pos_logits = logits[pos_mask]
            neg_logits = logits[neg_mask]

            # Individual loss contributions
            pos_losses = -F.logsigmoid(pos_logits)
            neg_losses = -F.logsigmoid(-neg_logits)

            wandb.log({
                # Loss breakdown
                "loss/siglip_total": siglip_loss.item(),
                "loss/pos_contribution": pos_losses.mean().item(),
                "loss/neg_contribution": neg_losses.mean().item(),
                "loss/diversity_weighted": diversity_loss.item(),

                # Raw similarities (cosine, range [-1, 1])
                "sim/pos_raw_mean": pos_sims_raw.mean().item(),
                "sim/pos_raw_std": pos_sims_raw.std().item(),
                "sim/pos_raw_min": pos_sims_raw.min().item(),
                "sim/pos_raw_max": pos_sims_raw.max().item(),

                "sim/neg_raw_mean": neg_sims_raw.mean().item(),
                "sim/neg_raw_std": neg_sims_raw.std().item(),

                # After scaling
                "sim/pos_logits_mean": pos_logits.mean().item(),
                "sim/neg_logits_mean": neg_logits.mean().item(),

                # Gap metrics
                "sim/gap_raw": (pos_sims_raw.mean() - neg_sims_raw.mean()).item(),
                "sim/gap_logits": (pos_logits.mean() - neg_logits.mean()).item(),

                # Scale
                "scale/logit_scale": logit_scale.item(),

                # Check for issues
                "debug/has_nan": torch.isnan(logits).any().item(),
                "debug/all_sims_std": raw_similarities.std().item(),
            })


        return total_loss, logits, raw_similarities

    def hard_negative_mining_loss(self, logits: torch.Tensor, logit_scale: float) -> torch.Tensor:
        """
        Calculates the margin-based hard negative mining loss.

        Args:
            logits (torch.Tensor): The similarity matrix (BxB).
            logit_scale (float): The current temperature/scale parameter.

        Returns:
            torch.Tensor: The mean symmetric hard negative loss.
        """
        # Ensure we are not tracking gradients for the hard-negative selection step
        with torch.no_grad():
            batch_size = logits.size(0)

            # 1. Create a mask to select only negative pairs (off-diagonal elements)
            # This is where your original `neg_mask` is used
            neg_mask = ~torch.eye(batch_size, device=self.device, dtype=torch.bool)

            # --- Text -> Vision Direction (Row-wise max) ---
            # Mask out positive pairs by setting their similarity to -inf
            # This ensures they are ignored when finding the max (hardest negative)
            neg_logits_T = logits.masked_fill(~neg_mask, -float('inf'))

            # Find the max similarity for each text row (the hardest negative image)
            hard_neg_logits_T = neg_logits_T.max(dim=1).values

            # --- Vision -> Text Direction (Column-wise max) ---
            # Mask out positive pairs (neg_mask is symmetric)
            neg_logits_I = logits.masked_fill(~neg_mask, -float('inf'))

            # Find the max similarity for each image column (the hardest negative text)
            # The .values ensures we only get the max similarity value
            hard_neg_logits_I = neg_logits_I.max(dim=0).values

        # 2. Get the positive pair similarities
        pos_logits = logits.diag()

        # 3. Calculate the Margin Loss (must be done WITH gradients)

        # Retrieve configuration settings
        margin = self.cfg.model.clip.get("hard_neg_margin", 0.2) * logit_scale

        # Triplet-like Loss: max(0, margin + hard_neg_sim - pos_sim)
        # This encourages pos_sim to be greater than hard_neg_sim by at least 'margin'

        # Text anchor loss
        hard_neg_loss_T = F.relu(margin + hard_neg_logits_T - pos_logits).mean()

        # Image anchor loss
        hard_neg_loss_I = F.relu(margin + hard_neg_logits_I - pos_logits).mean()

        # Return the symmetric mean loss
        return (hard_neg_loss_T + hard_neg_loss_I) / 2.0

    def diversity_regularization(self, text_embeddings, action_vision_embeddings):
        # Diversity regularization for both modalities
        text_sim_matrix = text_embeddings @ text_embeddings.T
        vision_sim_matrix = action_vision_embeddings @ action_vision_embeddings.T

        batch_size = text_embeddings.size(0)
        mask = ~torch.eye(batch_size, device=self.device, dtype=torch.bool)

        # Extract off-diagonal elements
        text_off_diag = text_sim_matrix[mask]
        vision_off_diag = vision_sim_matrix[mask]

        # Only penalize positive similarities
        text_diversity_loss = torch.clamp(text_off_diag, min=0).mean()
        vision_diversity_loss = torch.clamp(vision_off_diag, min=0).mean()

        diversity_loss = text_diversity_loss + vision_diversity_loss
        return diversity_loss

    def hybrid_loss(self, text_embeddings, action_vision_embeddings):
        """
        Hybrid approach combining SigLIP contrastive with Barlow Twins regularization.
        Good balance of semantic discrimination and continuous smoothness.
        """
        # Normalize embeddings
        text_embeddings = self.safe_normalize(text_embeddings, dim=-1, eps=1e-6)
        action_vision_embeddings = self.safe_normalize(action_vision_embeddings, dim=-1, eps=1e-6)

        batch_size = text_embeddings.size(0)

        # 1. SigLIP contrastive loss (for cross-modal alignment)
        logit_scale = self.logit_scale.exp().clamp(max=100)
        raw_similarities = text_embeddings @ action_vision_embeddings.T
        similarities = raw_similarities  * logit_scale
        labels = 2 * torch.eye(batch_size, device=self.device) - 1
        contrastive_loss = -F.logsigmoid(labels * similarities).sum() / batch_size # pylint: disable=not-callable

        # 2. Barlow Twins regularization (for redundancy reduction)
        c_text = (text_embeddings.T @ text_embeddings) / batch_size
        c_vision = (action_vision_embeddings.T @ action_vision_embeddings) / batch_size

        # Encourage decorrelation within each modality
        off_diag_text = self._off_diagonal(c_text).pow_(2).sum()
        off_diag_vision = self._off_diagonal(c_vision).pow_(2).sum()

        lambda_barlow = self.cfg.model.get("barlow_lambda", 0.005)
        barlow_reg = lambda_barlow * (off_diag_text + off_diag_vision)

        wandb.log({"debug/barlow_on_diag": torch.diagonal(c_text).mean().item()})
        wandb.log({"debug/barlow_off_diag": (off_diag_text + off_diag_vision).item()})
        wandb.log({"debug/siglip_loss": contrastive_loss.item()})
        wandb.log({"debug/barlow_reg": barlow_reg.item()})

        # 3. Combined loss
        alpha = self.cfg.model.get("hybrid_alpha", 0.5)  # Weight between losses
        total_loss = contrastive_loss + alpha * barlow_reg

        logits = similarities

        return total_loss, logits, raw_similarities

    def _off_diagonal(self, matrix):
        """
        Helper function to get off-diagonal elements of a square matrix.
        """
        n = matrix.shape[0]
        return matrix.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    # Optional: Learnable bias for SigLIP (add to your model's __init__)
    def init_siglip_bias(self):
        """
        Initialize learnable bias for SigLIP loss.
        Helps with better calibration of similarity scores.
        """
        initial_bias = -4.0
        self.siglip_bias = nn.Parameter(
            torch.tensor([initial_bias], device=self.device, dtype=self.dtype)
        )

    def safe_normalize(self, x, dim=-1, eps=1e-6):
        return F.normalize(x, p=2, eps=eps, dim=dim)

    @torch.no_grad()
    def score(self, trajectory, text, frames, instruction):
        batch = {
            "action": trajectory,
            "instruction": [instruction],
            "frames": frames,
            "captions": [text]
        }
        model_inputs = self.process_batch(batch)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask", None)
        pixel_values = model_inputs["pixel_values"]
        action = model_inputs["action"]
        output = self(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, action=action)
        return output.logits

    @torch.no_grad()
    def process_batch(self, batch):
        return self.processor.process_batch(batch)
