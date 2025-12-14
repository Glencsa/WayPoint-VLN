import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from transformers import AutoModel, AutoConfig, AutoImageProcessor
import timm

class VisionEncoder(nn.Module):
    """
    3D Vision encoder based on RGB + Depth (Depth Anything Backbone)
    """
    
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224", # RGB Encoder base
        depth_model_name: str = "LiheYoung/depth-anything-small-hf", # Depth Encoder base
        pretrained_path: Optional[str] = None,
        freeze_vision: bool = False,
        freeze_depth: bool = False,
        image_size: int = 224,
        patch_size: int = 14, # Depth Anything通常基于DINOv2, patch_size往往是14
        hidden_size: int = 4096,
    ):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        
        # 1. Build 2D RGB Image Encoder
        self.image_encoder = self._build_image_encoder(model_name)
        
        # 2. Build Depth Encoder (Based on Depth Anything)
        self.depth_encoder = self._build_depth_encoder(depth_model_name)
        
        # 3. Feature Fusion
        # Concatenate RGB features + Depth features
        fusion_input_dim = self.image_encoder.output_dim + self.depth_encoder.output_dim
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        
        # Freeze encoders if specified
        if freeze_vision:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        
        if freeze_depth:
            for param in self.depth_encoder.parameters():
                param.requires_grad = False
                
        # Load pretrained weights if provided (Custom checkpoint for the whole wrapper)
        if pretrained_path:
            self.load_pretrained_weights(pretrained_path)
    
    def _build_image_encoder(self, model_name):
        """Build Standard 2D image encoder (e.g., ViT or CLIP)"""
        try:
            # Using timm for flexibility
            image_encoder = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=0, # Remove classification head
                global_pool="avg", # Global average pooling
            )
            image_encoder.output_dim = image_encoder.num_features
        except Exception:
            # Fallback to simple ViT constructor if timm fails or custom name
            print(f"Warning: Could not load {model_name} via timm, using transformers CLIP")
            from transformers import CLIPVisionModel
            image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
            image_encoder.output_dim = image_encoder.config.hidden_size
            # Wrap forward to match timm style (return pooled output)
            original_forward = image_encoder.forward
            image_encoder.forward = lambda x: original_forward(x).pooler_output
            
        return image_encoder

    def _build_depth_encoder(self, model_name):
        """Build Depth encoder based on Depth Anything backbone"""
        print(f"Loading Depth Anything backbone: {model_name}")
        try:
            # Load the backbone from Depth Anything (which is essentially a DINOv2 encoder)
            # We use AutoModel to load the encoder part. 
            # Note: LiheYoung/depth-anything-small-hf is a DepthEstimator, we need its backbone.
            from transformers import AutoModelForDepthEstimation
            
            full_model = AutoModelForDepthEstimation.from_pretrained(model_name)
            depth_encoder = full_model.backbone
            
            # Identify output dimension
            depth_encoder.output_dim = full_model.config.hidden_size
            
        except Exception as e:
            print(f"Error loading Depth Anything: {e}. Fallback to standard DINOv2.")
            # Fallback: Use standard DINOv2 from timm as a proxy for Depth Anything architecture
            depth_encoder = timm.create_model(
                "vit_small_patch14_dinov2.lvd142m",
                pretrained=True,
                num_classes=0,
                global_pool="avg"
            )
            depth_encoder.output_dim = depth_encoder.num_features
            
        return depth_encoder
    
    def load_pretrained_weights(self, pretrained_path: str):
        try:
            checkpoint = torch.load(pretrained_path, map_location="cpu")
            if "model" in checkpoint:
                checkpoint = checkpoint["model"]
            self.load_state_dict(checkpoint, strict=False)
            print(f"Loaded pretrained vision weights from {pretrained_path}")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
    
    def forward(self, images: torch.Tensor, depth_images: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            images: (batch_size, num_images, 3, H, W) - RGB
            depth_images: (batch_size, num_images, 1, H, W) or (B, N, 3, H, W) - Depth Maps
            
        Returns:
            vision_features: (batch_size, num_images, hidden_size)
        """
        batch_size, num_images = images.shape[:2]
        
        # 1. Process RGB Images
        images_flat = images.view(-1, *images.shape[2:]) # (B*N, 3, H, W)
        image_features = self.image_encoder(images_flat) # (B*N, image_dim)
        
        # Handle cases where encoder returns (B, Seq, Dim) instead of pooled (B, Dim)
        if len(image_features.shape) == 3:
            image_features = image_features.mean(dim=1)
            
        # 2. Process Depth Images (if available)
        if depth_images is not None:
            # Flatten batch and sequence dims
            depth_flat = depth_images.view(-1, *depth_images.shape[2:]) # (B*N, C, H, W)
            
            # Handle Channel Dimension:
            # Depth Anything (and most ViTs) expect 3 channels. 
            # If input is 1 channel depth, we repeat it.
            if depth_flat.shape[1] == 1:
                depth_flat = depth_flat.repeat(1, 3, 1, 1)
            
            # Extract features using Depth Anything backbone
            # Note: HuggingFace Backbones usually output a specific object, we need last_hidden_state
            if hasattr(self.depth_encoder, "forward_features"): # timm style
                depth_features = self.depth_encoder.forward_features(depth_flat)
            else: # transformers style
                outputs = self.depth_encoder(depth_flat)
                if hasattr(outputs, "feature_maps"):
                     # Some backbones return list of feature maps, take the last one
                    depth_features = outputs.feature_maps[-1]
                elif hasattr(outputs, "last_hidden_state"):
                    depth_features = outputs.last_hidden_state
                else:
                    depth_features = outputs[0]

            # Global Pooling for Depth Features
            if len(depth_features.shape) == 3: # (B*N, Seq, Dim) -> (B*N, Dim)
                # Exclude CLS token if present (usually index 0), or just mean all
                depth_features = depth_features.mean(dim=1)
            elif len(depth_features.shape) == 4: # (B*N, Dim, H, W) -> (B*N, Dim)
                depth_features = depth_features.mean(dim=[2, 3])
                
            # 3. Fuse Features
            # Concatenate RGB and Depth vectors
            combined_features = torch.cat([image_features, depth_features], dim=-1)
            vision_features = self.feature_fusion(combined_features)
            
        else:
            # Fallback if no depth provided (though architecture expects it)
            # You might want to handle this differently (e.g. padding)
            print("Warning: No depth images provided to Depth-enabled encoder.")
            # Create dummy depth features or just project RGB (might cause shape mismatch if not handled)
            # For now, we assume depth is always present in this mode.
            raise ValueError("Depth images are required for this configuration")

        # Reshape back to (batch_size, num_images, hidden_size)
        vision_features = vision_features.view(batch_size, num_images, -1)
        
        return vision_features

class LanguageEncoder(nn.Module):
    """
    Language encoder based on LLaMA-2
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        pretrained_path: Optional[str] = None,
        freeze_lm: bool = False,
        hidden_size: int = 4096,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.hidden_size = hidden_size
        
        # Load language model
        self.language_model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        
        self.language_hidden_size = self.language_model.config.hidden_size
        
        # Projection layer to match hidden size
        if self.language_hidden_size != hidden_size:
            self.language_proj = nn.Linear(self.language_hidden_size, hidden_size)
        else:
            self.language_proj = nn.Identity()
            
        # Freeze language model if specified
        if freeze_lm:
            for param in self.language_model.parameters():
                param.requires_grad = False
                
        # Load pretrained weights if provided
        if pretrained_path:
            self.load_pretrained_weights(pretrained_path)
    
    def load_pretrained_weights(self, pretrained_path: str):
        """Load pretrained language model weights"""
        try:
            checkpoint = torch.load(pretrained_path, map_location="cpu")
            if "model" in checkpoint:
                checkpoint = checkpoint["model"]
            self.language_model.load_state_dict(checkpoint, strict=False)
            print(f"Loaded pretrained language weights from {pretrained_path}")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights from {pretrained_path}: {e}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through language encoder
        
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            
        Returns:
            language_features: (batch_size, seq_len, hidden_size)
        """
        # Get language model outputs
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        
        # Extract hidden states
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden)
        
        # Project to target hidden size
        hidden_states = self.language_proj(hidden_states)
        
        return hidden_states


class MultimodalFusion(nn.Module):
    """
    Multimodal fusion module using cross-attention
    """
    
    def __init__(
        self,
        hidden_size: int = 4096,
        num_heads: int = 32,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size)
            for _ in range(num_layers)
        ])
        
        # Feed-forward networks
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 4, hidden_size),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])
        
        # Final projection
        self.final_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(
        self,
        vision_features: torch.Tensor,
        language_features: torch.Tensor,
        vision_mask: Optional[torch.Tensor] = None,
        language_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through multimodal fusion
        
        Args:
            vision_features: (batch_size, num_images, hidden_size)
            language_features: (batch_size, seq_len, hidden_size)
            vision_mask: (batch_size, num_images) - 1 for valid images, 0 for padding
            language_mask: (batch_size, seq_len) - 1 for valid tokens, 0 for padding
            
        Returns:
            fused_features: (batch_size, seq_len, hidden_size)
        """
        # Use language features as query, vision features as key/value
        query = language_features
        key_value = vision_features
        
        # Create attention mask for vision features
        if vision_mask is not None:
            # Convert to attention mask format (True for valid positions)
            vision_attention_mask = vision_mask.bool()
        else:
            vision_attention_mask = None
            
        # Apply cross-attention layers
        for i, (cross_attn, layer_norm, ffn) in enumerate(
            zip(self.cross_attention_layers, self.layer_norms, self.ffns)
        ):
            # Cross-attention
            attn_output, _ = cross_attn(
                query=query,
                key=key_value,
                value=key_value,
                key_padding_mask=~vision_attention_mask if vision_attention_mask is not None else None,
            )
            
            # Residual connection and layer norm
            query = layer_norm(query + attn_output)
            
            # Feed-forward network
            ffn_output = ffn(query)
            query = query + ffn_output
            
        # Final projection
        fused_features = self.final_proj(query)
        
        return fused_features



class RVLNBackbone(nn.Module):
    """
    RVLN backbone model with Depth Anything integration
    """
    
    def __init__(
        self,
        vision_config: Dict[str, Any],
        language_config: Dict[str, Any],
        fusion_config: Dict[str, Any],
    ):
        super().__init__()
        
        # Initialize encoders
        self.vision_encoder = VisionEncoder(**vision_config)
        
        # Re-initialize Language Encoder (assuming class exists in context)
        # self.language_encoder = LanguageEncoder(**language_config)
        # Note: In a real script, ensure LanguageEncoder is defined or imported
        # For this snippet to run standalone, I'll mock it or assume previous def:
        self.language_encoder = LanguageEncoder(**language_config) 
        
        self.multimodal_fusion = MultimodalFusion(**fusion_config)
        
        self.hidden_size = fusion_config["hidden_size"]
        
    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_mask: Optional[torch.Tensor] = None,
        depth_images: Optional[torch.Tensor] = None, # Changed from point_clouds
    ) -> torch.Tensor:
        """
        Args:
            images: (batch_size, num_images, 3, H, W)
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            image_mask: (batch_size, num_images)
            depth_images: (batch_size, num_images, 1, H, W) - New Input
        """
        # Encode vision (RGB + Depth)
        vision_features = self.vision_encoder(images, depth_images) 
        
        # Encode language
        language_features = self.language_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Multimodal fusion
        fused_features = self.multimodal_fusion(
            vision_features=vision_features,
            language_features=language_features,
            vision_mask=image_mask,
            language_mask=attention_mask,
        )
        
        return fused_features