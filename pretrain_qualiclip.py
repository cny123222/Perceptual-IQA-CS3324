"""
QualiCLIP-Style Self-Supervised Pretraining for Swin Encoder

Pretrain the Swin-Base image encoder using quality-aware contrastive learning
before fine-tuning on supervised IQA tasks.
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import random
import numpy as np
from datetime import datetime
import torchvision.transforms as transforms

# Import our modules
from qualiclip_pretrain.degradation_generator import SyntheticDegradation, RandomDegradation
from qualiclip_pretrain.qualiclip_loss import QualiCLIPLoss, SimplifiedQualiCLIPLoss
from qualiclip_pretrain.pretrain_dataset import QualityAwarePretrainDataset, load_koniq_train_images
import models_swin


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Auto-detect device: CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_clip_text_encoder(device):
    """
    Load CLIP text encoder for extracting text embeddings.
    We use the local QualiCLIP CLIP implementation.
    """
    try:
        # Try OpenAI CLIP first
        import clip
        print("Loading CLIP text encoder (OpenAI)...")
        model, _ = clip.load("RN50", device=device)
        model.eval()
        
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        
        print("✓ CLIP text encoder loaded successfully (OpenAI)")
        return model, 'openai'
    except ImportError:
        # Fallback to QualiCLIP's CLIP implementation
        print("OpenAI CLIP not found, using QualiCLIP's CLIP implementation...")
        import sys
        import os
        qualiclip_path = os.path.join(os.path.dirname(__file__), 'benchmarks', 'QualiCLIP')
        sys.path.insert(0, qualiclip_path)
        
        try:
            from clip import clip as qualiclip
            print("Loading CLIP text encoder (QualiCLIP)...")
            model, _ = qualiclip.load("RN50", device=device)
            model.eval()
            
            # Freeze all parameters
            for param in model.parameters():
                param.requires_grad = False
            
            print("✓ CLIP text encoder loaded successfully (QualiCLIP)")
            return model, 'qualiclip'
        except Exception as e:
            print(f"Error: Failed to load CLIP from either source")
            print(f"  OpenAI CLIP: Not installed")
            print(f"  QualiCLIP CLIP: {e}")
            print("\nPlease install CLIP using one of these methods:")
            print("  1. pip install git+https://github.com/openai/CLIP.git")
            print("  2. Or via SSH: git clone git@github.com:openai/CLIP.git && cd CLIP && pip install -e .")
            raise
    except Exception as e:
        print(f"Error loading CLIP: {e}")
        raise


def extract_text_features(clip_model, device, clip_type='openai'):
    """
    Extract and cache text features for quality-related prompts.
    
    Returns:
        text_features_pos: Features for "Good photo" prompt
        text_features_neg: Features for "Bad photo" prompt
    """
    if clip_type == 'openai':
        import clip
        tokenize_fn = clip.tokenize
    else:
        # Use QualiCLIP's tokenizer
        import sys
        import os
        qualiclip_path = os.path.join(os.path.dirname(__file__), 'benchmarks', 'QualiCLIP')
        sys.path.insert(0, qualiclip_path)
        from clip import clip as qualiclip
        tokenize_fn = qualiclip.tokenize
    
    # Quality-related text prompts
    prompts_positive = ["Good photo", "High quality image", "Clear image"]
    prompts_negative = ["Bad photo", "Low quality image", "Blurry image"]
    
    # Use the simplest prompts (as in QualiCLIP paper)
    text_pos = tokenize_fn(["Good photo"]).to(device)
    text_neg = tokenize_fn(["Bad photo"]).to(device)
    
    with torch.no_grad():
        text_features_pos = clip_model.encode_text(text_pos).squeeze(0)  # [D]
        text_features_neg = clip_model.encode_text(text_neg).squeeze(0)  # [D]
    
    print(f"✓ Text features extracted: {text_features_pos.shape}")
    return text_features_pos, text_features_neg


def create_image_encoder(model_size='base', drop_path_rate=0.2, device='cuda'):
    """
    Create Swin image encoder (just the backbone, no HyperNet).
    
    Args:
        model_size: 'tiny', 'small', or 'base'
        drop_path_rate: Stochastic depth rate
        device: Device to load model on
        
    Returns:
        encoder: Swin backbone model
        feature_dim: Output feature dimension
    """
    print(f"Creating Swin-{model_size.upper()} image encoder...")
    
    # Use the SwinBackbone from models_swin.py
    # We need to determine the output dimension
    if model_size == 'base':
        feature_dim = 1024  # Last stage of Swin-Base
    elif model_size == 'small' or model_size == 'tiny':
        feature_dim = 768  # Last stage of Swin-Tiny/Small
    else:
        raise ValueError(f"Unknown model size: {model_size}")
    
    # Create encoder
    # For pretraining, we only need the backbone, not the full HyperNet
    # We'll use a simplified wrapper
    lda_out_channels = 16
    target_in_size = 224
    
    encoder = models_swin.swin_backbone(
        lda_out_channels=lda_out_channels,
        in_chn=target_in_size,
        pretrained=True,
        drop_path_rate=drop_path_rate,
        model_size=model_size
    ).to(device)
    
    print(f"✓ Swin encoder created (feature_dim={feature_dim})")
    return encoder, feature_dim


class SwinImageEncoder(nn.Module):
    """
    Wrapper for Swin backbone to extract image features for pretraining.
    
    Takes Swin backbone and adds a projection head to match CLIP's embedding dimension.
    """
    
    def __init__(self, swin_backbone, feature_dim=1024, embed_dim=1024):
        super(SwinImageEncoder, self).__init__()
        self.backbone = swin_backbone
        self.feature_dim = feature_dim
        
        # Projection head to map Swin features to CLIP embedding space
        # CLIP RN50 uses 1024-d embeddings
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] image tensor
            
        Returns:
            embeddings: [B, embed_dim] normalized embeddings
        """
        # Extract features from Swin backbone
        out = self.backbone(x)
        
        # Use the highest-level feature (feat3) for global representation
        feat3 = out['hyper_in_feat']  # [B, feature_dim, 7, 7]
        
        # Global average pooling
        global_feat = torch.nn.functional.adaptive_avg_pool2d(feat3, (1, 1))  # [B, feature_dim, 1, 1]
        global_feat = global_feat.flatten(1)  # [B, feature_dim]
        
        # Project to CLIP embedding space
        embeddings = self.projection(global_feat)  # [B, embed_dim]
        
        return embeddings


def pretrain_epoch(encoder, dataloader, criterion, optimizer, device, 
                   distortion_types, num_levels=5, text_features_pos=None, 
                   text_features_neg=None):
    """
    Run one epoch of pretraining.
    
    Args:
        encoder: Image encoder model
        dataloader: Training data loader
        criterion: QualiCLIP loss function
        optimizer: Optimizer
        device: Device
        distortion_types: List of distortion types to use
        num_levels: Number of degradation levels
        text_features_pos: Cached "Good photo" features
        text_features_neg: Cached "Bad photo" features
        
    Returns:
        avg_loss: Average loss for the epoch
        loss_components: Dictionary of loss component averages
    """
    encoder.train()
    
    total_loss = 0.0
    total_losses = {'total': 0.0, 'cons': 0.0, 'pos': 0.0, 'neg': 0.0}
    num_batches = 0
    
    pbar = tqdm(dataloader, desc='Pretraining')
    
    for crop1_batch, crop2_batch, img_indices in pbar:
        batch_size = len(crop1_batch)
        
        # Prepare lists to collect all features and metadata
        all_image_features = []
        all_degradation_levels = []
        all_pair_indices = []
        
        # Process each image in the batch
        for i in range(batch_size):
            crop1 = crop1_batch[i]
            crop2 = crop2_batch[i]
            img_idx = img_indices[i].item()
            
            # Randomly select a distortion type for this image pair
            distortion_type = random.choice(distortion_types)
            degrader = SyntheticDegradation(distortion_type, num_levels=num_levels)
            
            # Apply degradation to both crops
            degraded_crop1 = degrader(crop1)  # List of num_levels images
            degraded_crop2 = degrader(crop2)  # List of num_levels images
            
            # Stack all degraded crops
            all_crops = degraded_crop1 + degraded_crop2  # 2 * num_levels images
            
            # Convert to tensors and stack
            crop_tensors = torch.stack([
                torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                for img in all_crops
            ]).to(device)  # [2*num_levels, 3, H, W]
            
            # Extract features
            with torch.set_grad_enabled(True):
                features = encoder(crop_tensors)  # [2*num_levels, embed_dim]
            
            all_image_features.append(features)
            
            # Create metadata
            # Degradation levels: [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
            levels = torch.cat([
                torch.arange(num_levels),
                torch.arange(num_levels)
            ]).to(device)
            all_degradation_levels.append(levels)
            
            # Pair indices: all from same image, use a unique ID
            pair_idx = torch.full((2 * num_levels,), img_idx, dtype=torch.long).to(device)
            all_pair_indices.append(pair_idx)
        
        # Concatenate all features and metadata
        image_features = torch.cat(all_image_features, dim=0)  # [B*2*L, D]
        degradation_levels = torch.cat(all_degradation_levels, dim=0)  # [B*2*L]
        pair_indices = torch.cat(all_pair_indices, dim=0)  # [B*2*L]
        
        # Compute loss
        loss, loss_dict = criterion(
            image_features,
            text_features_pos,
            text_features_neg,
            degradation_levels,
            pair_indices
        )
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        for key in loss_dict:
            total_losses[key] += loss_dict[key]
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'cons': f"{loss_dict.get('cons', 0):.4f}",
            'rank': f"{loss_dict.get('pos', 0) + loss_dict.get('neg', 0):.4f}"
        })
    
    # Compute averages
    avg_loss = total_loss / num_batches
    avg_losses = {key: val / num_batches for key, val in total_losses.items()}
    
    return avg_loss, avg_losses


def main():
    parser = argparse.ArgumentParser(description='QualiCLIP-style pretraining for Swin encoder')
    
    # Data
    parser.add_argument('--data_root', type=str, default='/root/Perceptual-IQA-CS3324/koniq-10k',
                       help='Path to KonIQ-10k dataset')
    parser.add_argument('--crop_size', type=int, default=224,
                       help='Size of random crops')
    parser.add_argument('--base_size', type=int, default=512,
                       help='Resize images to this size before cropping')
    parser.add_argument('--overlap_ratio', type=float, default=0.5,
                       help='Minimum overlap between two crops')
    
    # Model
    parser.add_argument('--model_size', type=str, default='base', choices=['tiny', 'small', 'base'],
                       help='Swin model size')
    parser.add_argument('--drop_path_rate', type=float, default=0.2,
                       help='Stochastic depth rate')
    parser.add_argument('--embed_dim', type=int, default=1024,
                       help='CLIP embedding dimension')
    
    # Training
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of pretraining epochs')
    parser.add_argument('--batch_size', type=int, default=8,  # Reduced for memory
                       help='Batch size (note: effective batch is larger due to degradation levels)')
    parser.add_argument('--lr', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Degradation
    parser.add_argument('--distortion_types', type=str, nargs='+', 
                       default=['blur', 'jpeg', 'noise', 'brightness'],
                       help='Types of distortions to use')
    parser.add_argument('--num_levels', type=int, default=5,
                       help='Number of degradation levels')
    
    # Loss
    parser.add_argument('--loss_type', type=str, default='simplified', choices=['full', 'simplified'],
                       help='Type of QualiCLIP loss to use')
    parser.add_argument('--lambda_cons', type=float, default=1.0,
                       help='Weight for consistency loss')
    parser.add_argument('--lambda_rank', type=float, default=1.0,
                       help='Weight for ranking loss')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save_dir', type=str, default='/root/Perceptual-IQA-CS3324/checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--save_freq', type=int, default=5,
                       help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Setup
    device = get_device()
    print(f"Using device: {device}")
    print("="*80)
    print("QualiCLIP-Style Self-Supervised Pretraining")
    print("="*80)
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(args.save_dir, f'qualiclip_pretrain_{timestamp}')
    os.makedirs(save_path, exist_ok=True)
    print(f"Checkpoints will be saved to: {save_path}")
    
    # Load CLIP text encoder
    print("\n[1/6] Loading CLIP text encoder...")
    clip_model, clip_type = load_clip_text_encoder(device)
    text_features_pos, text_features_neg = extract_text_features(clip_model, device, clip_type)
    
    # Create image encoder
    print("\n[2/6] Creating Swin image encoder...")
    swin_backbone, feature_dim = create_image_encoder(
        model_size=args.model_size,
        drop_path_rate=args.drop_path_rate,
        device=device
    )
    encoder = SwinImageEncoder(swin_backbone, feature_dim=feature_dim, 
                              embed_dim=args.embed_dim).to(device)
    
    num_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad) / 1e6
    print(f"  Trainable parameters: {num_params:.2f}M")
    
    # Load dataset
    print("\n[3/6] Loading dataset...")
    image_paths = load_koniq_train_images(args.data_root)
    
    # Define transforms for crops
    crop_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = QualityAwarePretrainDataset(
        image_paths,
        crop_size=args.crop_size,
        base_size=args.base_size,
        overlap_ratio=args.overlap_ratio,
        transform=crop_transform
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    print(f"  Total batches per epoch: {len(dataloader)}")
    
    # Create loss function
    print("\n[4/6] Creating loss function...")
    if args.loss_type == 'full':
        criterion = QualiCLIPLoss(
            lambda_cons=args.lambda_cons,
            lambda_pos=args.lambda_rank,
            lambda_neg=args.lambda_rank
        )
    else:
        criterion = SimplifiedQualiCLIPLoss(
            lambda_cons=args.lambda_cons,
            lambda_rank=args.lambda_rank
        )
    print(f"  Using {args.loss_type} QualiCLIP loss")
    
    # Create optimizer and scheduler
    print("\n[5/6] Setting up optimization...")
    optimizer = AdamW(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    print(f"  Optimizer: AdamW (lr={args.lr}, weight_decay={args.weight_decay})")
    print(f"  Scheduler: CosineAnnealingLR (T_max={args.epochs})")
    
    # Training loop
    print("\n[6/6] Starting pretraining...")
    print("="*80)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-"*80)
        
        # Train one epoch
        avg_loss, avg_losses = pretrain_epoch(
            encoder, dataloader, criterion, optimizer, device,
            args.distortion_types, args.num_levels,
            text_features_pos, text_features_neg
        )
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Loss Components: {avg_losses}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.epochs:
            checkpoint_path = os.path.join(save_path, f'swin_{args.model_size}_epoch{epoch+1}.pkl')
            # Save only the Swin backbone (not the projection head)
            torch.save(encoder.backbone.state_dict(), checkpoint_path)
            print(f"  ✓ Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(save_path, f'swin_{args.model_size}_qualiclip_pretrained.pkl')
    torch.save(encoder.backbone.state_dict(), final_path)
    print(f"\n{'='*80}")
    print(f"Pretraining complete!")
    print(f"Final model saved to: {final_path}")
    print(f"{'='*80}")
    
    return final_path


if __name__ == '__main__':
    main()

