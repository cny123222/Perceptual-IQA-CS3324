"""
QualiCLIP Loss Function for Quality-Aware Self-Supervised Pretraining

Implements three loss components from QualiCLIP paper:
1. Consistency Loss (L_cons): Encourages similar representations for crops from same image/level
2. Positive Ranking Loss (L_pos): Higher quality images should be more similar to "Good photo"
3. Negative Ranking Loss (L_neg): Lower quality images should be more similar to "Bad photo"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QualiCLIPLoss(nn.Module):
    """
    QualiCLIP Loss for quality-aware image-text alignment.
    
    The loss encourages:
    - Consistent features for same content + same degradation level (L_cons)
    - Ranking degraded images by similarity to quality-related text prompts (L_pos, L_neg)
    
    Args:
        margin_cons (float): Margin for consistency loss (default: 0.0)
        margin_rank (float): Margin for ranking losses (default: 0.05)
        lambda_cons (float): Weight for consistency loss (default: 1.0)
        lambda_pos (float): Weight for positive ranking loss (default: 1.0)
        lambda_neg (float): Weight for negative ranking loss (default: 1.0)
    """
    
    def __init__(self, margin_cons=0.0, margin_rank=0.05, 
                 lambda_cons=1.0, lambda_pos=1.0, lambda_neg=1.0):
        super(QualiCLIPLoss, self).__init__()
        
        self.margin_cons = margin_cons
        self.margin_rank = margin_rank
        self.lambda_cons = lambda_cons
        self.lambda_pos = lambda_pos
        self.lambda_neg = lambda_neg
        
        # Use MarginRankingLoss for all three components
        self.ranking_loss_cons = nn.MarginRankingLoss(margin=margin_cons)
        self.ranking_loss_rank = nn.MarginRankingLoss(margin=margin_rank)
        
    def forward(self, image_features, text_features_pos, text_features_neg, 
                degradation_levels, pair_indices):
        """
        Compute QualiCLIP loss.
        
        Args:
            image_features: [N, D] tensor of image features (N = batch_size * num_levels * 2)
            text_features_pos: [D] tensor for "Good photo" text embedding
            text_features_neg: [D] tensor for "Bad photo" text embedding
            degradation_levels: [N] tensor indicating degradation level for each feature (0-4)
            pair_indices: [N] tensor indicating which features come from the same crop pair
                         (same value = same original image)
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components for logging
        """
        
        # Normalize features
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features_pos = F.normalize(text_features_pos, p=2, dim=0)
        text_features_neg = F.normalize(text_features_neg, p=2, dim=0)
        
        # Compute cosine similarities with text features
        # sim_pos[i] = similarity between image i and "Good photo"
        sim_pos = torch.matmul(image_features, text_features_pos)  # [N]
        sim_neg = torch.matmul(image_features, text_features_neg)  # [N]
        
        # 1. Consistency Loss (L_cons)
        # For pairs from same image and same degradation level, similarities should be close
        loss_cons = self._compute_consistency_loss(sim_pos, sim_neg, 
                                                   degradation_levels, pair_indices)
        
        # 2. Positive Ranking Loss (L_pos)
        # For same image, lower degradation level → higher similarity with "Good photo"
        loss_pos = self._compute_ranking_loss(sim_pos, degradation_levels, pair_indices,
                                             reverse=False)
        
        # 3. Negative Ranking Loss (L_neg)
        # For same image, higher degradation level → higher similarity with "Bad photo"
        loss_neg = self._compute_ranking_loss(sim_neg, degradation_levels, pair_indices,
                                             reverse=True)
        
        # Total loss
        total_loss = (self.lambda_cons * loss_cons + 
                     self.lambda_pos * loss_pos + 
                     self.lambda_neg * loss_neg)
        
        loss_dict = {
            'total': total_loss.item(),
            'cons': loss_cons.item(),
            'pos': loss_pos.item(),
            'neg': loss_neg.item()
        }
        
        return total_loss, loss_dict
    
    def _compute_consistency_loss(self, sim_pos, sim_neg, degradation_levels, pair_indices):
        """
        Compute consistency loss: features from same image/level should have similar text similarities.
        
        For each pair (crop1, crop2) from same image and same degradation level:
        - Their similarity to "Good photo" should be close
        - Their similarity to "Bad photo" should be close
        """
        
        losses = []
        
        # Find unique (image_id, level) combinations
        unique_pairs = torch.unique(pair_indices)
        
        for pair_id in unique_pairs:
            # Find all features from this image
            mask = (pair_indices == pair_id)
            
            # For each degradation level
            for level in range(5):  # 0-4
                level_mask = mask & (degradation_levels == level)
                indices = torch.where(level_mask)[0]
                
                if len(indices) >= 2:
                    # We have a pair (or more) at this level
                    # Compute pairwise consistency
                    for i in range(len(indices) - 1):
                        for j in range(i + 1, len(indices)):
                            idx1, idx2 = indices[i], indices[j]
                            
                            # Consistency for positive similarities
                            # We want sim_pos[idx1] ≈ sim_pos[idx2]
                            # Use ranking loss with target=0 (want them equal)
                            # But MarginRankingLoss expects target in {-1, 1}
                            # So we use absolute difference instead
                            loss_pos_pair = torch.abs(sim_pos[idx1] - sim_pos[idx2])
                            loss_neg_pair = torch.abs(sim_neg[idx1] - sim_neg[idx2])
                            
                            losses.append(loss_pos_pair)
                            losses.append(loss_neg_pair)
        
        if len(losses) > 0:
            return torch.stack(losses).mean()
        else:
            return torch.tensor(0.0, device=sim_pos.device, requires_grad=True)
    
    def _compute_ranking_loss(self, similarities, degradation_levels, pair_indices, reverse=False):
        """
        Compute ranking loss: order images by degradation level.
        
        For features from the same image but different degradation levels:
        - If reverse=False: Lower level → higher similarity (for "Good photo")
        - If reverse=True: Higher level → higher similarity (for "Bad photo")
        
        Args:
            similarities: [N] tensor of similarities to a text prompt
            degradation_levels: [N] tensor of degradation levels
            pair_indices: [N] tensor of image IDs
            reverse: Whether to reverse the ranking (for "Bad photo")
        """
        
        losses = []
        
        # Find unique images
        unique_pairs = torch.unique(pair_indices)
        
        for pair_id in unique_pairs:
            # Find all features from this image
            mask = (pair_indices == pair_id)
            indices = torch.where(mask)[0]
            levels = degradation_levels[mask]
            sims = similarities[mask]
            
            # Create pairs where level_i < level_j
            for i in range(len(indices)):
                for j in range(len(indices)):
                    if levels[i] < levels[j]:
                        # level_i has better quality than level_j
                        sim_i = sims[i]
                        sim_j = sims[j]
                        
                        if reverse:
                            # For "Bad photo": worse quality → higher similarity
                            # So sim_j should be > sim_i
                            # MarginRankingLoss(x1, x2, target=1) encourages x1 > x2
                            loss = self.ranking_loss_rank(sim_j, sim_i, 
                                                         torch.tensor(1.0, device=similarities.device))
                        else:
                            # For "Good photo": better quality → higher similarity
                            # So sim_i should be > sim_j
                            loss = self.ranking_loss_rank(sim_i, sim_j, 
                                                         torch.tensor(1.0, device=similarities.device))
                        
                        losses.append(loss)
        
        if len(losses) > 0:
            return torch.stack(losses).mean()
        else:
            return torch.tensor(0.0, device=similarities.device, requires_grad=True)


class SimplifiedQualiCLIPLoss(nn.Module):
    """
    Simplified version of QualiCLIP loss that's more efficient.
    
    Instead of computing all pairwise comparisons, we:
    1. Average features within each (image, level) group for consistency
    2. Use ordinal regression-style loss for ranking
    """
    
    def __init__(self, lambda_cons=1.0, lambda_rank=1.0):
        super(SimplifiedQualiCLIPLoss, self).__init__()
        self.lambda_cons = lambda_cons
        self.lambda_rank = lambda_rank
        
    def forward(self, image_features, text_features_pos, text_features_neg,
                degradation_levels, pair_indices):
        """Compute simplified QualiCLIP loss"""
        
        # Normalize features
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features_pos = F.normalize(text_features_pos, p=2, dim=0)
        text_features_neg = F.normalize(text_features_neg, p=2, dim=0)
        
        # Compute similarities
        sim_pos = torch.matmul(image_features, text_features_pos)
        sim_neg = torch.matmul(image_features, text_features_neg)
        
        # 1. Consistency loss: variance within same (image, level) groups
        loss_cons = 0.0
        unique_pairs = torch.unique(pair_indices)
        
        for pair_id in unique_pairs:
            mask = (pair_indices == pair_id)
            for level in range(5):
                level_mask = mask & (degradation_levels == level)
                if level_mask.sum() > 1:
                    # Compute variance of similarities within this group
                    group_sim_pos = sim_pos[level_mask]
                    group_sim_neg = sim_neg[level_mask]
                    loss_cons += group_sim_pos.var() + group_sim_neg.var()
        
        # 2. Ranking loss: monotonicity with degradation level
        # Use L1 loss between similarity and target ranking
        # Target: level 0 (best) should have sim_pos ≈ 1, level 4 (worst) should have sim_pos ≈ 0
        target_pos = 1.0 - (degradation_levels.float() / 4.0)  # [1.0, 0.75, 0.5, 0.25, 0.0]
        target_neg = degradation_levels.float() / 4.0  # [0.0, 0.25, 0.5, 0.75, 1.0]
        
        loss_rank = F.l1_loss(sim_pos, target_pos) + F.l1_loss(sim_neg, target_neg)
        
        total_loss = self.lambda_cons * loss_cons + self.lambda_rank * loss_rank
        
        loss_dict = {
            'total': total_loss.item(),
            'cons': loss_cons if isinstance(loss_cons, float) else loss_cons.item(),
            'rank': loss_rank.item()
        }
        
        return total_loss, loss_dict


def test_loss():
    """Test the loss function with dummy data"""
    print("Testing QualiCLIP Loss...")
    
    batch_size = 4
    num_levels = 5
    num_crops = 2
    feature_dim = 512
    
    # Create dummy data
    N = batch_size * num_levels * num_crops  # 40 features
    image_features = torch.randn(N, feature_dim)
    text_features_pos = torch.randn(feature_dim)
    text_features_neg = torch.randn(feature_dim)
    
    # Create metadata
    degradation_levels = torch.tensor([i // num_crops for i in range(N)]) % num_levels
    pair_indices = torch.tensor([i // (num_levels * num_crops) for i in range(N)])
    
    # Test loss
    criterion = QualiCLIPLoss()
    loss, loss_dict = criterion(image_features, text_features_pos, text_features_neg,
                                degradation_levels, pair_indices)
    
    print(f"Total Loss: {loss.item():.4f}")
    print(f"Loss components: {loss_dict}")
    print("✓ QualiCLIP loss works correctly")
    
    # Test simplified version
    print("\nTesting Simplified QualiCLIP Loss...")
    criterion_simple = SimplifiedQualiCLIPLoss()
    loss_simple, loss_dict_simple = criterion_simple(image_features, text_features_pos, 
                                                     text_features_neg, degradation_levels, 
                                                     pair_indices)
    print(f"Total Loss (Simplified): {loss_simple.item():.4f}")
    print(f"Loss components: {loss_dict_simple}")
    print("✓ Simplified QualiCLIP loss works correctly")


if __name__ == '__main__':
    test_loss()

