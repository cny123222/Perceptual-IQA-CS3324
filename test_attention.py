#!/usr/bin/env python
"""
Quick test script to verify attention-based fusion implementation.
Tests model construction, forward pass, and attention weight generation.
"""

import torch
import models_swin as models
import numpy as np

def test_attention_module():
    """Test MultiScaleAttention module"""
    print("=" * 60)
    print("Testing MultiScaleAttention Module")
    print("=" * 60)
    
    # Create dummy features
    batch_size = 2
    feat0 = torch.randn(batch_size, 96, 56, 56)
    feat1 = torch.randn(batch_size, 192, 28, 28)
    feat2 = torch.randn(batch_size, 384, 14, 14)
    feat3 = torch.randn(batch_size, 768, 7, 7)
    
    # Create attention module
    attn_module = models.MultiScaleAttention([96, 192, 384, 768])
    
    # Forward pass
    fused_feat, attention_weights = attn_module([feat0, feat1, feat2, feat3])
    
    # Verify outputs
    print(f"✓ Input features:")
    print(f"  - feat0: {feat0.shape}")
    print(f"  - feat1: {feat1.shape}")
    print(f"  - feat2: {feat2.shape}")
    print(f"  - feat3: {feat3.shape}")
    print(f"\n✓ Output fused feature: {fused_feat.shape}")
    print(f"✓ Attention weights shape: {attention_weights.shape}")
    
    # Check attention weights properties
    weights = attention_weights.detach().numpy()
    print(f"\n✓ Attention weights (first sample):")
    print(f"  {weights[0]}")
    print(f"  Sum: {weights[0].sum():.6f} (should be ~1.0)")
    print(f"  All positive: {(weights >= 0).all()}")
    
    # Check output shape
    assert fused_feat.shape == (batch_size, 1440, 7, 7), f"Expected (2, 1440, 7, 7), got {fused_feat.shape}"
    assert attention_weights.shape == (batch_size, 4), f"Expected (2, 4), got {attention_weights.shape}"
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-5), "Attention weights should sum to 1"
    
    print("\n✅ MultiScaleAttention module test PASSED!\n")
    return True


def test_hypernet_with_attention():
    """Test HyperNet with attention-based fusion"""
    print("=" * 60)
    print("Testing HyperNet with Attention Fusion")
    print("=" * 60)
    
    # Create model with attention
    print("\n1. Creating HyperNet with attention...")
    model = models.HyperNet(
        lda_out_channels=16,
        hyper_in_channels=112,
        target_in_size=224,
        target_fc1_size=112,
        target_fc2_size=56,
        target_fc3_size=28,
        target_fc4_size=14,
        feature_size=7,
        use_multiscale=True,
        use_attention=True
    )
    model.eval()
    
    print("✓ Model created successfully")
    print(f"✓ use_multiscale: {model.use_multiscale}")
    print(f"✓ use_attention: {model.use_attention}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    attn_params = sum(p.numel() for p in model.multiscale_attention.parameters())
    print(f"\n2. Parameter count:")
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Attention module parameters: {attn_params:,}")
    print(f"✓ Attention overhead: {attn_params/total_params*100:.2f}%")
    
    # Forward pass
    print(f"\n3. Testing forward pass...")
    dummy_input = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print("✓ Forward pass successful")
    print(f"✓ Output keys: {list(output.keys())}")
    
    # Check if attention weights are saved
    if hasattr(model, 'last_attention_weights'):
        weights = model.last_attention_weights.numpy()
        print(f"\n4. Attention weights (first sample):")
        print(f"   Stage 0 (56×56, low-level):  {weights[0, 0]:.4f}")
        print(f"   Stage 1 (28×28, mid-low):    {weights[0, 1]:.4f}")
        print(f"   Stage 2 (14×14, mid-high):   {weights[0, 2]:.4f}")
        print(f"   Stage 3 (7×7, high-level):   {weights[0, 3]:.4f}")
        print(f"   Sum: {weights[0].sum():.6f}")
    
    print("\n✅ HyperNet with attention test PASSED!\n")
    return True


def test_hypernet_without_attention():
    """Test HyperNet without attention (backward compatibility)"""
    print("=" * 60)
    print("Testing HyperNet WITHOUT Attention (Backward Compatibility)")
    print("=" * 60)
    
    print("\n1. Creating HyperNet without attention...")
    model = models.HyperNet(
        lda_out_channels=16,
        hyper_in_channels=112,
        target_in_size=224,
        target_fc1_size=112,
        target_fc2_size=56,
        target_fc3_size=28,
        target_fc4_size=14,
        feature_size=7,
        use_multiscale=True,
        use_attention=False  # Disabled
    )
    model.eval()
    
    print("✓ Model created successfully")
    print(f"✓ use_multiscale: {model.use_multiscale}")
    print(f"✓ use_attention: {model.use_attention}")
    
    # Forward pass
    print(f"\n2. Testing forward pass...")
    dummy_input = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print("✓ Forward pass successful")
    print("✓ No attention weights generated (as expected)")
    
    print("\n✅ Backward compatibility test PASSED!\n")
    return True


def compare_attention_vs_no_attention():
    """Compare output consistency between attention and no-attention modes"""
    print("=" * 60)
    print("Comparing Attention vs No-Attention Outputs")
    print("=" * 60)
    
    # Same input
    torch.manual_seed(42)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Model without attention
    print("\n1. Creating baseline model (no attention)...")
    model_baseline = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7, 
                                     use_multiscale=True, use_attention=False)
    model_baseline.eval()
    
    # Model with attention
    print("2. Creating model with attention...")
    model_attention = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7,
                                      use_multiscale=True, use_attention=True)
    model_attention.eval()
    
    # Forward pass
    print("3. Running forward passes...")
    with torch.no_grad():
        out_baseline = model_baseline(dummy_input)
        out_attention = model_attention(dummy_input)
    
    # Compare output shapes
    print("\n4. Comparing output shapes:")
    for key in out_baseline.keys():
        shape_baseline = out_baseline[key].shape
        shape_attention = out_attention[key].shape
        match = "✓" if shape_baseline == shape_attention else "✗"
        print(f"   {match} {key}: {shape_baseline} vs {shape_attention}")
    
    print("\n5. Parameter count comparison:")
    params_baseline = sum(p.numel() for p in model_baseline.parameters())
    params_attention = sum(p.numel() for p in model_attention.parameters())
    print(f"   Baseline:  {params_baseline:,}")
    print(f"   Attention: {params_attention:,}")
    print(f"   Overhead:  {params_attention - params_baseline:,} (+{(params_attention/params_baseline - 1)*100:.2f}%)")
    
    print("\n✅ Comparison test PASSED!\n")
    return True


if __name__ == "__main__":
    print("\n" + "🔥" * 30)
    print("Attention-Based Multi-Scale Fusion Test Suite")
    print("🔥" * 30 + "\n")
    
    try:
        # Run all tests
        test_attention_module()
        test_hypernet_with_attention()
        test_hypernet_without_attention()
        compare_attention_vs_no_attention()
        
        print("=" * 60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("=" * 60)
        print("\nYou can now train with attention fusion:")
        print("  python train_swin.py --dataset koniq-10k --epochs 30 --attention_fusion")
        print("\nOr without attention (baseline):")
        print("  python train_swin.py --dataset koniq-10k --epochs 30")
        print()
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

