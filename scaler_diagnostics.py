import torch
import pickle
from sklearn.preprocessing import StandardScaler

def diagnose_model_scalers(model_path='trained_attribution_model.pth'):
    """Diagnose scaler issues in saved model"""
    
    print("🔍 Diagnosing model scalers...")
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        print("✅ Model file loaded successfully")
        print(f"📋 Available keys: {list(checkpoint.keys())}")
        
        # Check scalers
        if 'scaler_features' in checkpoint:
            scaler_features = checkpoint['scaler_features']
            print(f"\n📊 Feature Scaler:")
            print(f"   Type: {type(scaler_features)}")
            
            # Check if fitted
            try:
                if hasattr(scaler_features, 'mean_'):
                    print(f"   ✅ Fitted: YES")
                    print(f"   Mean shape: {scaler_features.mean_.shape}")
                    print(f"   Scale shape: {scaler_features.scale_.shape}")
                    print(f"   Feature means: {scaler_features.mean_[:5]}...")  # First 5
                    print(f"   Feature scales: {scaler_features.scale_[:5]}...")  # First 5
                else:
                    print(f"   ❌ Fitted: NO - This is the problem!")
            except Exception as e:
                print(f"   ❌ Error checking scaler: {e}")
        else:
            print("❌ scaler_features not found in checkpoint")
        
        if 'scaler_targets' in checkpoint:
            scaler_targets = checkpoint['scaler_targets']
            print(f"\n🎯 Target Scaler:")
            print(f"   Type: {type(scaler_targets)}")
            
            try:
                if hasattr(scaler_targets, 'mean_'):
                    print(f"   ✅ Fitted: YES")
                    print(f"   Mean shape: {scaler_targets.mean_.shape}")
                    print(f"   Scale shape: {scaler_targets.scale_.shape}")
                    print(f"   Target means: {scaler_targets.mean_}")
                    print(f"   Target scales: {scaler_targets.scale_}")
                else:
                    print(f"   ❌ Fitted: NO - This is the problem!")
            except Exception as e:
                print(f"   ❌ Error checking target scaler: {e}")
        else:
            print("❌ scaler_targets not found in checkpoint")
        
        # Check feature columns
        if 'feature_columns' in checkpoint:
            print(f"\n📋 Feature Columns ({len(checkpoint['feature_columns'])}):")
            print(f"   {checkpoint['feature_columns']}")
        
        # Check target columns  
        if 'target_columns' in checkpoint:
            print(f"\n🎯 Target Columns ({len(checkpoint['target_columns'])}):")
            print(f"   {checkpoint['target_columns']}")
            
        return checkpoint
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def create_emergency_fitted_scalers():
    """Create properly fitted scalers as emergency fix"""
    
    print("\n🚑 Creating emergency fitted scalers...")
    
    from portfolio_attribution_model import PortfolioAttributionEngine
    import numpy as np
    
    # Create temporary attribution engine to get scalers
    temp_engine = PortfolioAttributionEngine()
    
    # Generate some data to fit scalers
    print("📊 Generating synthetic data to fit scalers...")
    features, targets = temp_engine.generate_synthetic_data(1000)
    
    # Fit the scalers
    temp_engine.scaler_features.fit(features)
    temp_engine.scaler_targets.fit(targets)
    
    print(f"✅ Feature scaler fitted on {features.shape} samples")
    print(f"✅ Target scaler fitted on {targets.shape} samples")
    
    # Show scaler parameters
    print(f"📊 Feature scaler means: {temp_engine.scaler_features.mean_[:5]}...")
    print(f"📊 Feature scaler scales: {temp_engine.scaler_features.scale_[:5]}...")
    print(f"🎯 Target scaler means: {temp_engine.scaler_targets.mean_}")
    print(f"🎯 Target scaler scales: {temp_engine.scaler_targets.scale_}")
    
    return temp_engine.scaler_features, temp_engine.scaler_targets, features.columns.tolist()

def fix_model_scalers():
    """Fix the model by adding properly fitted scalers"""
    
    print("🔧 Fixing model scalers...")
    
    # Load the broken model
    checkpoint = diagnose_model_scalers()
    if not checkpoint:
        print("❌ Could not load model")
        return False
    
    # Create fitted scalers
    fitted_scaler_features, fitted_scaler_targets, feature_columns = create_emergency_fitted_scalers()
    
    # Update the checkpoint
    checkpoint['scaler_features'] = fitted_scaler_features
    checkpoint['scaler_targets'] = fitted_scaler_targets
    
    # Make sure feature columns match
    if 'feature_columns' not in checkpoint:
        checkpoint['feature_columns'] = feature_columns
    
    # Save the fixed model
    fixed_path = 'trained_attribution_model_fixed_scalers.pth'
    torch.save(checkpoint, fixed_path)
    
    print(f"✅ Fixed model saved as: {fixed_path}")
    
    # Verify the fix
    print("\n🔍 Verifying the fix...")
    test_checkpoint = torch.load(fixed_path, map_location='cpu', weights_only=False)
    
    if hasattr(test_checkpoint['scaler_features'], 'mean_'):
        print("✅ Feature scaler is now fitted")
    if hasattr(test_checkpoint['scaler_targets'], 'mean_'):
        print("✅ Target scaler is now fitted")
    
    return True

if __name__ == "__main__":
    print("🩺 Model Scaler Diagnostic Tool")
    print("=" * 40)
    
    # First diagnose
    checkpoint = diagnose_model_scalers()
    
    if checkpoint:
        # Check if scalers are missing or not fitted
        needs_fix = False
        
        if 'scaler_features' not in checkpoint:
            needs_fix = True
            print("\n❌ Missing feature scaler")
        elif not hasattr(checkpoint['scaler_features'], 'mean_'):
            needs_fix = True
            print("\n❌ Feature scaler not fitted")
            
        if 'scaler_targets' not in checkpoint:
            needs_fix = True
            print("\n❌ Missing target scaler")
        elif not hasattr(checkpoint['scaler_targets'], 'mean_'):
            needs_fix = True
            print("\n❌ Target scaler not fitted")
        
        if needs_fix:
            print("\n💡 Running automatic fix...")
            if fix_model_scalers():
                print("\n🎉 Model fixed! Update your frontend to use:")
                print("   'trained_attribution_model_fixed_scalers.pth'")
            else:
                print("\n❌ Could not fix automatically")
        else:
            print("\n✅ Model scalers look good!")