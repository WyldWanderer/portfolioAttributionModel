import torch
import os
import pickle

def diagnose_model_file(model_path='trained_attribution_model.pth'):
    """Diagnose what's wrong with the model file"""
    
    print(f"🔍 Diagnosing model file: {model_path}")
    
    # Check if file exists
    if not os.path.exists(model_path):
        print(f"❌ File {model_path} does not exist!")
        return False
    
    # Check file size
    file_size = os.path.getsize(model_path)
    print(f"📁 File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    
    if file_size < 1000:  # Less than 1KB is suspicious
        print("⚠️ File size is very small - possibly corrupted")
    
    # Try different loading methods
    print("\n🧪 Testing different loading approaches...")
    
    # Method 1: Standard torch.load
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print("✅ Standard torch.load: SUCCESS")
        print(f"   Keys: {list(checkpoint.keys())}")
        return checkpoint
    except Exception as e:
        print(f"❌ Standard torch.load failed: {e}")
    
    # Method 2: Weights only
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        print("✅ Weights-only torch.load: SUCCESS")
        return checkpoint
    except Exception as e:
        print(f"❌ Weights-only torch.load failed: {e}")
    
    # Method 3: Pickle directly
    try:
        with open(model_path, 'rb') as f:
            checkpoint = pickle.load(f)
        print("✅ Direct pickle load: SUCCESS")
        return checkpoint
    except Exception as e:
        print(f"❌ Direct pickle failed: {e}")
    
    # Method 4: Check if it's a state_dict only
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        if isinstance(state_dict, dict) and 'model_state_dict' not in state_dict:
            print("⚠️ This appears to be a raw state_dict, not a full checkpoint")
            # Wrap it in expected format
            wrapped = {
                'model_state_dict': state_dict,
                'feature_columns': [],  # Will need to be filled
                'target_columns': ['asset_selection', 'allocation', 'timing', 'currency', 'interaction']
            }
            return wrapped
    except Exception as e:
        print(f"❌ State dict interpretation failed: {e}")
    
    print("❌ All loading methods failed - file is corrupted")
    return None

def create_emergency_model():
    """Create a working model if the saved one is broken"""
    print("\n🚑 Creating emergency replacement model...")
    
    from portfolio_attribution_model import PortfolioAttributionEngine
    
    # Create new model with same architecture as retrained
    attribution_engine = PortfolioAttributionEngine({
        'hidden_dims': [32, 16],
        'dropout_rate': 0.5,
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 100
    })
    
    # Quick training
    print("🏃‍♂️ Quick training replacement model...")
    features, targets = attribution_engine.generate_synthetic_data(1000)
    train_loader, test_loader = attribution_engine.prepare_data(features, targets)
    attribution_engine.train_model(train_loader, test_loader)
    
    # Save it properly
    torch.save({
        'model_state_dict': attribution_engine.model.state_dict(),
        'model_params': attribution_engine.model_params,
        'scaler_features': attribution_engine.scaler_features,
        'scaler_targets': attribution_engine.scaler_targets,
        'feature_columns': attribution_engine.feature_columns,
        'target_columns': attribution_engine.target_columns,
        'input_dim': len(attribution_engine.feature_columns)
    }, 'trained_attribution_model_emergency.pth')
    
    print("✅ Emergency model saved as 'trained_attribution_model_emergency.pth'")
    return True

def fix_model_loading_issue():
    """Main function to fix the loading issue"""
    
    print("🔧 Fixing model loading issue...\n")
    
    # First, diagnose the current file
    checkpoint = diagnose_model_file()
    
    if checkpoint is None:
        print("\n💡 Current model file is corrupted. Creating replacement...")
        create_emergency_model()
        print("\n📋 Update your frontend to use: 'trained_attribution_model_emergency.pth'")
        return False
    else:
        print("\n✅ Model file is readable!")
        
        # Check what's missing
        required_keys = ['model_state_dict', 'feature_columns', 'target_columns', 'scaler_features', 'scaler_targets']
        missing_keys = [key for key in required_keys if key not in checkpoint]
        
        if missing_keys:
            print(f"⚠️ Missing keys: {missing_keys}")
            print("🔧 Attempting to fix...")
            
            # Add default values for missing keys
            if 'feature_columns' not in checkpoint:
                checkpoint['feature_columns'] = [
                    'portfolio_return', 'benchmark_return', 'excess_return',
                    'n_assets', 'max_weight', 'weight_concentration'
                ]
            
            if 'target_columns' not in checkpoint:
                checkpoint['target_columns'] = ['asset_selection', 'allocation', 'timing', 'currency', 'interaction']
            
            if 'model_params' not in checkpoint:
                checkpoint['model_params'] = {
                    'hidden_dims': [32, 16],
                    'dropout_rate': 0.5
                }
            
            # Save the fixed version
            torch.save(checkpoint, 'trained_attribution_model_fixed.pth')
            print("✅ Fixed model saved as 'trained_attribution_model_fixed.pth'")
        
        return True

if __name__ == "__main__":
    success = fix_model_loading_issue()
    
    if success:
        print("\n🎉 Model loading should now work!")
    else:
        print("\n💡 Use the emergency model or retrain if needed")