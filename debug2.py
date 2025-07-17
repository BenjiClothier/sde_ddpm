import torch
import torch.nn.functional as F
from unet import Unet  # Your UNet import

def debug_model_loading(device):
    """Debug and compare different model loading approaches"""
    
    print("=== Debugging Model Loading ===\n")
    
    regular_model = None
    ema_model_wrapper = None
    ema_model_direct = None
    test_input = torch.randn(1, 3, 32, 32).to(device)
    
    # Method 1: Load regular model
    print("1. Loading regular model...")
    try:
        regular_model = torch.load('./unet.pt', map_location=device, weights_only=False)
        regular_model.eval()
        print("✓ Regular model loaded successfully")
        
        # Test regular model
        with torch.no_grad():
            regular_output = regular_model(test_input, torch.randint(0, 1000, (1,)).to(device))
            print(f"   Regular model output range: [{regular_output.min():.4f}, {regular_output.max():.4f}]")
            print(f"   Regular model output mean: {regular_output.mean():.4f}")
    except Exception as e:
        print(f"❌ Failed to load regular model: {e}")
        print(f"   Error details: {type(e).__name__}: {str(e)}")
    
    # Method 2: Load EMA with wrapper (your current approach)
    print("\n2. Loading EMA model with wrapper...")
    try:
        base_model = Unet().to(device)
        ema_model_wrapper = torch.optim.swa_utils.AveragedModel(
            base_model, 
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999)
        )
        ema_state_dict = torch.load('./ema_sd.pt', map_location=device)
        ema_model_wrapper.load_state_dict(ema_state_dict)
        ema_model_wrapper.eval()
        print("✓ EMA wrapper model loaded successfully")
        
        # Test EMA wrapper
        with torch.no_grad():
            ema_wrapper_output = ema_model_wrapper.module(test_input, torch.randint(0, 1000, (1,)).to(device))
            print(f"   EMA wrapper output range: [{ema_wrapper_output.min():.4f}, {ema_wrapper_output.max():.4f}]")
            print(f"   EMA wrapper output mean: {ema_wrapper_output.mean():.4f}")
    except Exception as e:
        print(f"❌ Failed to load EMA wrapper: {e}")
        ema_model_wrapper = None
    
    # Method 3: Load EMA weights directly into UNet (alternative approach)
    print("\n3. Loading EMA weights directly into UNet...")
    try:
        ema_model_direct = Unet().to(device)
        ema_state_dict = torch.load('./ema_sd.pt', map_location=device)
        
        # Remove 'module.' prefix and 'n_averaged' key
        cleaned_state_dict = {}
        for key, value in ema_state_dict.items():
            if key.startswith('module.'):
                cleaned_state_dict[key[7:]] = value  # Remove 'module.' prefix
            elif key != 'n_averaged':  # Skip the n_averaged key
                cleaned_state_dict[key] = value
        
        ema_model_direct.load_state_dict(cleaned_state_dict)
        ema_model_direct.eval()
        print("✓ EMA direct model loaded successfully")
        
        # Test EMA direct
        with torch.no_grad():
            ema_direct_output = ema_model_direct(test_input, torch.randint(0, 1000, (1,)).to(device))
            print(f"   EMA direct output range: [{ema_direct_output.min():.4f}, {ema_direct_output.max():.4f}]")
            print(f"   EMA direct output mean: {ema_direct_output.mean():.4f}")
    except Exception as e:
        print(f"❌ Failed to load EMA direct: {e}")
        ema_model_direct = None
    
    # Compare outputs
    print("\n=== Model Comparison ===")
    if ema_model_wrapper and ema_model_direct:
        with torch.no_grad():
            t_test = torch.randint(0, 1000, (1,)).to(device)
            wrapper_out = ema_model_wrapper.module(test_input, t_test)
            direct_out = ema_model_direct(test_input, t_test)
            diff = torch.abs(wrapper_out - direct_out).mean()
            print(f"Difference between wrapper and direct EMA: {diff:.6f}")
            if diff < 1e-5:
                print("✓ Both EMA loading methods give identical results")
            else:
                print("⚠ EMA loading methods give different results")
    
    return regular_model, ema_model_wrapper, ema_model_direct

def diagnose_model_state(model, model_name):
    """Diagnose if model is in correct state"""
    
    print(f"\n=== Diagnosing {model_name} ===")
    
    # Check if model is in eval mode
    if hasattr(model, 'training'):
        print(f"Training mode: {model.training}")
        if model.training:
            print("⚠ Model is in training mode - should be in eval mode for inference")
    
    # Check parameter statistics
    total_params = 0
    zero_params = 0
    
    if hasattr(model, 'parameters'):
        params_to_check = model.parameters()
    elif hasattr(model, 'module') and hasattr(model.module, 'parameters'):
        params_to_check = model.module.parameters()
    else:
        print("❌ Cannot access model parameters")
        return
    
    for param in params_to_check:
        total_params += param.numel()
        zero_params += (param.abs() < 1e-8).sum().item()
    
    print(f"Total parameters: {total_params:,}")
    print(f"Near-zero parameters: {zero_params:,} ({100*zero_params/total_params:.2f}%)")
    
    if zero_params / total_params > 0.5:
        print("⚠ More than 50% of parameters are near zero - model might not be loaded correctly")
    else:
        print("✓ Parameter distribution looks normal")

def test_sampling_pipeline(model, device, model_name="model"):
    """Test the sampling pipeline with different models"""
    
    print(f"\n=== Testing Sampling Pipeline with {model_name} ===")
    
    # Simple test: just run model forward pass with timestep
    test_input = torch.randn(1, 3, 32, 32).to(device)
    test_timestep = torch.randint(0, 1000, (1,)).to(device)
    
    # Handle both regular model and EMA wrapper
    if hasattr(model, 'module'):
        actual_model = model.module
        print(f"Using wrapped model (.module)")
    else:
        actual_model = model
        print(f"Using direct model")
    
    actual_model.eval()
    
    with torch.no_grad():
        try:
            # Try with timestep first (diffusion UNet)
            output = actual_model(test_input, test_timestep)
            print(f"✓ Forward pass successful (with timestep)")
            print(f"   Input shape: {test_input.shape}")
            print(f"   Timestep: {test_timestep.item()}")
            print(f"   Output shape: {output.shape}")
            print(f"   Output range: [{output.min():.4f}, {output.max():.4f}]")
            print(f"   Output std: {output.std():.4f}")
            
            # Check if output looks reasonable
            if output.std() < 0.01:
                print("⚠ Output has very low variance - might be all zeros/constant")
            elif output.std() > 10:
                print("⚠ Output has very high variance - might be random noise")
            else:
                print("✓ Output variance looks reasonable")
                
            return True
            
        except Exception as e:
            print(f"❌ Forward pass with timestep failed: {e}")
            
            # Try without timestep as fallback
            try:
                output = actual_model(test_input)
                print(f"✓ Forward pass successful (without timestep)")
                print(f"   Output shape: {output.shape}")
                return True
            except Exception as e2:
                print(f"❌ Forward pass without timestep also failed: {e2}")
                return False

def compare_with_random_model(device):
    """Compare your models with a random UNet to see the difference"""
    
    print("\n=== Comparing with Random Model ===")
    
    # Create random UNet
    try:
        from unet import Unet
        random_model = Unet().to(device)
        random_model.eval()
        
        test_input = torch.randn(1, 3, 32, 32).to(device)
        test_timestep = torch.randint(0, 1000, (1,)).to(device)
        
        with torch.no_grad():
            random_output = random_model(test_input, test_timestep)
            print(f"Random UNet output range: [{random_output.min():.4f}, {random_output.max():.4f}]")
            print(f"Random UNet output std: {random_output.std():.4f}")
            
        return random_output
    except Exception as e:
        print(f"❌ Random model test failed: {e}")
        return None

def main_debug():
    """Main debugging function"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load all models
    regular_model, ema_wrapper, ema_direct = debug_model_loading(device)
    
    # Diagnose each model
    if regular_model:
        diagnose_model_state(regular_model, "Regular Model")
        test_sampling_pipeline(regular_model, device, "Regular Model")
    
    if ema_wrapper:
        diagnose_model_state(ema_wrapper, "EMA Wrapper")
        test_sampling_pipeline(ema_wrapper, device, "EMA Wrapper")
    
    if ema_direct:
        diagnose_model_state(ema_direct, "EMA Direct")
        test_sampling_pipeline(ema_direct, device, "EMA Direct")
    
    # Compare with random
    compare_with_random_model(device)
    
    # Recommendations
    print("\n=== Recommendations ===")
    if regular_model and ema_direct:
        print("✓ Try using the 'EMA Direct' approach - it's simpler and less error-prone")
        print("✓ If regular model works but EMA doesn't, there might be an issue with EMA weights")
    
    print("\nNext steps:")
    print("1. If all models output noise, check your UNet architecture")
    print("2. If only EMA outputs noise, try using regular model for now")
    print("3. Check your sampling/inference code (timesteps, noise schedule, etc.)")
    print("4. Verify the model was trained properly (check training logs)")

# Quick fix suggestion
def quick_fix_suggestion():
    """Suggest quick fixes"""
    
    fix_code = '''
# QUICK FIX: Try this loading approach for diffusion UNet
def load_best_model(device):
    """Load the model that's most likely to work"""
    
    # Option 1: Try regular model first
    try:
        model = torch.load('./unet.pt', map_location=device, weights_only=False)
        model.eval()
        print("Using regular model")
        return model
    except:
        pass
    
    # Option 2: Try EMA direct loading
    try:
        from unet import Unet
        model = Unet().to(device)
        ema_state_dict = torch.load('./ema_sd.pt', map_location=device)
        
        # Clean EMA state dict
        cleaned_state_dict = {}
        for key, value in ema_state_dict.items():
            if key.startswith('module.'):
                cleaned_state_dict[key[7:]] = value
            elif key != 'n_averaged':
                cleaned_state_dict[key] = value
        
        model.load_state_dict(cleaned_state_dict)
        model.eval()
        print("Using EMA model (direct loading)")
        return model
    except Exception as e:
        print(f"Failed to load any model: {e}")
        return None

# Use like this for diffusion UNet:
model = load_best_model(device)
if model:
    # Your sampling code here - REMEMBER TO PASS TIMESTEP!
    x = your_noisy_image
    t = your_timestep_tensor  # Important!
    
    with torch.no_grad():
        predicted_noise = model(x, t)  # Both x and t required!
        
# Example timestep creation:
# For single image: t = torch.tensor([timestep_value]).to(device)
# For batch: t = torch.randint(0, 1000, (batch_size,)).to(device)
'''
    
    print("=== Quick Fix Code ===")
    print(fix_code)

def simple_debug():
    """Simple debugging without fancy features"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\n=== Simple Model Loading Test ===")
    
    # Test 1: Load regular model
    print("\n1. Testing regular model loading...")
    try:
        model = torch.load('./unet.pt', map_location=device, weights_only=False)
        print(f"✓ Loaded model type: {type(model)}")
        model.eval()
        
        # Test forward pass with timestep
        x = torch.randn(1, 3, 32, 32).to(device)
        t = torch.randint(0, 1000, (1,)).to(device)  # Random timestep
        
        with torch.no_grad():
            out = model(x, t)  # Pass both x and t
            print(f"✓ Forward pass successful: {x.shape}, t={t.item()} -> {out.shape}")
            print(f"   Output range: [{out.min():.4f}, {out.max():.4f}]")
            print(f"   Output mean: {out.mean():.4f}, std: {out.std():.4f}")
            
            # Check if output looks like noise (which would be bad)
            if out.std() > 2.0:
                print("⚠ High output variance - might be outputting noise")
            elif out.std() < 0.01:
                print("⚠ Very low output variance - might be outputting zeros")
            else:
                print("✓ Output variance looks reasonable")
                
    except Exception as e:
        print(f"❌ Regular model failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Load EMA direct
    print("\n2. Testing EMA direct loading...")
    try:
        from unet import Unet
        model = Unet().to(device)
        print(f"✓ Created UNet: {type(model)}")
        
        ema_dict = torch.load('./ema_sd.pt', map_location=device)
        print(f"✓ Loaded EMA dict with {len(ema_dict)} keys")
        
        # Show some keys to debug
        sample_keys = list(ema_dict.keys())[:5]
        print(f"   Sample keys: {sample_keys}")
        
        # Clean the dict
        cleaned = {}
        for k, v in ema_dict.items():
            if k.startswith('module.'):
                cleaned[k[7:]] = v
            elif k != 'n_averaged':
                cleaned[k] = v
        
        print(f"✓ Cleaned dict has {len(cleaned)} keys")
        model.load_state_dict(cleaned)
        model.eval()
        
        # Test forward pass with timestep
        x = torch.randn(1, 3, 32, 32).to(device)
        t = torch.randint(0, 1000, (1,)).to(device)  # Random timestep
        
        with torch.no_grad():
            out = model(x, t)  # Pass both x and t
            print(f"✓ EMA forward pass successful: {x.shape}, t={t.item()} -> {out.shape}")
            print(f"   Output range: [{out.min():.4f}, {out.max():.4f}]")
            print(f"   Output mean: {out.mean():.4f}, std: {out.std():.4f}")
            
            # Check output quality
            if out.std() > 2.0:
                print("⚠ High output variance - might be outputting noise")
            elif out.std() < 0.01:
                print("⚠ Very low output variance - might be outputting zeros") 
            else:
                print("✓ Output variance looks reasonable")
            
    except Exception as e:
        print(f"❌ EMA direct failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Compare regular vs EMA
    print("\n3. Comparing regular vs EMA outputs...")
    try:
        # Load both models
        regular_model = torch.load('./unet.pt', map_location=device, weights_only=False)
        regular_model.eval()
        
        ema_model = Unet().to(device)
        ema_dict = torch.load('./ema_sd.pt', map_location=device)
        cleaned = {k[7:] if k.startswith('module.') else k: v 
                  for k, v in ema_dict.items() if k != 'n_averaged'}
        ema_model.load_state_dict(cleaned)
        ema_model.eval()
        
        # Same input for both
        x = torch.randn(1, 3, 32, 32).to(device)
        t = torch.randint(0, 1000, (1,)).to(device)
        
        with torch.no_grad():
            regular_out = regular_model(x, t)
            ema_out = ema_model(x, t)
            
            diff = torch.abs(regular_out - ema_out).mean()
            print(f"✓ Output difference between regular and EMA: {diff:.6f}")
            
            if diff < 1e-6:
                print("⚠ Models produce identical outputs - EMA might not be working")
            elif diff > 1.0:
                print("⚠ Very large difference - one model might be broken")
            else:
                print("✓ Reasonable difference between models")
                
    except Exception as e:
        print(f"❌ Comparison failed: {e}")

if __name__ == "__main__":
    # Run simple debug first
    simple_debug()
    
    # Only run full debug if simple works
    try:
        main_debug()
        quick_fix_suggestion()
    except Exception as e:
        print(f"\nFull debug failed: {e}")
        print("But simple debug results above should help identify the issue.")