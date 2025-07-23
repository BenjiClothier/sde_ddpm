import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

class SingleDigitMNIST(Dataset):
    """Custom Dataset class for single digit MNIST"""
    
    def __init__(self, root='./data', train=True, target_digit=8, transform=None):
        """
        Create a MNIST dataset containing only samples of a specific digit.
        
        Parameters:
        - root: Root directory for dataset
        - train: Whether to use training or test set
        - target_digit: The specific digit to include (0-9)
        - transform: PyTorch transforms to apply
        """
        
        self.target_digit = target_digit
        self.transform = transform
        
        # Load original MNIST dataset
        original_dataset = MNIST(root=root, train=train, download=True, transform=None)
        
        # Extract data and labels
        if hasattr(original_dataset, 'data'):
            data = original_dataset.data.numpy()
            targets = original_dataset.targets.numpy()
        else:
            # Handle different PyTorch versions
            data = original_dataset.train_data.numpy() if train else original_dataset.test_data.numpy()
            targets = original_dataset.train_labels.numpy() if train else original_dataset.test_labels.numpy()
        
        # Filter for specific digit
        digit_indices = np.where(targets == target_digit)[0]
        self.data = data[digit_indices]
        self.targets = targets[digit_indices]
        
        print(f"{'Training' if train else 'Test'} set created with {len(self.data)} samples of digit {target_digit}")
        
        # Store class names for compatibility
        self.classes = [str(i) for i in range(10)]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]
        
        # Convert to PIL Image if transform expects it
        if self.transform:
            sample = transforms.ToPILImage()(sample)
            sample = self.transform(sample)
        else:
            # Convert to tensor and normalize
            sample = torch.from_numpy(sample).float() / 255.0
            sample = sample.unsqueeze(0)  # Add channel dimension
        
        return sample, target

class AnomalousMNIST(Dataset):
    """Custom Dataset class for anomalous MNIST"""
    
    def __init__(self, root='./data', train=True, target_digit=8, removal_percentage=0.98, 
                 transform=None, random_state=42):
        """
        Create an anomalous MNIST dataset by removing a specified percentage of a target digit.
        
        Parameters:
        - root: Root directory for dataset
        - train: Whether to use training or test set
        - target_digit: The digit to make anomalous (default: 8)
        - removal_percentage: Percentage of target digit samples to remove (default: 0.98)
        - transform: PyTorch transforms to apply
        - random_state: Random seed for reproducibility
        """
        
        self.target_digit = target_digit
        self.removal_percentage = removal_percentage
        self.transform = transform
        
        # Load original MNIST dataset
        original_dataset = MNIST(root=root, train=train, download=True, transform=None)
        
        # Extract data and labels
        if hasattr(original_dataset, 'data'):
            self.data = original_dataset.data.numpy()
            self.targets = original_dataset.targets.numpy()
        else:
            # Handle different PyTorch versions
            self.data = original_dataset.train_data.numpy() if train else original_dataset.test_data.numpy()
            self.targets = original_dataset.train_labels.numpy() if train else original_dataset.test_labels.numpy()
        
        # Apply anomaly modification
        self._create_anomalous_dataset(random_state)
        
        print(f"{'Training' if train else 'Test'} set created with {len(self.data)} samples")
        print(f"Remaining digit {target_digit}: {np.sum(self.targets == target_digit)}")
        
        # Store class names for compatibility
        self.classes = [str(i) for i in range(10)]
    
    def _create_anomalous_dataset(self, random_state):
        """Remove specified percentage of target digit samples"""
        np.random.seed(random_state)
        
        # Find indices of target digit
        target_indices = np.where(self.targets == self.target_digit)[0]
        original_count = len(target_indices)
        
        # Calculate how many samples to keep
        samples_to_keep = int(original_count * (1 - self.removal_percentage))
        
        # Randomly select indices to keep
        if samples_to_keep > 0:
            indices_to_keep = np.random.choice(
                target_indices, 
                size=samples_to_keep, 
                replace=False
            )
        else:
            indices_to_keep = np.array([])
        
        # Create boolean mask for samples to keep
        keep_mask = np.ones(len(self.targets), dtype=bool)
        indices_to_remove = np.setdiff1d(target_indices, indices_to_keep)
        keep_mask[indices_to_remove] = False
        
        # Apply mask
        self.data = self.data[keep_mask]
        self.targets = self.targets[keep_mask]
        
        print(f"Removed {len(indices_to_remove)} samples of digit {self.target_digit}")
        print(f"Kept {samples_to_keep} samples of digit {self.target_digit}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]
        
        # Convert to PIL Image if transform expects it
        if self.transform:
            sample = transforms.ToPILImage()(sample)
            sample = self.transform(sample)
        else:
            # Convert to tensor and normalize
            sample = torch.from_numpy(sample).float() / 255.0
            sample = sample.unsqueeze(0)  # Add channel dimension
        
        return sample, target

def get_dataset(config, uniform_dequantization=False, evaluation=False):
    """Create data loaders for training and evaluation.

    Args:
        config: A configuration dictionary with keys like:
            - training.batch_size or batch_size: batch size for training
            - eval.batch_size: batch size for evaluation  
            - data.image_size or image_size: target image size
            - data.random_flip or random_flip: whether to apply random horizontal flip
            - data.dataset or dataset: dataset name ('CIFAR10', 'MNIST', 'ANOMALOUS_MNIST', 'SINGLE_DIGIT_MNIST')
            - data.target_digit: digit to make anomalous or specific digit to include
            - data.removal_percentage: percentage of target digit to remove (for ANOMALOUS_MNIST)
            - data.random_state: random seed for anomaly creation
        uniform_dequantization: If True, add uniform dequantization to images.
        evaluation: If True, use evaluation batch size and disable shuffling.

    Returns:
        train_loader, eval_loader, dataset_info
    """
    # Compute batch size for this worker
    # Support both nested and flat config structures
    if hasattr(config, 'training') and hasattr(config.training, 'batch_size'):
        train_batch_size = config.training.batch_size
    else:
        train_batch_size = config.get('training.batch_size', config.get('batch_size', 128))
    
    if hasattr(config, 'eval') and hasattr(config.eval, 'batch_size'):
        eval_batch_size = config.eval.batch_size
    else:
        eval_batch_size = config.get('eval.batch_size', train_batch_size)
    
    batch_size = train_batch_size if not evaluation else eval_batch_size

    # Get dataset name from config
    if hasattr(config, 'data') and hasattr(config.data, 'dataset'):
        dataset_name = config.data.dataset
    else:
        dataset_name = config.get('data.dataset', config.get('dataset', 'CIFAR10'))
    
    # Get image size from config
    if hasattr(config, 'data') and hasattr(config.data, 'image_size'):
        image_size = config.data.image_size
    else:
        # Default image sizes based on dataset
        default_size = 32 if dataset_name == 'CIFAR10' else 28
        image_size = config.get('data.image_size', config.get('image_size', default_size))
    
    # Get random flip setting from config
    if hasattr(config, 'data') and hasattr(config.data, 'random_flip'):
        random_flip = config.data.random_flip
    else:
        random_flip = config.get('data.random_flip', config.get('random_flip', True))

    # Get dataset-specific parameters
    if hasattr(config, 'data') and hasattr(config.data, 'target_digit'):
        target_digit = config.data.target_digit
    else:
        target_digit = config.get('data.target_digit', config.get('target_digit', 8))
    
    if hasattr(config, 'data') and hasattr(config.data, 'removal_percentage'):
        removal_percentage = config.data.removal_percentage
    else:
        removal_percentage = config.get('data.removal_percentage', config.get('removal_percentage', 0.98))
    
    if hasattr(config, 'data') and hasattr(config.data, 'random_state'):
        random_state = config.data.random_state
    else:
        random_state = config.get('data.random_state', config.get('random_state', 42))

    # Create transforms based on dataset
    transform_list = []
    
    if dataset_name in ['MNIST', 'ANOMALOUS_MNIST', 'SINGLE_DIGIT_MNIST']:
        # MNIST-specific transforms
        # Resize for U-Net if image_size is specified and different from 28
        if image_size != 28:
            transform_list.append(transforms.Resize((image_size, image_size)))
        
        # Convert to tensor (scales to [0, 1])
        transform_list.append(transforms.ToTensor())
        
        # Convert grayscale to RGB (1 channel -> 3 channels) only if needed
        if image_size == 32:  # Assuming you want RGB for 32x32 images
            transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
        
    elif dataset_name == 'CIFAR10':
        # CIFAR-10 specific transforms
        # Resize if needed (CIFAR-10 is 32x32 by default)
        if image_size != 32:
            transform_list.append(transforms.Resize((image_size, image_size)))
        
        # Convert to tensor (scales to [0, 1])
        transform_list.append(transforms.ToTensor())
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported: 'CIFAR10', 'MNIST', 'ANOMALOUS_MNIST', 'SINGLE_DIGIT_MNIST'")
    
    # Random flip for training (apply to both datasets)
    train_transform_list = transform_list.copy()
    if random_flip and not evaluation and dataset_name != 'MNIST':  # Usually don't flip MNIST
        train_transform_list.insert(-1, transforms.RandomHorizontalFlip())
    
    # Apply uniform dequantization if requested
    if uniform_dequantization:
        def uniform_dequant(img):
            # Add uniform noise and rescale
            noise = torch.rand_like(img) / 256.0
            return (img * 255.0 + noise) / 256.0
        
        train_transform_list.append(transforms.Lambda(uniform_dequant))
        if evaluation:
            transform_list.append(transforms.Lambda(uniform_dequant))

    train_transform = transforms.Compose(train_transform_list)
    eval_transform = transforms.Compose(transform_list)

    # Create datasets based on dataset name
    if dataset_name == 'CIFAR10':
        train_dataset = CIFAR10(
            root='./data', 
            train=True, 
            download=True, 
            transform=train_transform
        )
        
        eval_dataset = CIFAR10(
            root='./data', 
            train=False, 
            download=True, 
            transform=eval_transform
        )
        
        num_classes = 10
        class_names = train_dataset.classes
        
    elif dataset_name == 'MNIST':
        train_dataset = MNIST(
            root='./data', 
            train=True, 
            download=True, 
            transform=train_transform
        )
        
        eval_dataset = MNIST(
            root='./data', 
            train=False, 
            download=True, 
            transform=eval_transform
        )
        
        num_classes = 10
        class_names = [str(i) for i in range(10)]
        
    elif dataset_name == 'ANOMALOUS_MNIST':
        train_dataset = AnomalousMNIST(
            root='./data', 
            train=True, 
            target_digit=target_digit,
            removal_percentage=removal_percentage,
            transform=train_transform,
            random_state=random_state
        )
        
        eval_dataset = AnomalousMNIST(
            root='./data', 
            train=False, 
            target_digit=target_digit,
            removal_percentage=removal_percentage,
            transform=eval_transform,
            random_state=random_state
        )
        
        num_classes = 10
        class_names = train_dataset.classes
        
    elif dataset_name == 'SINGLE_DIGIT_MNIST':
        train_dataset = SingleDigitMNIST(
            root='./data', 
            train=True, 
            target_digit=target_digit,
            transform=train_transform
        )
        
        eval_dataset = SingleDigitMNIST(
            root='./data', 
            train=False, 
            target_digit=target_digit,
            transform=eval_transform
        )
        
        num_classes = 10
        class_names = train_dataset.classes

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=not evaluation,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )

    # Dataset info
    dataset_info = {
        'dataset_name': dataset_name,
        'num_classes': num_classes,
        'image_shape': (len(transform_list[1](torch.zeros(1, 28, 28))), image_size, image_size),  # Correct number of channels
        'train_size': len(train_dataset),
        'eval_size': len(eval_dataset),
        'class_names': class_names
    }
    
    # Add dataset-specific info
    if dataset_name == 'ANOMALOUS_MNIST':
        anomaly_train_count = np.sum(train_dataset.targets == target_digit)
        anomaly_test_count = np.sum(eval_dataset.targets == target_digit)
        dataset_info.update({
            'target_digit': target_digit,
            'removal_percentage': removal_percentage,
            'anomaly_train_count': anomaly_train_count,
            'anomaly_test_count': anomaly_test_count,
            'normal_train_count': len(train_dataset) - anomaly_train_count,
            'normal_test_count': len(eval_dataset) - anomaly_test_count
        })
    elif dataset_name == 'SINGLE_DIGIT_MNIST':
        dataset_info.update({
            'target_digit': target_digit,
            'digit_train_count': len(train_dataset),
            'digit_test_count': len(eval_dataset)
        })

    return train_loader, eval_loader, dataset_info

def test_model_on_all_digits(model, sde, config, device='cuda', num_samples_per_digit=50, 
                            apply_scaler=True, compute_likelihood=False, likelihood_fn=None):
    """
    Test a trained model on all digits (0-9) using get_dataset() for each digit.
    
    Args:
        model: Trained model (e.g., your U-Net)
        sde: The SDE object used for training
        config: Base config dictionary (will be modified for each digit)
        device: Device to run on
        num_samples_per_digit: How many samples of each digit to test
        apply_scaler: Whether to apply the [-1, 1] scaler used in training
        compute_likelihood: Whether to compute likelihood (requires likelihood_fn)
        likelihood_fn: Likelihood function from your likelihood.py
    
    Returns:
        Dictionary with results for each digit
    """
    
    # Scaler function (same as in your eval.py)
    scaler = lambda x: x * 2 - 1. if apply_scaler else lambda x: x
    
    results = {}
    
    for digit in range(10):
        print(f"\nTesting digit {digit}...")
        
        # Create config for this specific digit
        digit_config = config.copy()
        digit_config['data.dataset'] = 'SINGLE_DIGIT_MNIST'
        digit_config['data.target_digit'] = digit
        digit_config['eval.batch_size'] = min(num_samples_per_digit, 100)  # Reasonable batch size
        
        try:
            # Get dataloader for this digit using your get_dataset function
            _, eval_loader, dataset_info = get_dataset(
                digit_config, 
                uniform_dequantization=config.get('dequant', True), 
                evaluation=True
            )
            
            # Get samples of this digit
            samples_collected = 0
            all_images = []
            all_labels = []
            
            for images, labels in eval_loader:
                all_images.append(images)
                all_labels.append(labels)
                samples_collected += len(images)
                if samples_collected >= num_samples_per_digit:
                    break
            
            if not all_images:
                print(f"Warning: No samples found for digit {digit}")
                continue
                
            # Concatenate and take only requested number
            images = torch.cat(all_images, dim=0)[:num_samples_per_digit]
            labels = torch.cat(all_labels, dim=0)[:num_samples_per_digit]
            
        except Exception as e:
            print(f"Error loading digit {digit}: {e}")
            continue
        
        images = images.to(device)
        labels = labels.to(device)
        
        # Apply scaler if needed
        images_scaled = scaler(images) if apply_scaler else images
        
        digit_results = {
            'digit': digit,
            'num_samples': len(images),
            'images': images.cpu(),
            'labels': labels.cpu(),
            'dataset_info': dataset_info
        }
        
        # Compute model scores if model is available
        if model is not None:
            model.eval()
            with torch.no_grad():
                # Test at multiple timesteps
                timesteps_to_test = [0.001, 0.1, 0.5, 1.0]
                scores = {}
                
                for t_val in timesteps_to_test:
                    t = torch.ones(len(images), device=device) * t_val
                    
                    try:
                        # Get score from model
                        score = model(images_scaled, t * 999)  # Scale to [0, 999] like in your training
                        
                        # Compute mean squared score as a simple metric
                        mean_score = torch.mean(score**2, dim=(1, 2, 3))
                        scores[f't_{t_val}'] = mean_score.cpu().numpy()
                    except Exception as e:
                        print(f"Error computing score for digit {digit} at t={t_val}: {e}")
                        scores[f't_{t_val}'] = None
                
                digit_results['scores'] = scores
        
        # Compute likelihood if requested
        if compute_likelihood and likelihood_fn is not None:
            print(f"Computing likelihood for digit {digit}...")
            try:
                # Process one sample at a time for likelihood computation
                bpds = []
                nfes = []
                logp_finals = []
                
                for i in range(min(len(images_scaled), 10)):  # Limit to 10 samples for speed
                    single_image = images_scaled[i:i+1]
                    bpd, z, nfe, logp_traj = likelihood_fn(model, single_image)
                    
                    bpds.append(bpd.item() if torch.is_tensor(bpd) else bpd)
                    nfes.append(nfe)
                    if isinstance(logp_traj, np.ndarray) and len(logp_traj) > 0:
                        logp_finals.append(logp_traj[-1])
                    else:
                        logp_finals.append(logp_traj)
                
                digit_results['likelihood'] = {
                    'bpd': np.array(bpds),
                    'nfe': np.array(nfes),
                    'logp_final': np.array(logp_finals)
                }
            except Exception as e:
                print(f"Error computing likelihood for digit {digit}: {e}")
                digit_results['likelihood'] = None
        
        results[digit] = digit_results
    
    return results

def analyze_digit_results(results, target_digit=8):
    """
    Analyze results from test_model_on_all_digits to identify anomalies.
    
    Args:
        results: Output from test_model_on_all_digits
        target_digit: The digit that should be anomalous
    
    Returns:
        Analysis summary
    """
    
    print(f"\n{'='*60}")
    print(f"ANALYSIS: Digit {target_digit} as Anomaly")
    print(f"{'='*60}")
    
    # Analyze scores at different timesteps
    if len(results) > 0 and 'scores' in list(results.values())[0]:
        first_result = list(results.values())[0]
        if first_result['scores']:
            for t_key in first_result['scores'].keys():
                print(f"\n--- Scores at {t_key} ---")
                scores_by_digit = {}
                
                for digit in range(10):
                    if digit in results and 'scores' in results[digit] and results[digit]['scores'].get(t_key) is not None:
                        scores = results[digit]['scores'][t_key]
                        mean_score = np.mean(scores)
                        std_score = np.std(scores)
                        scores_by_digit[digit] = (mean_score, std_score)
                        
                        status = "🚨 ANOMALY" if digit == target_digit else "✅ Normal"
                        print(f"Digit {digit}: {mean_score:.4f} ± {std_score:.4f} {status}")
                
                # Find digit with highest mean score (potential anomaly indicator)
                if scores_by_digit:
                    max_score_digit = max(scores_by_digit.keys(), key=lambda d: scores_by_digit[d][0])
                    print(f"Highest score: Digit {max_score_digit} (Expected anomaly: {target_digit})")
    
    # Analyze likelihoods
    if len(results) > 0 and 'likelihood' in list(results.values())[0]:
        first_result = list(results.values())[0]
        if first_result.get('likelihood') is not None:
            print(f"\n--- Likelihood Analysis ---")
            likelihoods_by_digit = {}
            
            for digit in range(10):
                if digit in results and 'likelihood' in results[digit] and results[digit]['likelihood'] is not None:
                    bpds = results[digit]['likelihood']['bpd']
                    mean_bpd = np.mean(bpds)
                    likelihoods_by_digit[digit] = mean_bpd
                    
                    status = "🚨 ANOMALY" if digit == target_digit else "✅ Normal"
                    print(f"Digit {digit}: {mean_bpd:.4f} bits/dim {status}")
            
            # Find digit with highest bits/dim (lowest likelihood - potential anomaly)
            if likelihoods_by_digit:
                max_bpd_digit = max(likelihoods_by_digit.keys(), key=lambda d: likelihoods_by_digit[d])
                print(f"Highest bits/dim (lowest likelihood): Digit {max_bpd_digit} (Expected anomaly: {target_digit})")
    
    return results

def visualize_digit_samples(digit, num_samples=10, train=False, title_prefix=""):
    """
    Visualize samples of a specific digit using get_dataset.
    
    Args:
        digit: Digit to visualize (0-9)
        num_samples: Number of samples to show
        train: Whether to use training or test set
        title_prefix: Prefix for the plot title
    """
    
    config = {
        'data.dataset': 'SINGLE_DIGIT_MNIST',
        'data.target_digit': digit,
        'eval.batch_size': num_samples,
        'data.image_size': 28  # Keep original size for visualization
    }
    
    # Get dataloader for this digit
    train_loader, eval_loader, _ = get_dataset(config, evaluation=not train)
    dataloader = train_loader if train else eval_loader
    
    # Get a batch
    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    
    # Plot
    fig, axes = plt.subplots(1, min(num_samples, len(images)), figsize=(15, 3))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(min(num_samples, len(images))):
        img = images[i].squeeze()  # Remove channel dimension
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Digit {labels[i].item()}')
        axes[i].axis('off')
    
    plt.suptitle(f'{title_prefix}Samples of Digit {digit}')
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    print("=== Testing Single Digit Functionality with get_dataset() ===")
    
    # Example 1: Test individual digits using get_dataset()
    print("\n--- Example 1: Single Digit using get_dataset() ---")
    
    # Config for digit 8 only
    digit_8_config = {
        'beta_min': 0.1,
        'beta_max': 20.0,
        'timesteps': 1000,
        'lr': 2e-4,
        'batch_size': 32,
        'eval.batch_size': 16,
        'dequant': True,
        'data.dataset': 'SINGLE_DIGIT_MNIST',
        'data.target_digit': 8,
        'data.image_size': 32,
        'data.random_flip': False
    }
    
    # Create dataloader for digit 8 using your framework
    train_loader, eval_loader, dataset_info = get_dataset(
        digit_8_config, 
        uniform_dequantization=True, 
        evaluation=False
    )
    
    print(f"Dataset info: {dataset_info}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Evaluation batches: {len(eval_loader)}")
    
    # Test loading a batch
    for batch_idx, (images, labels) in enumerate(eval_loader):
        print(f"Batch {batch_idx}: images shape {images.shape}, labels shape {labels.shape}")
        print(f"All labels are 8: {torch.all(labels == 8).item()}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        if batch_idx == 0:
            break
    
    # Example 2: Complete model testing workflow
    print("\n--- Example 2: Complete Testing Workflow ---")
    
    test_config = {
        'beta_min': 0.1,
        'beta_max': 20.0,
        'timesteps': 1000,
        'eval.batch_size': 50,
        'dequant': True,
        'data.dataset': 'SINGLE_DIGIT_MNIST',
        'data.image_size': 32,
        'data.random_flip': False
    }
    
    print("""
    # Load your trained model
    device = 'cuda:0'
    model = Unet().to(device)
    if config['use_ema']:
        ema_model = torch.optim.swa_utils.AveragedModel(model, ...)
        ema_model.load_state_dict(torch.load('ema_chk_200.pt'))
        model = ema_model
    
    # Create SDE
    sde = SubVPSDE(config)
    
    # Test on all digits
    results = test_model_on_all_digits(
        model=model,
        sde=sde,
        config=test_config,
        device=device,
        num_samples_per_digit=50,
        apply_scaler=True,
        compute_likelihood=True,
        likelihood_fn=like_fn
    )
    
    # Analyze results
    analyze_digit_results(results, target_digit=8)
    """)
    
    print("\n--- How to use in eval.py ---")
    print("""
    # In your eval.py, to test specific digit:
    config['data.dataset'] = 'SINGLE_DIGIT_MNIST'
    config['data.target_digit'] = 8
    config['eval.batch_size'] = 1
    
    _, eval_loader, info = get_dataset(config=config, uniform_dequantization=True, evaluation=True)
    
    for x, _ in eval_loader:
        x = x.to(device)
        x = scaler(x)  # Apply [-1, 1] scaling
        
        score_fn = get_score_fn(sde, score_model)
        bpd, z, nfe, logp_traj = like_fn(score_model, x)
        print(f"Digit 8 likelihood: {bpd}")
        break
    """)
    
    # Example 3: Complete pipeline
    print("\n--- Example 3: Complete Anomaly Detection Pipeline ---")
    
    # Training config (anomalous dataset)
    train_config = {
        'data.dataset': 'ANOMALOUS_MNIST',
        'data.target_digit': 8,
        'data.removal_percentage': 0.98,
        'batch_size': 256,
        'data.image_size': 32
    }
    
    # Testing config (individual digits)
    test_config = {
        'data.dataset': 'SINGLE_DIGIT_MNIST',
        'eval.batch_size': 50,
        'dequant': True,
        'data.image_size': 32
    }
    
    print("Complete pipeline:")
    print("1. Train model using ANOMALOUS_MNIST (digit 8 is rare)")
    print("2. Test model using SINGLE_DIGIT_MNIST for each digit 0-9")
    print("3. Compare scores/likelihoods - digit 8 should be anomalous")
    
    # Show dataset statistics
    print("\n--- Dataset Statistics ---")
    for digit in range(10):
        config = {
            'data.dataset': 'SINGLE_DIGIT_MNIST',
            'data.target_digit': digit,
            'eval.batch_size': 1000
        }
        try:
            _, eval_loader, dataset_info = get_dataset(config, evaluation=True)
            print(f"Digit {digit}: {dataset_info['digit_test_count']} samples available")
        except:
            print(f"Digit {digit}: Error loading dataset")
    
    print("\nNow you can seamlessly test your model on individual digits!")