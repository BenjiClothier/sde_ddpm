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
            - data.dataset or dataset: dataset name ('CIFAR10', 'MNIST', or 'ANOMALOUS_MNIST')
            - data.target_digit: digit to make anomalous (for ANOMALOUS_MNIST)
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

    # Get anomaly-specific parameters
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
    
    if dataset_name in ['MNIST', 'ANOMALOUS_MNIST']:
        # MNIST-specific transforms
        # Resize for U-Net
        transform_list.append(transforms.Resize((32, 32)))
        
        # Convert to tensor (scales to [0, 1])
        transform_list.append(transforms.ToTensor())
        
        # Convert grayscale to RGB (1 channel -> 3 channels)
        transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
        
    elif dataset_name == 'CIFAR10':
        # CIFAR-10 specific transforms
        # Resize if needed (CIFAR-10 is 32x32 by default)
        if image_size != 32:
            transform_list.append(transforms.Resize((image_size, image_size)))
        
        # Convert to tensor (scales to [0, 1])
        transform_list.append(transforms.ToTensor())
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported: 'CIFAR10', 'MNIST', 'ANOMALOUS_MNIST'")
    
    # Random flip for training (apply to both datasets)
    train_transform_list = transform_list.copy()
    if random_flip and not evaluation:
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
        class_names = [str(i) for i in range(10)]  # MNIST classes are 0-9
        
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
        'image_shape': (3, image_size, image_size),  # Always 3 channels after transforms
        'train_size': len(train_dataset),
        'eval_size': len(eval_dataset),
        'class_names': class_names
    }
    
    # Add anomaly-specific info
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

    return train_loader, eval_loader, dataset_info
def create_single_digit_dataloader(digit, root='./data', train=False, batch_size=32, 
                                 apply_transforms=True, image_size=32, shuffle=True):
    """
    Create a DataLoader that contains only samples of a specific digit.
    
    Args:
        digit: The digit to include (0-9)
        root: Root directory for dataset
        train: Whether to use training or test set
        batch_size: Batch size for the dataloader
        apply_transforms: Whether to apply standard transforms (resize, RGB conversion)
        image_size: Target image size (default: 32 for compatibility with your U-Net)
        shuffle: Whether to shuffle the data
    
    Returns:
        DataLoader containing only the specified digit
    """
    
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
    digit_indices = np.where(targets == digit)[0]
    digit_data = data[digit_indices]
    digit_targets = targets[digit_indices]
    
    print(f"Found {len(digit_data)} samples of digit {digit} in {'training' if train else 'test'} set")
    
    # Convert to tensors
    digit_data = torch.from_numpy(digit_data).float() / 255.0  # Normalize to [0, 1]
    digit_targets = torch.from_numpy(digit_targets).long()
    
    # Add channel dimension
    digit_data = digit_data.unsqueeze(1)  # Shape: (N, 1, 28, 28)
    
    # Apply transforms if requested
    if apply_transforms:
        transform_list = []
        
        # Resize to target size
        if image_size != 28:
            transform_list.append(transforms.Resize((image_size, image_size)))
        
        # Convert grayscale to RGB (1 channel -> 3 channels)
        transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
        
        if transform_list:
            transform = transforms.Compose(transform_list)
            # Apply transforms to each sample
            transformed_data = []
            for i in range(len(digit_data)):
                transformed_sample = transform(digit_data[i])
                transformed_data.append(transformed_sample)
            digit_data = torch.stack(transformed_data)
    
    # Create dataset and dataloader
    dataset = TensorDataset(digit_data, digit_targets)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        drop_last=False  # Don't drop last batch to get all samples
    )
    
    return dataloader


def create_all_digits_dataloaders(root='./data', train=False, batch_size=32, 
                                 apply_transforms=True, image_size=32, shuffle=True):
    """
    Create DataLoaders for all digits (0-9), each containing only samples of that digit.
    
    Returns:
        Dictionary with keys 0-9, each containing a DataLoader for that digit
    """
    
    dataloaders = {}
    
    for digit in range(10):
        dataloader = create_single_digit_dataloader(
            digit=digit,
            root=root,
            train=train,
            batch_size=batch_size,
            apply_transforms=apply_transforms,
            image_size=image_size,
            shuffle=shuffle
        )
        dataloaders[digit] = dataloader
    
def test_model_on_digits(model, sde, device='cuda', num_samples_per_digit=10, 
                        apply_scaler=True, compute_likelihood=False, likelihood_fn=None):
    """
    Test a trained model on specific digits and compute various metrics.
    
    Args:
        model: Trained model (e.g., your U-Net)
        sde: The SDE object used for training
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
    
    # Create dataloaders for all digits
    digit_dataloaders = create_all_digits_dataloaders(
        train=False,  # Use test set
        batch_size=num_samples_per_digit,
        apply_transforms=True,
        image_size=32,
        shuffle=True
    )
    
    results = {}
    
    for digit in range(10):
        print(f"\nTesting digit {digit}...")
        
        # Get a batch of this digit
        dataloader = digit_dataloaders[digit]
        data_iter = iter(dataloader)
        images, labels = next(data_iter)
        
        # Take only the requested number of samples
        images = images[:num_samples_per_digit]
        labels = labels[:num_samples_per_digit]
        
        images = images.to(device)
        labels = labels.to(device)
        
        # Apply scaler if needed
        if apply_scaler:
            images_scaled = scaler(images)
        else:
            images_scaled = images
        
        digit_results = {
            'digit': digit,
            'num_samples': len(images),
            'images': images.cpu(),
            'labels': labels.cpu()
        }
        
        # Compute reconstruction/score if model is available
        if model is not None:
            model.eval()
            with torch.no_grad():
                # Test at multiple timesteps
                timesteps_to_test = [0.001, 0.1, 0.5, 1.0]
                scores = {}
                
                for t_val in timesteps_to_test:
                    t = torch.ones(len(images), device=device) * t_val
                    
                    # Get score from model
                    score = model(images_scaled, t * 999)  # Scale to [0, 999] like in your training
                    
                    # Compute mean squared score as a simple metric
                    mean_score = torch.mean(score**2, dim=(1, 2, 3))
                    scores[f't_{t_val}'] = mean_score.cpu().numpy()
                
                digit_results['scores'] = scores
        
        # Compute likelihood if requested
        if compute_likelihood and likelihood_fn is not None:
            print(f"Computing likelihood for digit {digit}...")
            try:
                bpd, z, nfe, logp_traj = likelihood_fn(model, images_scaled)
                digit_results['likelihood'] = {
                    'bpd': bpd.cpu().numpy() if torch.is_tensor(bpd) else bpd,
                    'nfe': nfe,
                    'logp_final': logp_traj[-1] if isinstance(logp_traj, np.ndarray) else logp_traj
                }
            except Exception as e:
                print(f"Error computing likelihood for digit {digit}: {e}")
                digit_results['likelihood'] = None
        
        results[digit] = digit_results
    
    return results


def analyze_digit_results(results, target_digit=8):
    """
    Analyze results from test_model_on_digits to identify anomalies.
    
    Args:
        results: Output from test_model_on_digits
        target_digit: The digit that should be anomalous
    
    Returns:
        Analysis summary
    """
    
    print(f"\n{'='*60}")
    print(f"ANALYSIS: Digit {target_digit} as Anomaly")
    print(f"{'='*60}")
    
    # Analyze scores at different timesteps
    if 'scores' in results[0]:
        for t_key in results[0]['scores'].keys():
            print(f"\n--- Scores at {t_key} ---")
            scores_by_digit = {}
            
            for digit in range(10):
                if digit in results and 'scores' in results[digit]:
                    mean_score = np.mean(results[digit]['scores'][t_key])
                    std_score = np.std(results[digit]['scores'][t_key])
                    scores_by_digit[digit] = (mean_score, std_score)
                    
                    status = "🚨 ANOMALY" if digit == target_digit else "✅ Normal"
                    print(f"Digit {digit}: {mean_score:.4f} ± {std_score:.4f} {status}")
            
            # Find digit with highest mean score (potential anomaly indicator)
            if scores_by_digit:
                max_score_digit = max(scores_by_digit.keys(), key=lambda d: scores_by_digit[d][0])
                print(f"Highest score: Digit {max_score_digit} (Expected anomaly: {target_digit})")
    
    # Analyze likelihoods
    if 'likelihood' in results[0] and results[0]['likelihood'] is not None:
        print(f"\n--- Likelihood Analysis ---")
        likelihoods_by_digit = {}
        
        for digit in range(10):
            if digit in results and 'likelihood' in results[digit] and results[digit]['likelihood'] is not None:
                bpd = results[digit]['likelihood']['bpd']
                mean_bpd = np.mean(bpd) if hasattr(bpd, '__len__') else bpd
                likelihoods_by_digit[digit] = mean_bpd
                
                status = "🚨 ANOMALY" if digit == target_digit else "✅ Normal"
                print(f"Digit {digit}: {mean_bpd:.4f} bits/dim {status}")
        
        # Find digit with highest bits/dim (lowest likelihood - potential anomaly)
        if likelihoods_by_digit:
            max_bpd_digit = max(likelihoods_by_digit.keys(), key=lambda d: likelihoods_by_digit[d])
            print(f"Highest bits/dim (lowest likelihood): Digit {max_bpd_digit} (Expected anomaly: {target_digit})")
    
    return results


def create_anomaly_dataloaders(root='./data', target_digit=8, removal_percentage=0.98, 
                              batch_size=64, random_state=42):
    """
    Create DataLoaders for anomaly detection with separate normal and anomalous samples.
    
    Returns:
    - Dictionary with separate DataLoaders for normal and anomalous samples
    """
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Create modified datasets
    train_dataset = AnomalousMNIST(root=root, train=True, target_digit=target_digit, 
                                  removal_percentage=removal_percentage, 
                                  transform=None, random_state=random_state)
    
    test_dataset = AnomalousMNIST(root=root, train=False, target_digit=target_digit, 
                                 removal_percentage=removal_percentage, 
                                 transform=None, random_state=random_state)
    
    # Separate normal and anomalous samples
    def separate_normal_anomalous(dataset):
        normal_data = []
        normal_targets = []
        anomaly_data = []
        anomaly_targets = []
        
        for i in range(len(dataset)):
            sample, target = dataset[i]
            if target == target_digit:
                anomaly_data.append(sample)
                anomaly_targets.append(target)
            else:
                normal_data.append(sample)
                normal_targets.append(target)
        
        if normal_data:
            normal_data = torch.stack(normal_data)
            normal_targets = torch.tensor(normal_targets)
        else:
            normal_data = torch.empty(0, 1, 28, 28)
            normal_targets = torch.empty(0, dtype=torch.long)
            
        if anomaly_data:
            anomaly_data = torch.stack(anomaly_data)
            anomaly_targets = torch.tensor(anomaly_targets)
        else:
            anomaly_data = torch.empty(0, 1, 28, 28)
            anomaly_targets = torch.empty(0, dtype=torch.long)
        
        return (normal_data, normal_targets), (anomaly_data, anomaly_targets)
    
    # Separate training data
    (train_normal_data, train_normal_targets), (train_anomaly_data, train_anomaly_targets) = separate_normal_anomalous(train_dataset)
    
    # Separate test data
    (test_normal_data, test_normal_targets), (test_anomaly_data, test_anomaly_targets) = separate_normal_anomalous(test_dataset)
    
    # Create DataLoaders
    dataloaders = {
        'train_normal': DataLoader(TensorDataset(train_normal_data, train_normal_targets), 
                                  batch_size=batch_size, shuffle=True),
        'train_anomaly': DataLoader(TensorDataset(train_anomaly_data, train_anomaly_targets), 
                                   batch_size=batch_size, shuffle=True),
        'test_normal': DataLoader(TensorDataset(test_normal_data, test_normal_targets), 
                                 batch_size=batch_size, shuffle=False),
        'test_anomaly': DataLoader(TensorDataset(test_anomaly_data, test_anomaly_targets), 
                                  batch_size=batch_size, shuffle=False)
    }
    
    # Print dataset statistics
    print(f"\nDataset Statistics:")
    print(f"Training normal samples: {len(train_normal_data)}")
    print(f"Training anomalous samples: {len(train_anomaly_data)}")
    print(f"Test normal samples: {len(test_normal_data)}")
    print(f"Test anomalous samples: {len(test_anomaly_data)}")
    
    return dataloaders

class SimpleAutoencoder(nn.Module):
    """Simple autoencoder for anomaly detection"""
    
    def __init__(self, input_dim=784, hidden_dims=[128, 64, 32]):
        super(SimpleAutoencoder, self).__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        # Decoder
        decoder_layers = []
        for hidden_dim in reversed(hidden_dims[:-1]):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        decoder_layers.extend([
            nn.Linear(prev_dim, input_dim),
            nn.Sigmoid()
        ])
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        # Encode
        encoded = self.encoder(x)
        # Decode
        decoded = self.decoder(encoded)
        # Reshape to original dimensions
        decoded = decoded.view(x.size(0), 1, 28, 28)
        return decoded

def train_autoencoder(model, train_loader, num_epochs=50, learning_rate=0.001):
    """Train the autoencoder on normal samples only"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            
            optimizer.zero_grad()
            reconstructed = model(data)
            loss = criterion(reconstructed, data)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
    
    return model

def evaluate_anomaly_detection(model, normal_loader, anomaly_loader):
    """Evaluate anomaly detection performance"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    def calculate_reconstruction_errors(dataloader):
        errors = []
        with torch.no_grad():
            for data, _ in dataloader:
                data = data.to(device)
                reconstructed = model(data)
                # Calculate MSE for each sample
                mse = torch.mean((data - reconstructed) ** 2, dim=(1, 2, 3))
                errors.extend(mse.cpu().numpy())
        return np.array(errors)
    
    # Calculate reconstruction errors
    normal_errors = calculate_reconstruction_errors(normal_loader)
    anomaly_errors = calculate_reconstruction_errors(anomaly_loader)
    
    print(f"\nReconstruction Error Statistics:")
    print(f"Normal samples - Mean: {np.mean(normal_errors):.4f}, Std: {np.std(normal_errors):.4f}")
    print(f"Anomalous samples - Mean: {np.mean(anomaly_errors):.4f}, Std: {np.std(anomaly_errors):.4f}")
    
    # Calculate AUC score
    if len(anomaly_errors) > 0:
        y_true = np.concatenate([np.zeros(len(normal_errors)), np.ones(len(anomaly_errors))])
        y_scores = np.concatenate([normal_errors, anomaly_errors])
        auc_score = roc_auc_score(y_true, y_scores)
        print(f"AUC Score: {auc_score:.4f}")
    
    # Plot reconstruction errors
    plt.figure(figsize=(10, 6))
    plt.hist(normal_errors, bins=50, alpha=0.7, label='Normal (non-8)', density=True)
    if len(anomaly_errors) > 0:
        plt.hist(anomaly_errors, bins=50, alpha=0.7, label='Anomalous (8)', density=True)
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.title('Reconstruction Error Distribution')
    plt.legend()
    plt.show()
    
    return normal_errors, anomaly_errors

def visualize_samples(dataloader, num_samples=5, title="Samples"):
    """Visualize samples from a dataloader"""
    
    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    
    fig, axes = plt.subplots(1, min(num_samples, len(images)), figsize=(15, 3))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(min(num_samples, len(images))):
        img = images[i].squeeze()
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Label: {labels[i].item()}')
        axes[i].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

    return dataloader


def visualize_digit_samples(digit, num_samples=10, train=False, title_prefix=""):
    """
    Visualize samples of a specific digit.
    
    Args:
        digit: Digit to visualize (0-9)
        num_samples: Number of samples to show
        train: Whether to use training or test set
        title_prefix: Prefix for the plot title
    """
    
    dataloader = create_single_digit_dataloader(
        digit=digit,
        train=train,
        batch_size=num_samples,
        apply_transforms=False,  # Keep original format for visualization
        shuffle=True
    )
    
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


# Example usage for testing specific digits
if __name__ == "__main__":
    print("=== Testing Single Digit DataLoaders ===")
    
    # Example 1: Create a dataloader for digit 8 only
    print("\n--- Example 1: Single Digit DataLoader ---")
    digit_8_loader = create_single_digit_dataloader(
        digit=8,
        train=False,  # Use test set
        batch_size=16,
        apply_transforms=True,
        image_size=32,
        shuffle=True
    )
    
    # Test loading a batch
    for batch_idx, (images, labels) in enumerate(digit_8_loader):
        print(f"Batch {batch_idx}: images shape {images.shape}, labels shape {labels.shape}")
        print(f"All labels are 8: {torch.all(labels == 8).item()}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        if batch_idx == 0:  # Only show first batch info
            break
    
    # Example 2: Create dataloaders for all digits
    print("\n--- Example 2: All Digits DataLoaders ---")
    all_digit_loaders = create_all_digits_dataloaders(
        train=False,
        batch_size=10,
        apply_transforms=True,
        image_size=32,
        shuffle=True
    )
    
    # Show statistics for each digit
    for digit, loader in all_digit_loaders.items():
        total_samples = len(loader.dataset)
        print(f"Digit {digit}: {total_samples} samples, {len(loader)} batches")
    
    # Example 3: Visualize samples of each digit
    print("\n--- Example 3: Visualizing Digit Samples ---")
    for digit in [0, 1, 8, 9]:  # Show a few examples
        print(f"Visualizing digit {digit}")
        visualize_digit_samples(digit, num_samples=5, train=False, title_prefix="Test Set - ")
    
    # Example 4: How to use with your model for testing
    print("\n--- Example 4: Model Testing Setup ---")
    print("""
    To test your trained model on specific digits:
    
    # Load your trained model
    model = Unet().to(device)
    model.load_state_dict(torch.load('your_model.pth'))
    
    # Create SDE (same as training)
    sde = SubVPSDE(config)
    
    # Test on specific digits
    results = test_model_on_digits(
        model=model,
        sde=sde,
        device='cuda',
        num_samples_per_digit=20,
        apply_scaler=True,
        compute_likelihood=False  # Set to True if you want likelihood analysis
    )
    
    # Analyze results to identify anomalies
    analysis = analyze_digit_results(results, target_digit=8)
    """)
    
    # Example 5: Integration with your existing framework
    print("\n--- Example 5: Integration with get_dataset() ---")
    
    # You can still use the regular anomalous dataset for training
    config = {
        'data.dataset': 'ANOMALOUS_MNIST',
        'data.target_digit': 8,
        'data.removal_percentage': 0.98,
        'batch_size': 128
    }
    
    print("1. Train your model on ANOMALOUS_MNIST using get_dataset()")
    print("2. Test the trained model on individual digits using create_single_digit_dataloader()")
    print("3. Compare scores/likelihoods across digits to validate anomaly detection")
    
    # Example config for likelihood testing
    likelihood_config = {
        'beta_min': 0.1,
        'beta_max': 20.0,
        'timesteps': 1000,
        'ODE': True,
        'eval.batch_size': 1
    }
    
    print(f"\nFor likelihood-based anomaly detection:")
    print(f"- Train on normal digits (0-7, 9) with rare digit 8")
    print(f"- Test likelihood on individual digits")
    print(f"- Digit 8 should have lower likelihood (higher bits/dim)")
    
    # Show how many samples of each digit are available
    print(f"\n--- Available samples per digit in test set ---")
    all_loaders = create_all_digits_dataloaders(train=False, batch_size=1000)
    for digit, loader in all_loaders.items():
        print(f"Digit {digit}: {len(loader.dataset)} samples available for testing")
