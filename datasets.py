import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST


def get_dataset(config, uniform_dequantization=False, evaluation=False):
    """Create data loaders for training and evaluation.

    Args:
        config: A configuration dictionary with keys like:
            - training.batch_size or batch_size: batch size for training
            - eval.batch_size: batch size for evaluation  
            - data.image_size or image_size: target image size
            - data.random_flip or random_flip: whether to apply random horizontal flip
            - data.dataset or dataset: dataset name ('CIFAR10' or 'MNIST')
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

    # Create transforms based on dataset
    transform_list = []
    
    if dataset_name == 'MNIST':
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
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported: 'CIFAR10', 'MNIST'")
    
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

    return train_loader, eval_loader, dataset_info


# Example usage with your config structure
if __name__ == "__main__":
    # CIFAR-10 config
    cifar_config = {
        'beta_min': 0.1,
        'beta_max': 20.0,
        'timesteps': 1000,
        'lr': 2e-4,
        'warmup': 5000,
        'batch_size': 128,
        'epochs': 100,
        'log_freq': 100,
        'num_workers': 2,
        'use_ema': True,
        'ODE': True,
        'dequant': True,
        # Dataset specific keys:
        'training.batch_size': 128,
        'eval.batch_size': 256,
        'data.dataset': 'CIFAR10',  # or 'MNIST'
        'data.image_size': 32,
        'data.random_flip': True
    }
    
    # MNIST config
    mnist_config = {
        'beta_min': 0.1,
        'beta_max': 20.0,
        'timesteps': 1000,
        'lr': 2e-4,
        'warmup': 5000,
        'batch_size': 128,
        'epochs': 100,
        'log_freq': 100,
        'num_workers': 2,
        'use_ema': True,
        'ODE': True,
        'dequant': True,
        # Dataset specific keys:
        'training.batch_size': 128,
        'eval.batch_size': 256,
        'data.dataset': 'MNIST',
        'data.image_size': 28,  # MNIST native size, or 32 to match CIFAR-10
        'data.random_flip': False  # Usually False for MNIST
    }
    
    # Test CIFAR-10
    print("=== CIFAR-10 ===")
    train_loader, eval_loader, dataset_info = get_dataset(cifar_config, uniform_dequantization=cifar_config['dequant'], evaluation=False)
    
    print(f"Dataset info: {dataset_info}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Evaluation batches: {len(eval_loader)}")
    
    # Test loading a batch
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"CIFAR-10 Batch {batch_idx}: images shape {images.shape}, labels shape {labels.shape}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        break
    
    # Test MNIST
    print("\n=== MNIST ===")
    train_loader, eval_loader, dataset_info = get_dataset(mnist_config, uniform_dequantization=mnist_config['dequant'], evaluation=False)
    
    print(f"Dataset info: {dataset_info}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Evaluation batches: {len(eval_loader)}")
    
    # Test loading a batch
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"MNIST Batch {batch_idx}: images shape {images.shape}, labels shape {labels.shape}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"Sample labels: {labels[:5].tolist()}")
        break