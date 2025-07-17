import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
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

# Example usage
if __name__ == "__main__":
    # Create anomalous MNIST dataloaders
    dataloaders = create_anomaly_dataloaders(
        target_digit=8, 
        removal_percentage=0.98,
        batch_size=64,
        random_state=42
    )
    
    # Visualize some samples
    print("Visualizing normal samples:")
    visualize_samples(dataloaders['train_normal'], num_samples=5, title="Normal Training Samples")
    
    if len(dataloaders['train_anomaly'].dataset) > 0:
        print("Visualizing anomalous samples:")
        visualize_samples(dataloaders['train_anomaly'], num_samples=5, title="Anomalous Training Samples")
    
    # Create and train autoencoder
    print("\nTraining autoencoder on normal samples...")
    autoencoder = SimpleAutoencoder(input_dim=784, hidden_dims=[128, 64, 32])
    trained_model = train_autoencoder(autoencoder, dataloaders['train_normal'], num_epochs=50)
    
    # Evaluate anomaly detection
    print("\nEvaluating anomaly detection performance...")
    normal_errors, anomaly_errors = evaluate_anomaly_detection(
        trained_model, 
        dataloaders['test_normal'], 
        dataloaders['test_anomaly']
    )
    
    # Save the model
    torch.save(trained_model.state_dict(), 'anomaly_autoencoder.pth')
    print("\nModel saved as 'anomaly_autoencoder.pth'")
