import numpy as np
import matplotlib.pyplot as plt
from likelihood import get_likelihood_fn
from sde import SubVPSDE, get_score_fn, EulerMaruayamaPredictor, sample_images
import os

def plot_digit_trajectories_grouped(trajectories_by_digit, title="Log Probability Trajectories by Digit",
                                   xlabel="Integration Steps", ylabel="Log Probability", figsize=(14, 10)):
    """
    Plot trajectories grouped by digit with same color per digit.
    
    Args:
        trajectories_by_digit: Dict with digit as key and list of trajectories as value
                              e.g., {0: [traj1, traj2, ...], 1: [traj1, traj2, ...], ...}
        title: Plot title
        xlabel: X-axis label  
        ylabel: Y-axis label
        figsize: Figure size
    
    Returns:
        fig, ax: matplotlib figure and axis objects
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get colors for each digit
    colors = plt.cm.tab10(np.linspace(0, 1, 10))  # 10 colors for digits 0-9
    
    # Plot trajectories for each digit
    for digit, trajectories in trajectories_by_digit.items():
        color = colors[digit]
        
        for i, traj in enumerate(trajectories):
            if traj is not None and len(traj) > 0:
                steps = np.arange(len(traj))
                
                # Only add label for the first trajectory of each digit
                label = f'Digit {digit}' if i == 0 else ""
                alpha = 0.7 if i == 0 else 0.4  # Make first trajectory more prominent
                linewidth = 2 if i == 0 else 1
                
                ax.plot(steps, traj, color=color, alpha=alpha, 
                       linewidth=linewidth, label=label)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig, ax

def plot_digit_trajectories_with_stats(trajectories_by_digit, title="Log Probability Trajectories with Statistics"):
    """
    Plot trajectories with mean and std bands for each digit.
    
    Args:
        trajectories_by_digit: Dict with digit as key and list of trajectories as value
        title: Plot title
    
    Returns:
        fig, ax: matplotlib figure and axis objects
    """
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for digit, trajectories in trajectories_by_digit.items():
        if not trajectories:
            continue
            
        color = colors[digit]
        
        # Find the minimum length to align all trajectories
        min_length = min(len(traj) for traj in trajectories if traj is not None and len(traj) > 0)
        
        if min_length == 0:
            continue
            
        # Truncate all trajectories to the same length and stack
        aligned_trajs = []
        for traj in trajectories:
            if traj is not None and len(traj) >= min_length:
                aligned_trajs.append(traj[:min_length])
        
        if not aligned_trajs:
            continue
            
        # Convert to numpy array and compute statistics
        trajs_array = np.array(aligned_trajs)
        mean_traj = np.mean(trajs_array, axis=0)
        std_traj = np.std(trajs_array, axis=0)
        
        steps = np.arange(min_length)
        
        # Plot individual trajectories with low alpha
        for traj in aligned_trajs:
            ax.plot(steps, traj, color=color, alpha=0.2, linewidth=0.8)
        
        # Plot mean trajectory
        ax.plot(steps, mean_traj, color=color, linewidth=3, 
               label=f'Digit {digit} (mean)', alpha=0.9)
        
        # Plot std band
        ax.fill_between(steps, mean_traj - std_traj, mean_traj + std_traj,
                       color=color, alpha=0.2)
    
    ax.set_xlabel("Integration Steps", fontsize=12)
    ax.set_ylabel("Log Probability", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig, ax


def plot_multiple_arrays(arrays, labels=None, title="Log Probability Trajectories", 
                        xlabel="Steps", ylabel="Log Probability", figsize=(12, 8),
                        colors=None, linestyles=None, alpha=0.7):
    """
    Plot multiple 1D numpy arrays of different lengths on a single graph.
    
    Args:
        arrays: List of 1D numpy arrays
        labels: List of labels for each array (optional)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size tuple
        colors: List of colors for each line (optional)
        linestyles: List of line styles for each line (optional)
        alpha: Transparency level
    
    Returns:
        fig, ax: matplotlib figure and axis objects
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Default colors and line styles
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(arrays)))
    if linestyles is None:
        linestyles = ['-'] * len(arrays)
    if labels is None:
        labels = [f'Array {i+1}' for i in range(len(arrays))]
    
    # Plot each array
    for i, arr in enumerate(arrays):
        if arr is not None and len(arr) > 0:
            steps = np.arange(len(arr))
            ax.plot(steps, arr, 
                   color=colors[i], 
                   linestyle=linestyles[i % len(linestyles)],
                   alpha=alpha,
                   label=labels[i],
                   linewidth=2)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig, ax

def plot_digit_trajectories(arrays, digit_labels=None, title="Log Probability Trajectories by Digit"):
    """
    Specialized function for plotting log probability trajectories for different digits.
    
    Args:
        arrays: List of 1D numpy arrays (log probability trajectories)
        digit_labels: List of digit labels (e.g., [0, 1, 2, 8, 9])
        title: Plot title
    
    Returns:
        fig, ax: matplotlib figure and axis objects
    """
    
    if digit_labels is None:
        digit_labels = list(range(len(arrays)))
    
    # Create labels with digit information
    labels = [f'Digit {digit}' for digit in digit_labels]
    
    # Use different colors for different digits
    colors = plt.cm.Set1(np.linspace(0, 1, len(arrays)))
    
    return plot_multiple_arrays(arrays, labels=labels, title=title, 
                               xlabel="Integration Steps", ylabel="Log Probability",
                               colors=colors)

def plot_anomaly_comparison(normal_arrays, anomaly_arrays, normal_labels=None, anomaly_labels=None,
                           title="Normal vs Anomalous Trajectories"):
    """
    Plot normal and anomalous trajectories with different styling.
    
    Args:
        normal_arrays: List of arrays for normal samples
        anomaly_arrays: List of arrays for anomalous samples
        normal_labels: Labels for normal samples
        anomaly_labels: Labels for anomalous samples
        title: Plot title
    
    Returns:
        fig, ax: matplotlib figure and axis objects
    """
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot normal trajectories in blue
    for i, arr in enumerate(normal_arrays):
        if arr is not None and len(arr) > 0:
            steps = np.arange(len(arr))
            label = f'Normal {normal_labels[i]}' if normal_labels else f'Normal {i+1}'
            ax.plot(steps, arr, color='blue', alpha=0.6, linewidth=1.5, 
                   linestyle='-', label=label if i < 5 else "")  # Only label first 5
    
    # Plot anomalous trajectories in red
    for i, arr in enumerate(anomaly_arrays):
        if arr is not None and len(arr) > 0:
            steps = np.arange(len(arr))
            label = f'Anomaly {anomaly_labels[i]}' if anomaly_labels else f'Anomaly {i+1}'
            ax.plot(steps, arr, color='red', alpha=0.8, linewidth=2, 
                   linestyle='--', label=label)
    
    ax.set_xlabel("Integration Steps", fontsize=12)
    ax.set_ylabel("Log Probability", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig, ax

def plot_and_save_individual_digits(trajectories_by_digit, save_dir="likelihood_results", 
                                   title_prefix="Log Probability Trajectories", 
                                   xlabel="Integration Steps", ylabel="Log Probability",
                                   figsize=(10, 6), dpi=300):
    """
    Create separate plots for each digit and save both data and plots.
    
    Args:
        trajectories_by_digit: Dict with digit as key and list of trajectories as value
        save_dir: Directory to save files
        title_prefix: Prefix for plot titles
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size for each plot
        dpi: DPI for saved PNG files
    
    Returns:
        saved_files: Dictionary with information about saved files
    """
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    saved_files = {}
    
    for digit, trajectories in trajectories_by_digit.items():
        if not trajectories:
            print(f"Warning: No trajectories found for digit {digit}")
            continue
        
        print(f"Processing and saving digit {digit}...")
        
        # Create figure for this digit
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get color for this digit
        color = plt.cm.tab10(digit / 10.0)
        
        # Prepare data for saving
        trajectories_data = []
        
        # Plot all trajectories for this digit
        for i, traj in enumerate(trajectories):
            if traj is not None and len(traj) > 0:
                steps = np.arange(len(traj))
                
                # Plot trajectory
                alpha = 0.8 if i == 0 else 0.6
                linewidth = 2.5 if i == 0 else 1.5
                
                ax.plot(steps, traj, color=color, alpha=alpha, 
                       linewidth=linewidth, label=f'Sample {i+1}' if i < 5 else "")
                
                # Store trajectory for saving
                trajectories_data.append(traj)
        
        # Customize plot
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'{title_prefix} - Digit {digit}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add legend only if we have labels
        if len(trajectories) <= 5:
            ax.legend()
        else:
            ax.text(0.02, 0.98, f'{len(trajectories)} samples', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Add statistics text
        if trajectories_data:
            final_values = [traj[-1] for traj in trajectories_data if len(traj) > 0]
            if final_values:
                mean_final = np.mean(final_values)
                std_final = np.std(final_values)
                ax.text(0.02, 0.02, f'Final Log-Prob:\n{mean_final:.3f} ± {std_final:.3f}', 
                       transform=ax.transAxes, verticalalignment='bottom',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Save PNG
        png_filename = f"digit_{digit}_trajectories.png"
        png_path = os.path.join(save_dir, png_filename)
        fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
        plt.show()  # Display the plot
        plt.close()  # Close to free memory
        
        # Save NPY data
        npy_filename = f"digit_{digit}_trajectories.npy"
        npy_path = os.path.join(save_dir, npy_filename)
        
        # Convert to object array to handle different lengths
        trajectories_array = np.array(trajectories_data, dtype=object)
        np.save(npy_path, trajectories_array)
        
        # Save individual trajectory files as well
        individual_dir = os.path.join(save_dir, f"digit_{digit}_individual")
        os.makedirs(individual_dir, exist_ok=True)
        
        for i, traj in enumerate(trajectories_data):
            individual_path = os.path.join(individual_dir, f"sample_{i+1}.npy")
            np.save(individual_path, traj)
        
        # Store file information
        saved_files[digit] = {
            'png_path': png_path,
            'npy_path': npy_path,
            'individual_dir': individual_dir,
            'num_samples': len(trajectories_data),
            'trajectory_lengths': [len(traj) for traj in trajectories_data]
        }
        
        print(f"  Saved PNG: {png_path}")
        print(f"  Saved NPY: {npy_path}")
        print(f"  Individual samples: {individual_dir}")
    
    # Save summary information
    summary_path = os.path.join(save_dir, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Likelihood Trajectory Analysis Summary\n")
        f.write("=" * 40 + "\n\n")
        
        for digit, info in saved_files.items():
            f.write(f"Digit {digit}:\n")
            f.write(f"  Samples: {info['num_samples']}\n")
            f.write(f"  Trajectory lengths: {info['trajectory_lengths']}\n")
            f.write(f"  PNG: {info['png_path']}\n")
            f.write(f"  NPY: {info['npy_path']}\n")
            f.write(f"  Individual: {info['individual_dir']}\n\n")
    
    print(f"\nSummary saved to: {summary_path}")
    return saved_files

def load_and_verify_saved_data(save_dir="likelihood_results", digit=None):
    """
    Load and verify saved trajectory data.
    
    Args:
        save_dir: Directory where files were saved
        digit: Specific digit to load (None for all)
    
    Returns:
        loaded_data: Dictionary of loaded trajectories
    """
    
    loaded_data = {}
    
    if digit is not None:
        digits_to_load = [digit]
    else:
        digits_to_load = range(10)
    
    for d in digits_to_load:
        npy_path = os.path.join(save_dir, f"digit_{d}_trajectories.npy")
        
        if os.path.exists(npy_path):
            trajectories = np.load(npy_path, allow_pickle=True)
            loaded_data[d] = trajectories
            print(f"Loaded digit {d}: {len(trajectories)} trajectories")
            
            # Print some statistics
            lengths = [len(traj) for traj in trajectories]
            final_values = [traj[-1] for traj in trajectories if len(traj) > 0]
            
            print(f"  Trajectory lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")
            if final_values:
                print(f"  Final log-probs: mean={np.mean(final_values):.3f}, std={np.std(final_values):.3f}")
        else:
            print(f"Warning: File not found for digit {d}: {npy_path}")
    
    return loaded_data

# Your modified loop with saving functionality
def run_likelihood_analysis_with_saving(config, device, sde, score_model, inverse_scaler, 
                                       samples_per_digit=10, save_dir="likelihood_results"):
    """
    Run likelihood analysis for all digits, create separate plots, and save everything.
    
    Args:
        config: Your config dictionary
        device: Device to run on
        sde: SDE object
        score_model: Your trained model
        inverse_scaler: Inverse scaler function
        samples_per_digit: Number of samples to collect per digit
        save_dir: Directory to save results
    
    Returns:
        trajectories_by_digit: Dictionary of trajectories organized by digit
        saved_files: Information about saved files
    """
    
    print(f'Getting likelihood trajectories and saving results...')
    like_fn = get_likelihood_fn(sde, inverse_scaler=inverse_scaler)
    scaler = lambda x: x * 2 - 1.
    
    trajectories_by_digit = {}
    
    # Loop through each digit
    for digit in range(10):
        print(f'\nProcessing digit {digit}...')
        
        # Update config for this digit
        config['data.target_digit'] = digit
        config['data.dataset'] = 'SINGLE_DIGIT_MNIST'
        
        # Get new dataloader for this digit
        from datasets import get_dataset
        _, eval_loader, _ = get_dataset(config=config, uniform_dequantization=True, evaluation=True)
        
        trajectories_by_digit[digit] = []
        sample_count = 0
        
        # Collect samples for this digit
        for x, y in eval_loader:
            if sample_count >= samples_per_digit:
                break
                
            x = x.to(device)
            x = scaler(x)
            
            score_fn = get_score_fn(sde, score_model)
            bpd, z, nfe, logp_traj = like_fn(score_model, x)
            
            trajectories_by_digit[digit].append(logp_traj)
            sample_count += 1
            
            print(f'  Sample {sample_count}: BPD = {bpd:.4f}, NFE = {nfe}')
    
    # Create plots and save everything
    saved_files = plot_and_save_individual_digits(
        trajectories_by_digit, 
        save_dir=save_dir,
        title_prefix="Log Probability Trajectory"
    )
    
    print(f"\nAll results saved to directory: {save_dir}")
    return trajectories_by_digit, saved_files

# Example usage
if __name__ == "__main__":
    # Create example data with different lengths
    np.random.seed(42)
    
    # Simulate log probability trajectories of different lengths
    arrays = []
    arrays.append(np.cumsum(np.random.randn(100) * 0.1) - 50)  # 100 steps
    arrays.append(np.cumsum(np.random.randn(150) * 0.15) - 45)  # 150 steps
    arrays.append(np.cumsum(np.random.randn(80) * 0.12) - 55)   # 80 steps
    arrays.append(np.cumsum(np.random.randn(120) * 0.2) - 30)   # 120 steps (anomaly)
    arrays.append(np.cumsum(np.random.randn(90) * 0.08) - 48)   # 90 steps
    
    # Example 1: Basic plotting
    print("Example 1: Basic plotting")
    fig1, ax1 = plot_multiple_arrays(arrays, 
                                    labels=['Sample 1', 'Sample 2', 'Sample 3', 'Sample 4', 'Sample 5'])
    plt.show()
    
    # Example 2: Digit-specific plotting
    print("Example 2: Digit trajectories")
    digit_arrays = arrays[:5]
    digit_labels = [0, 1, 2, 8, 9]  # Digits corresponding to each array
    
    fig2, ax2 = plot_digit_trajectories(digit_arrays, digit_labels, 
                                       title="Log Probability Trajectories by Digit")
    plt.show()
    
    # Example 3: Normal vs Anomaly comparison
    print("Example 3: Normal vs Anomaly comparison")
    normal_arrays = arrays[:3]  # First 3 are normal
    anomaly_arrays = arrays[3:4]  # 4th is anomaly
    
    fig3, ax3 = plot_anomaly_comparison(normal_arrays, anomaly_arrays,
                                       normal_labels=['Digit 0', 'Digit 1', 'Digit 2'],
                                       anomaly_labels=['Digit 8'],
                                       title="Normal Digits vs Anomalous Digit 8")
    plt.show()
    
    # Example 4: Real usage scenario (like from your likelihood computation)
    print("Example 4: Real scenario example")
    
    # Simulate what you might get from your model
    digit_trajectories = {
        0: np.cumsum(np.random.randn(95) * 0.1) - 50,   # Normal
        1: np.cumsum(np.random.randn(103) * 0.12) - 48,  # Normal
        2: np.cumsum(np.random.randn(88) * 0.09) - 52,   # Normal
        8: np.cumsum(np.random.randn(110) * 0.25) - 25,  # Anomaly (higher variance, higher values)
        9: np.cumsum(np.random.randn(92) * 0.11) - 49,   # Normal
    }
    
    # Convert to lists for plotting
    arrays_list = list(digit_trajectories.values())
    digit_list = list(digit_trajectories.keys())
    
    fig4, ax4 = plot_digit_trajectories(arrays_list, digit_list,
                                       title="Likelihood Trajectories: Digit 8 Shows Anomalous Behavior")
    
    # Add annotation for the anomaly
    ax4.annotate('Anomalous Digit 8\n(Higher likelihood)', 
                xy=(50, -25), xytext=(70, -10),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=12, color='red', fontweight='bold')
    
    plt.show()
    
    print("\nTo use with your data:")
    print("arrays = [logp_traj_digit_0, logp_traj_digit_1, ..., logp_traj_digit_9]")
    print("digit_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]")
    print("fig, ax = plot_digit_trajectories(arrays, digit_labels)")
