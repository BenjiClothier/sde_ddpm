import os
import numpy as np

import torch
import matplotlib.pyplot as plt

from sde import SubVPSDE, get_score_fn, EulerMaruayamaPredictor, sample_images
from unet import Unet
from utils import get_loaders, make_im_grid
from likelihood import get_likelihood_fn
from datasets import get_dataset
from plot_arrays import plot_multiple_arrays, plot_digit_trajectories_grouped, plot_digit_trajectories_with_stats, plot_and_save_individual_digits

torch.manual_seed(159753)
np.random.seed(159753)

#Config
config = {
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
    'ODE' : True,
    'dequant' : True,
    'data.dataset': 'SINGLE_DIGIT_MNIST',
    'data.target_digit': 0,
    'eval.batch_size': 1,
    'data.image_size': 32
}

#Device
device = 'cuda:0'

#Model
model = Unet().to(device)
if config['use_ema']:
    ema_model = torch.optim.swa_utils.AveragedModel(
        model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999)
    )

ema_state_dict = torch.load('./checkpoints/ema_chk_200.pt', map_location=device)


cleaned_state_dict = {}
for key, value in ema_state_dict.items():
    if key.startswith('module.'):
        cleaned_state_dict[key[7:]] = value  # Remove 'module.' prefix
    elif key != 'n_averaged':  # Skip the n_averaged key
        cleaned_state_dict[key] = value

ema_model.load_state_dict(ema_state_dict)
ema_model.eval()   

#SDE
sde = SubVPSDE(config)

#Sampler
sampler = EulerMaruayamaPredictor()

#Center the data [-1, 1]
scaler = lambda x: x * 2 - 1. 
inverse_scaler = lambda x: (x + 1) / 2.

#Get test data
_, eval_loader, info = get_dataset(config=config, uniform_dequantization=True, evaluation=True)

#Get a sample
# with torch.no_grad():
#     print(f'Generating samples')
#     score_model = model if not config['use_ema'] else ema_model
#     score_fn = get_score_fn(sde, score_model)
#     gen_x = sample_images(sde, score_fn, (64, 3, 32, 32), device=device, predictor=sampler)
#     image = make_im_grid(gen_x, (8, 8))
#     image.save(f'samples/test.png')
score_model = model if not config['use_ema'] else ema_model
print(f'Getting likelihood trajectory...')
like_fn = get_likelihood_fn(sde, inverse_scaler=inverse_scaler)
# i = 0
# logps = []
# for k in range(9):
#     config['data.target_digit'] = k
#     for x, _ in eval_loader:
#         x = x.to(device)
#         x = scaler(x)

#         score_fn = get_score_fn(sde, score_model)

#         bpd, z, nfe, logp_traj = like_fn(score_model, x)
#         # print(bpd)
#         # print(logp_traj)
#         logps.append(logp_traj)
#         i += 1
#         if i == 9:
#             break
#     fig, ax = plot_multiple_arrays(logps)
#     plt.show() 

trajectories_by_digit = {}
    
    # Loop through each digit
for digit in range(10):
    print(f'Processing digit {digit}...')
    
    # Update config for this digit
    config['data.target_digit'] = digit
    config['data.dataset'] = 'SINGLE_DIGIT_MNIST'  # Use single digit dataset
    
    # Get new dataloader for this digit
    from datasets import get_dataset  # Import your get_dataset function
    _, eval_loader, _ = get_dataset(config=config, uniform_dequantization=True, evaluation=True)
    
    trajectories_by_digit[digit] = []
    sample_count = 0
    
    # Collect samples for this digit
    for x, y in eval_loader:
        if sample_count > 10:
            break
            
        x = x.to(device)
        x = scaler(x)
        
        score_fn = get_score_fn(sde, score_model)
        bpd, z, nfe, logp_traj = like_fn(score_model, x)
        
        trajectories_by_digit[digit].append(logp_traj)
        sample_count += 1
        
        print(f'  Digit {digit}, Sample {sample_count}')

saved_files = plot_and_save_individual_digits(
    trajectories_by_digit, 
    save_dir="likelihood_results",
    title_prefix="Log Probability Trajectory"
)
plt.show()  







