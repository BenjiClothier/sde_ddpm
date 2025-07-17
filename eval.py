import os
import numpy as np

import torch

from sde import SubVPSDE, get_score_fn, EulerMaruayamaPredictor, sample_images
from unet import Unet
from utils import get_loaders, make_im_grid
from likelihood import get_likelihood_fn
from datasets import get_dataset

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
    'eval.batch_size' : 1,
    'dataset' : 'MNIST'
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

for x, _ in eval_loader:
    x = x.to(device)
    x = scaler(x)

    score_fn = get_score_fn(sde, score_model)

    bpd, z, nfe, logp_traj = like_fn(score_model, x)
    print(bpd)
    print(logp_traj)
    break
  







