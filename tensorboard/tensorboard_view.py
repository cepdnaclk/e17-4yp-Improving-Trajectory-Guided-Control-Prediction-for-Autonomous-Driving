from ..TCP.model import TCP
from ..TCP.config import GlobalConfig
from ..TCP.data import CARLA_Data

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from itertools import islice

config = GlobalConfig()
train_set = CARLA_Data(root=config.root_dir_all, data_folders=config.train_data, img_aug = config.img_aug)
dataloader_train = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=8)

dataiter = iter(dataloader_train)
batch =  next(dataiter)

front_img = batch['front_img']
speed = batch['speed'].to(dtype=torch.float32).view(-1,1) / 12.
target_point = batch['target_point'].to(dtype=torch.float32)
command = batch['target_command']
state = torch.cat([speed, target_point, command], 1)

model = TCP(config)

try:
    writer = SummaryWriter('runs/TCP_experiment')
    writer.add_graph(model, (front_img, state, target_point))
    writer.close()

except Exception as e:
    error_message = str(e)

    with open('error.txt', 'w') as file:
        file.write(error_message)

