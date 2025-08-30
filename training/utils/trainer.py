import torch
from torch.utils.data import DataLoader
from data import dataset
import os, sys
import json
from torch.utils import tensorboard
from utils import utilities
from tqdm import tqdm



sys.path.append('..')
from models import deep_equilibrium
from models import utils as models_utils

sys.path.insert(0,"../utils/")
from training.utils.utils import build_model

class Trainer:
    """
    """
    def __init__(self, config, seed, device):
        self.config = config
        self.seed = seed

        self.device = device
        self.noise_val = config['noise_val']
        self.noise_range = config['noise_range']
        self.valid_epoch_num = 0

        # Datasets
        train_dataset = dataset.H5PY(config['train_dataloader']['train_data_file'])
        val_dataset = dataset.H5PY(config['val_dataloader']['val_data_file'])
        bsd68 = dataset.H5PY('data/BSD/test.h5')
       
        # Dataloaders
        print('Preparing the dataloaders')
        self.batch_size = config["train_dataloader"]["batch_size"]

        self.train_dataloader = DataLoader(train_dataset, batch_size=config["train_dataloader"]["batch_size"], shuffle=True, num_workers=config["train_dataloader"]["num_workers"], drop_last=True)
        
        self.val_dataloader = DataLoader(val_dataset, batch_size=config["val_dataloader"]["batch_size"], shuffle=False, num_workers=config["val_dataloader"]["num_workers"])
        self.bsd68 = DataLoader(bsd68, batch_size=1, shuffle=False, num_workers=1)

        # Build the model
        print('Building the model')
        self.model, self.config = build_model(self.config)
        self.model = self.model.to(device)

        


        print(self.model)
        print("Number of parameters in the model: ", self.model.num_params)
        self.config["number_of_parameters"] = self.model.num_params

        # self.model.conv_layer.check_tranpose()
        # Set up the optimizer
        self.set_optimization()

        # Set the DEQ solver
        self.denoise = deep_equilibrium.DEQFixedPoint(self.model, self.config["training_options"]["fixed_point_solver_fw_params"], self.config["training_options"]["fixed_point_solver_bw_params"])

        # Loss
        self.criterion = torch.nn.L1Loss(reduction='mean')

       

        # CHECKPOINTS & TENSOBOARD
        self.checkpoint_dir = os.path.join(config["logging_info"]['log_dir'], config["exp_name"], 'checkpoints')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        config_save_path = os.path.join(config["logging_info"]['log_dir'], config["exp_name"], f'config.json')
        with open(config_save_path, 'w') as handle:
            json.dump(getattr(self, f"config"), handle, indent=4, sort_keys=True)

        writer_dir = os.path.join(config["logging_info"]['log_dir'], config["exp_name"], 'tensorboard_logs')
        self.writer = tensorboard.SummaryWriter(writer_dir)

    def set_optimization(self):

        """ """
        # optimizer
        optimizer = torch.optim.Adam

        self.optimizers = []
        params_dicts = []
        model = self.model
    
        lr = self.config["optimization"]["lr"]

        # 1 - conv parameters
        params_conv = model.conv_layer.parameters()
        params_dicts.append({"params": params_conv, "lr": lr["conv"]})
  
        # 2 - spline activations
        params_dicts.append({"params": model.nonlinearity.parameters(), "lr": lr["nonlinearity"]})

        # 3 - spline mu
        params_dicts.append({"params": [model.mu_], "lr": lr["mu"]})

        # 4 - spline scaling
        params_dicts.append({"params": [model.spline_scaling.coefficients], "lr": lr["spline_scaling"]})

        self.optimizer = optimizer(params_dicts)

        # scheduler
        if self.config["training_options"]["scheduler"]["use"]:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=self.config["training_options"]["scheduler"]["gamma"], verbose=True)


    def train(self):
        self.batch_seen = 0
        while self.batch_seen < self.config["training_options"]["n_batches"]:
            # train epoch
            self.train_epoch()

        self.writer.flush()
        self.writer.close()

        

    def train_epoch(self):
        """
        """
        self.model.train()

        tbar = tqdm(self.train_dataloader, ncols=80, position=0, leave=True)
      

        log = {}

        for batch_idx, data in enumerate(tbar):
            self.batch_seen += 1

            # validation / save checkpoints / logs / scheduler...
            if self.batch_seen % self.config["logging_info"]["log_batch"] == 0:
                self.valid_epoch()
                self.model.train()
                # SAVE CHECKPOINT
                self.save_checkpoint(self.batch_seen)


            if self.config["training_options"]["scheduler"]["use"] and self.batch_seen % self.config["training_options"]["scheduler"]["n_batch"] == 0:
                # check n steps
                nsteps = (self.batch_seen // self.config["training_options"]["scheduler"]["n_batch"])
                if nsteps <= self.config["training_options"]["scheduler"]["nb_steps"]:
                    self.scheduler.step()


            if self.batch_seen > self.config["training_options"]["n_batches"]:
                break

            data = data.to(self.device)

            sigma = torch.torch.empty((data.shape[0], 1, 1, 1, 1), device=data.device).uniform_(self.noise_range[0], self.noise_range[1])

            noise = sigma/255 * torch.randn(data.shape,device=data.device)
    
            noisy_data = data + noise

            self.optimizer.zero_grad()
            # estimate fixed point
            output = self.denoise(noisy_data, sigma = sigma)

            loss = self.criterion(output, data)
  
            loss.backward()

            self.optimizer.step()
     

            log['loss'] = loss.item()
            log['forward_mean_iter'] = self.denoise.forward_niter_mean
            log['forward_max_iter'] = self.denoise.forward_niter_max
            log['backward_iter'] = self.denoise.backward_niter
            log['conv_spectral_norm'] = self.model.conv_layer.L
 

            self.wrt_step = self.batch_seen
            self.write_scalars_tb(log)

            tbar.set_description(f"T ({self.valid_epoch_num}) | TotalLoss {log['loss']:.7f}")

        return log

    
    def optimizer_step(self):
        """ """
        for i in range(len(self.optimizers)):
            self.optimizers[i].step()


    def valid_epoch(self):
        self.valid_epoch_num += 1
        self.model.eval()

        psnr_val = len(self.noise_val)*[0.0]

        tbar_val = tqdm(self.val_dataloader, ncols=40, position=0, leave=True)
        
        with torch.no_grad():
                for batch_idx, data in enumerate(tbar_val):
                    data = data.to(self.device)
                    for i, sigma in enumerate(self.noise_val):
                        sigma = sigma*torch.ones(1, 1, 1, 1, 1, device=self.device)
                        noisy_data = data + sigma / 255 * torch.randn(data.shape, device=data.device)
                        output = self.denoise(noisy_data, sigma=sigma)
                        out_val = torch.clamp(output, 0., 1.)
                        psnr_val[i] = psnr_val[i] + utilities.batch_PSNR(out_val, data, 1.)/len(self.val_dataloader)

        epoch = self.valid_epoch_num
        self.writer.add_image('image_xy', out_val[0,0,:,:,out_val.shape[4]//2], epoch, dataformats='HW')
        self.writer.add_image('image_xz', out_val[0,0,:,out_val.shape[3]//2,:], epoch, dataformats='HW')
        self.writer.add_image('image_yz', out_val[0,0,out_val.shape[2]//2,:,:], epoch, dataformats='HW')

        psnr_bsd = len(self.noise_val)*[0.0]

        tbar_val = tqdm(self.bsd68, ncols=40, position=0, leave=True)
        with torch.no_grad():
                for batch_idx, data in enumerate(tbar_val):
                    data = data.to(self.device)
                    for i, sigma in enumerate(self.noise_val):
                        sigma = sigma*torch.ones(1, 1, 1, 1, device=self.device)
                        noisy_data = data + sigma / 255 * torch.randn(data.shape,device=data.device)
                        noisy_data = noisy_data.unsqueeze(4).repeat(1, 1, 1, 1, 8)
                        output = self.denoise(noisy_data, sigma=sigma)
                        output = output.mean(dim=4)

                        out_val = torch.clamp(output, 0., 1.)
                        psnr_bsd[i] = psnr_bsd[i] + utilities.batch_PSNR(out_val, data, 1.).mean().item() / 68

        # METRICS TO TENSORBOARD
        self.wrt_mode = 'Convolutional'
        for i in range(len(self.noise_val)):
            self.writer.add_scalar(f'{self.wrt_mode}/Validation PSNR sigma={self.noise_val[i]}', psnr_val[i], epoch)
            self.writer.add_scalar(f'{self.wrt_mode}/Validation PSNR BSD68 sigma={self.noise_val[i]}', psnr_bsd[i], epoch)

    def write_scalars_tb(self, logs):
        for k, v in logs.items():
            self.writer.add_scalar(f'Convolutional/Training {k}', v, self.wrt_step)

    def save_checkpoint(self, epoch):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'L': self.model.conv_layer.L
        }

        state['optimizer_state_dict'] = self.optimizer.state_dict()

        print('Saving a checkpoint:')
        # CHECKPOINTS & TENSOBOARD
        self.checkpoint_dir = os.path.join(self.config["logging_info"]['log_dir'], self.config["exp_name"], 'checkpoints')

        filename = self.checkpoint_dir + '/checkpoint_' + str(epoch) + '.pth'
        torch.save(state, filename)
