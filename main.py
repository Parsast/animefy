import torch
from net.generator_pytorch import G_net
from net.discriminator_pytorch import D_net
from tools.data_loader_pytorch import get_dataloader, AnimeDataset
from tools.GuidedFilter_pytorch import guided_filter
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from ops_pytorch import *
import os
import cv2
import numpy as np
import time
import logging
import warnings

warnings.filterwarnings("ignore", message="")  

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class Trainer:
    def __init__(self,generator, discriminator,optimizer_g, optimizer_d , data_dir, save_dir, save_dir_val, image_size,
                 batch_size, num_workers, num_epochs, lr, beta1, beta2, save_interval, device, resume, resuming_epoch=None):
        self.generator = generator
        self.discriminator = discriminator
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.save_dir_val = save_dir_val
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.save_interval = save_interval
        self.device = device
        self.resume = resume
        self.resuming_epoch = resuming_epoch
        self.init_loss_functions()
        self.init_data_loaders()
    
    def init_loss_functions(self):
        self.con_loss_init = con_loss(device=self.device)
        self.style_loss_decentralization = style_loss_decentralization_3(device=self.device,weight=[0.1,2.0,28])
        self.region_smoothing_loss = region_smoothing_loss(device=self.device, weight=0.8)
        self.VGG_LOSS = VGG_LOSS(device=self.device)
        self.Lab_color_loss = color_loss(device=self.device, weight=0.8)
        self.total_variation_loss = total_variation_loss(device=self.device)
        self.generator_loss = generator_loss(device=self.device)
        self.discriminator_loss = discriminator_loss(device=self.device)
        self.discriminator_loss_346 = discriminator_loss_346(device=self.device)
    
    def init_data_loaders(self):
        self.real_image_loader = get_dataloader(os.path.join(self.data_dir,"train_photo"), self.image_size, self.batch_size, self.num_workers)
        self.anime_image_loader = get_dataloader(os.path.join(self.data_dir,"Hayao/smooth"), self.image_size, self.batch_size, self.num_workers)
        self.anime_smooth_generator = get_dataloader(os.path.join(self.data_dir,"Hayao/style"), self.image_size, self.batch_size, self.num_workers)
        self.val_image_loader = get_dataloader(os.path.join(self.data_dir,"val"), self.image_size, self.batch_size, self.num_workers)
    def train(self):
        
        writer = SummaryWriter('runs/experiment_1')
        starting_epoch = self.resuming_epoch if self.resume else 0
        
        for epoch in range(starting_epoch, self.num_epochs):
            epoch_start_time = time.time()
            if epoch < 10 and self.resume == False:
                for i, (real_images, anime_images) in enumerate(zip(self.real_image_loader, self.anime_image_loader)):
                    real_images = real_images[0].to(self.device)
                    generated_s, generated_m = self.generator(real_images)
                    generated_s = generated_s.to(self.device)
                    # Train generator
                    generated = tanh_out_scale(guided_filter(sigm_out_scale(generated_s).to(self.device),sigm_out_scale(generated_s).to(self.device),r=2,eps=1e-2))
                    generated_init_loss = self.con_loss_init(real_images, generated)
                    self.optimizer_g.zero_grad()
                    generated_init_loss.backward()
                    self.optimizer_g.step()
                logging.info(f"Epoch [{epoch}/{self.num_epochs}] [{i}/{len(self.real_image_loader)}] Loss: {generated_init_loss.item()}")
            else:
                
                for i, (real_images, anime_images) in enumerate(zip(self.real_image_loader, self.anime_image_loader)):
                    start_time = time.time() 
                    real_images = real_images[0].to(self.device)
                    generated_s, generated_m = self.generator(real_images)
                    generated_s = generated_s.to(self.device)
                    generated_m = generated_m.to(self.device)
                    # print(f"generated tensor values before tanh: {generated_s[0][0][0:10]}")
                    # print(f"generated tensor values before tanh: {generated_m[0][0][0:10]}")
                    
                    # # Train generator
                    generated = tanh_out_scale(guided_filter(sigm_out_scale(generated_s).to(self.device),sigm_out_scale(generated_s).to(self.device),r=2,eps=1e-2))
                    # print(f"generated tensor values after tanh: {generated[0][0][0:10]}")
                    # # gray mapping
                    fake_sty_gray = grayscale_to_rgb(rgb_to_grayscale(generated).to(self.device))
                    anime_sty_gray = grayscale_to_rgb(rgb_to_grayscale(anime_images[0].to(self.device)).to(self.device))
                    gray_anime_smooth = grayscale_to_rgb(rgb_to_grayscale(anime_images[1].to(self.device)).to(self.device))
                    # # support
                    fake_gray_logit = self.discriminator(fake_sty_gray)
                    anime_gray_logit = self.discriminator(anime_sty_gray)
                    gray_anime_smooth_logit = self.discriminator(gray_anime_smooth)
                    fake_superpixel = get_seg(generated.to('cpu').detach().numpy().reshape(self.batch_size,256,256,3))
                    fake_superpixel = torch.tensor(fake_superpixel).to(self.device)
                    fake_NLMean_l0 = get_NLMean_l0(generated_s.to('cpu').detach().numpy().reshape(self.batch_size,256,256,3))
                    fake_NLMean_l0 = torch.tensor(fake_NLMean_l0).to(self.device)
                    fake_superpixel = torch.tensor(fake_superpixel).to(self.device)
                    fake_NLMean_l0 = torch.tensor(fake_NLMean_l0).to(self.device)
                    fake_superpixel = fake_superpixel.permute(0,3,1,2)
                    fake_NLMean_l0 = fake_NLMean_l0.permute(0,3,1,2)
                    # main
                    generated_m_logit = self.discriminator(generated_m)
                    fake_NLMean_logit = self.discriminator(fake_NLMean_l0)
                    """ support"""
                    support_con_loss = self.con_loss_init(real_images, generated)
                    s22, s33, s44 = self.style_loss_decentralization(real_images, generated)
                    sty_loss = s22 + s33 + s44
                    rs_loss = self.region_smoothing_loss(fake_superpixel,generated) + 0.5 * self.VGG_LOSS(real_images[1],generated)
                    color_loss = self.Lab_color_loss(real_images,generated)
                    tv_loss = 0.0001 * self.total_variation_loss(generated)
                    g_adv_loss = self.generator_loss(fake_gray_logit)
                    
                    G_loss_support = support_con_loss + sty_loss + rs_loss + color_loss + g_adv_loss + tv_loss
                    D_loss_support = self.discriminator_loss(anime_gray_logit,fake_gray_logit) + self.discriminator_loss_346(gray_anime_smooth_logit)*5.
                    """ main """
                    tv_loss_main = 0.0001 * self.total_variation_loss(generated_m)
                    p4_loss_main =  self.VGG_LOSS(fake_NLMean_l0,generated_m)*0.5
                    p0_loss_main = L1_loss(fake_NLMean_l0,generated_m)*50.
                    g_m_loss = self.generator_loss(generated_m_logit) * 0.02
                    
                    G_loss_main = p0_loss_main + p4_loss_main + g_m_loss + tv_loss_main
                    D_loss_main = discriminator_loss_m(fake_NLMean_logit,generated_m_logit)*0.1
                    
                    Discriminator_loss = D_loss_support + D_loss_main
                    Generator_loss =  G_loss_main + G_loss_support
                    
                    Generator_loss.backward(retain_graph=True)
                    Discriminator_loss.backward(retain_graph=True)
                    for name, parameter in self.generator.named_parameters():
                        if  parameter.grad is not None:
                            writer.add_scalar(f'Generator Gradients/{name}', parameter.grad.norm(), epoch)
                    for name, parameter in self.discriminator.named_parameters():
                        if  parameter.grad is not None:
                            writer.add_scalar(f'Discrminator Gradients/{name}', parameter.grad.norm(), epoch)
                    
                    # torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
                    # torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                    

                    self.optimizer_g.step()
                    self.optimizer_d.step()
                
                    self.optimizer_d.zero_grad()
                    self.optimizer_g.zero_grad()
                    
                    self.generator.eval()
                    generated_s, generated_m = self.generator(real_images)
                    generated = tanh_out_scale(guided_filter(sigm_out_scale(generated_s).to(self.device),sigm_out_scale(generated_s).to(self.device),r=2,eps=1e-2))
                    print(f"generated tensor values after tanh: {generated[0][0][0:10]}")
                    self.generator.train()
                    step_time = time.time() - start_time
                    logging.info(f"Epoch {epoch}, Step {i}/{len(self.real_image_loader)}, Generator Loss: {Generator_loss.item()}, Time: {step_time}")
                    logging.info(f"Epoch {epoch}, Step {i}/{len(self.real_image_loader)}, Discriminator Loss: {Discriminator_loss.item()}, Time: {step_time}")

            epoch_time = time.time() - epoch_start_time
            # logging.info(f"Epoch [{epoch}/{self.num_epochs}] [{i}/{len(self.real_image_loader)}] Loss: {Generator_loss.item()} Time: {epoch_time}")      
            
            writer.add_graph(self.generator, real_images)
            writer.add_graph(self.discriminator, real_images)


            if epoch % self.save_interval == 0:
                self.save_model(epoch)
            
            self.save_val(epoch)
        writer.close()

    def save_model(self,epoch):
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save({
        'epoch': epoch,
        'generator_state_dict': self.generator.state_dict(),
        'discriminator_state_dict': self.discriminator.state_dict(),
        'optimizer_g_state_dict': self.optimizer_g.state_dict(),
        'optimizer_d_state_dict': self.optimizer_d.state_dict(),
        }, os.path.join(self.save_dir, f'checkpoint_{epoch}.pth'))
        logging.info(f"Model saved at epoch {epoch}")

    def load_model(self):
        checkpoint = torch.load(os.path.join(self.save_dir, f'checkpoint_{self.resuming_epoch}.pth'))
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        # Set the start epoch if it's stored in checkpoint
        self.resuming_epoch = checkpoint['epoch'] + 1  # Set the next epoch to start from
        logging.info(f"Model loaded from epoch {self.resuming_epoch}")
        return self.resuming_epoch  # Return the next epoch number
    

    
    def save_val(self,epoch):
        #generate validation images
        self.generator.eval()
        with torch.no_grad():
            for i, val_images in enumerate(self.val_image_loader):
                val_images = val_images[0].to(self.device)
                generated_s, generated_m = self.generator(val_images)
                generated_s = generated_s.to(self.device)
                generated_m = generated_m.to(self.device)
                generated = tanh_out_scale(guided_filter(sigm_out_scale(generated_s).to(self.device),sigm_out_scale(generated_s).to(self.device),r=2,eps=1e-2))
                actual_batch_size = val_images.size(0)
                for j in range(actual_batch_size):
                    os.makedirs(self.save_dir_val, exist_ok=True)
                    vutils.save_image(generated[j].detach(), os.path.join(self.save_dir_val, f'val_{i*self.batch_size+j}_epoch{epoch}_.png'))
                    # print(f"tensor values: {generated[j].detach()}")
                    vutils.save_image(val_images[j].detach(), os.path.join(self.save_dir_val, f'val_{i*self.batch_size+j}_real_epoch{epoch}_.png'))
                    vutils.save_image(generated_m[j].detach(), os.path.join(self.save_dir_val, f'val_{i*self.batch_size+j}_main_epoch{epoch}_.png'))
                    # print(f"tensor values: {generated_m[j].detach()}")
                    vutils.save_image(generated_s[j].detach(), os.path.join(self.save_dir_val, f'val_{i*self.batch_size+j}_support_epoch{epoch}_.png'))
        self.generator.train()
        logging.info(f"Validation images saved at epoch {epoch}")
        
def main():
    data_dir = '/Users/parsa/codes/animefy/dataset'
    save_dir = '/Users/parsa/codes/animefy/saved_models'
    save_dir_val = '/Users/parsa/codes/animefy/val_results'
    image_size = 256
    batch_size = 4
    num_workers = 4
    num_epochs = 60
    lr = 2e-8
    beta1 = 0.5
    beta2 = 0.999
    save_interval = 5
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    resume = True
    resume_epoch = 5
    generator = G_net(inputs_channels=3).to(device)
    discriminator = D_net(3,ch=3).to(device)
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0000002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0000002, betas=(0.5, 0.999))
    trainer = Trainer(generator, discriminator, optimizer_g, optimizer_d, data_dir, save_dir, save_dir_val, image_size,
                 batch_size, num_workers, num_epochs, lr, beta1, beta2, save_interval, device, resume, resuming_epoch=resume_epoch)
    # trainer.train()
    
    # Begin training for resuming_epoch
    # next_epoch = trainer.load_model() if resume else 0  # Load the model if resuming
    trainer.train()  


if __name__ == '__main__':
    main()