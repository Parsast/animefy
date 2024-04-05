import torch
from generator_pytorch import G_net
from discriminator_pytorch import D_net
from tools.data_loader_pytorch import get_dataloader, AnimeDataset
from tools.GuidedFilter_pytorch import guided_filter
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision import transforms
from PIL import Image
from ops_pytorch import *
import os
import cv2
import numpy as np
import time


def check_nan(x, name="Tensor"):
    if torch.isnan(x).any():
        print(f"NaN detected in {name}")



if __name__ == '__main__':
    
    real_image_dir = '/Users/parsa/codes/animefy/dataset/train_photo'
    anime_image_dir = '/Users/parsa/codes/animefy/dataset/Hayao/smooth'
    anime_smooth_dir = '/Users/parsa/codes/animefy/dataset/Hayao/style'
    val_image_dir = '/Users/parsa/codes/animefy/dataset/val'
    save_dir = '/Users/parsa/codes/animefy/saved_models'
    save_dir_val = '/Users/parsa/codes/animefy/val_results'
    image_size = 256
    batch_size = 4
    num_workers = 4
    num_epochs = 60
    lr = 1e-4
    beta1 = 0.5
    beta2 = 0.999
    save_interval = 5
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # Data loaders
    real_image_loader = get_dataloader(real_image_dir, image_size, batch_size, num_workers) # DataLoader for real images
    anime_image_loader = get_dataloader(anime_image_dir, image_size, batch_size, num_workers) # DataLoader for anime images
    anime_smooth_generator = get_dataloader(anime_smooth_dir, image_size, batch_size, num_workers) # DataLoader for anime images with smoothing

    # Models and optimizers
    # generator_init = G_net(inputs_channels=3).to(device)
    generator = G_net(inputs_channels=3).to(device)
    discriminator = D_net(3,ch=3).to(device)
    # opttimizer_g_init = optim.Adam(generator_init.parameters(), lr=0.0002)
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002)

    con_loss_init = con_loss(device=device)
    style_loss_decentralization_3 = style_loss_decentralization_3(device=device,weight=[0.1,2.0,28])
    region_smoothing_loss = region_smoothing_loss(device=device)
    VGG_LOSS = VGG_LOSS(device=device)
    Lab_color_loss = color_loss(device=device)
    total_variation_loss = total_variation_loss(device=device)
    generator_loss = generator_loss(device=device)
    discriminator_loss = discriminator_loss(device=device)
    discriminator_loss_346 = discriminator_loss_346(device=device)


    for epoch in range(num_epochs):
        
        
        if epoch < 5:
            
            for i, (real_images, anime_images) in enumerate(zip(real_image_loader, anime_image_loader)):
                real_images = real_images[0].to(device)
                generated_s, generated_m = generator(real_images)
                generated_s = generated_s.to(device)
                # Train generator
                generated = tanh_out_scale(guided_filter(sigm_out_scale(generated_s).to(device),sigm_out_scale(generated_s).to(device),r=2,eps=1e-2))
                generated_init_loss = con_loss_init(real_images, generated)
                optimizer_g.zero_grad()
                generated_init_loss.backward()
                optimizer_g.step()
                generator.eval()
            
            print(f"Epoch [{epoch}/{num_epochs}] [{i}/{len(real_image_loader)}] Loss: {generated_init_loss.item()}")

        else:
            # torch.autograd.set_detect_anomaly(True)
            for i, (real_images, anime_images) in enumerate(zip(real_image_loader, anime_image_loader)):
                start_time = time.time() 
                real_images = real_images[0].to(device)
                generated_s, generated_m = generator(real_images)
                generated_s = generated_s.to(device)
                generated_m = generated_m.to(device)
                # # Train generator
                generated = tanh_out_scale(guided_filter(sigm_out_scale(generated_s).to(device),sigm_out_scale(generated_s).to(device),r=2,eps=1e-2))
                
                # # gray mapping
                fake_sty_gray = grayscale_to_rgb(rgb_to_grayscale(generated_s).to(device))
                anime_sty_gray = grayscale_to_rgb(rgb_to_grayscale(anime_images[0].to(device)).to(device))
                gray_anime_smooth = grayscale_to_rgb(rgb_to_grayscale(anime_images[1].to(device)).to(device))
                
                # # support
                fake_gray_logit = discriminator(fake_sty_gray)
                anime_gray_logit = discriminator(anime_sty_gray)
                gray_anime_smooth_logit = discriminator(gray_anime_smooth)

                
                fake_superpixel = get_seg(generated.to('cpu').detach().numpy().reshape(batch_size,256,256,3))
                fake_superpixel = torch.tensor(fake_superpixel).to(device)
                fake_NLMean_l0 = get_NLMean_l0(generated_s.to('cpu').detach().numpy().reshape(batch_size,256,256,3))
                fake_NLMean_l0 = torch.tensor(fake_NLMean_l0).to(device)
                
                
                fake_superpixel = torch.tensor(fake_superpixel).to(device)
                fake_NLMean_l0 = torch.tensor(fake_NLMean_l0).to(device)
                fake_superpixel = fake_superpixel.permute(0,3,1,2)
                fake_NLMean_l0 = fake_NLMean_l0.permute(0,3,1,2)
               

                # main
                generated_m_logit = discriminator(generated_m)
                fake_NLMean_logit = discriminator(fake_NLMean_l0)
                

                """ support"""

                support_con_loss = con_loss_init(real_images, generated)
                s22, s33, s44 = style_loss_decentralization_3(anime_sty_gray,fake_sty_gray)
                sty_loss = s22 + s33 + s44
                rs_loss = region_smoothing_loss(fake_superpixel,generated) + 0.5 * VGG_LOSS(real_images[1],generated)
                color_loss = Lab_color_loss(real_images,generated)
                tv_loss = 0.01 * total_variation_loss(generated)
                g_adv_loss = generator_loss(fake_gray_logit)
                G_loss_support = support_con_loss + sty_loss + rs_loss + color_loss + g_adv_loss + tv_loss
                D_loss_support = discriminator_loss(anime_gray_logit,fake_gray_logit) + discriminator_loss_346(gray_anime_smooth_logit)*5.
                
                """ main """
                tv_loss_main = 0.01 * total_variation_loss(generated)
                p4_loss_main =  VGG_LOSS(fake_NLMean_l0,generated_m)*0.5
                p0_loss_main = L1_loss(fake_NLMean_l0,generated_m)*50.
                g_m_loss = generator_loss(generated_m_logit) * 0.02
                G_loss_main = p0_loss_main + p4_loss_main + g_m_loss + tv_loss_main
                D_loss_main = discriminator_loss_m(fake_NLMean_logit,generated_m_logit)*0.1


                Discriminator_loss = D_loss_support + D_loss_main
                Generator_loss =  G_loss_main + G_loss_support

                Generator_loss.backward(retain_graph=True)
                Discriminator_loss.backward(retain_graph=True)
                
                optimizer_g.step()
                optimizer_d.step()

                optimizer_d.zero_grad()
                optimizer_g.zero_grad()

                step_time = time.time() - start_time
                print(f"Epoch [{epoch}/{num_epochs}] [{i}/{len(real_image_loader)}] Loss: {Generator_loss.item()} Time: {step_time}")
                print(f"Epoch [{epoch}/{num_epochs}] [{i}/{len(real_image_loader)}] Loss: {Discriminator_loss.item()} Time: {step_time}")
        if epoch % save_interval == 0:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(generator.state_dict(), os.path.join(save_dir, f'generator_{epoch}.pth'))
            torch.save(discriminator.state_dict(), os.path.join(save_dir, f'discriminator_{epoch}.pth'))
        if epoch > 5:
            #generate validation images
            generator.eval()
            with torch.no_grad():
                val_image_loader = get_dataloader(val_image_dir, image_size, batch_size, num_workers)
                for i, val_images in enumerate(val_image_loader):
                    val_images = val_images[0].to(device)
                    generated_s, generated_m = generator(val_images)
                    generated_s = generated_s.to(device)
                    generated_m = generated_m.to(device)
                    generated = tanh_out_scale(guided_filter(sigm_out_scale(generated_s).to(device),sigm_out_scale(generated_s).to(device),r=2,eps=1e-2))
                    
                    for j in range(batch_size):
                        os.makedirs(save_dir_val, exist_ok=True)
                        vutils.save_image(generated[j].detach(), os.path.join(save_dir_val, f'val_{i*batch_size+j}_{epoch}.png'))
                        vutils.save_image(val_images[j].detach(), os.path.join(save_dir_val, f'val_{i*batch_size+j}_{epoch}_real.png'))
                        vutils.save_image(generated_m[j].detach(), os.path.join(save_dir_val, f'val_{i*batch_size+j}_{epoch}_main.png'))
                        vutils.save_image(generated_s[j].detach(), os.path.join(save_dir_val, f'val_{i*batch_size+j}_{epoch}_support.png'))
            generator.train()