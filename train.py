import torch
from dataset import Dataset_AB 
import sys
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


from torchvision.utils import save_image 

import contextlib                                   #for conditionally running with blocks
import csv
from tqdm import tqdm 
import torch
import torch.nn as nn 
import config

from gen_disc_classes import Generator, Discriminator

# > > > SAVE and LOAD Checkpoint Functions > > > 
def save_checkpoint(model,optimizer, filename = "checkpoint.pth.tar"):
    print("Checkpoint is being saved . . .")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint,filename)

def load_checkpoint(checkpoint_file,model,optimizer,lr):
    print("Checkpoint is being loaded . . . ")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
# < < < SAVE and LOAD Checkpoint Functions < < <  


# - - - MAIN TRAIN FUNCTION - - -
def train_fn(discriminator_A, discriminator_B, generator_A, generator_B, loader, opt_discriminator, opt_generator, L1, mse, dis_scaler, gen_scaler, generator_losses, discriminator_A_losses, discriminator_B_losses, identity_losses, cycle_A_losses):
    loop  = tqdm(loader, leave = True) #to see progress bar 
    
    #loop through the dataloader to get batche sof images
    for idx, (imgA,imgB) in enumerate(loop):
        #transfer images to DEVICE GPU
        imgA = imgA.to(config.DEVICE)
        imgB = imgB.to(config.DEVICE)

        #DISCRIMINATOR TRAINIGN
        #Enable automatic mixed precision (AMP) training for MacOS GPU   
        with (torch.cuda.amp.autocast() if config.DEVICE == "cuda" else contextlib.nullcontext()):
        # > > first discriminator 
            #generate fake image A, send imageB to genA
            fake_imgA = generator_A(imgB)                       
            disc_A_real = discriminator_A(imgA)
            #Prevent gradients from flowing back into the generator during the backpropagation step for the discriminator
            disc_A_fake = discriminator_A(fake_imgA.detach())   #send fake image A
            disc_A_real_loss = mse(disc_A_real,torch.ones_like(disc_A_real))
            disc_A_fake_loss = mse(disc_A_fake,torch.zeros_like(disc_A_fake))
            disc_A_loss = disc_A_real_loss  + disc_A_fake_loss; 
        # < < first discriminator  
        # > > second discriminator 
            #generate fake image B, send imageA to genB
            fake_imgB = generator_B(imgA)                      
            disc_B_real = discriminator_B(imgB)
            disc_B_fake = discriminator_B(fake_imgB.detach())   #send fake image B
            disc_B_real_loss = mse(disc_B_real,torch.ones_like(disc_B_real))
            disc_B_fake_loss = mse(disc_B_fake,torch.zeros_like(disc_B_fake))
            disc_B_loss = disc_B_real_loss  + disc_B_fake_loss; 
        # < < second discriminator  
        
        #calculate the total discriminator loss
        disc_loss = (disc_A_loss + disc_B_loss)/2

        if config.DEVICE == 'cuda' or dis_scaler is not None:
            #scales loss and computes gradients as per original paper
            dis_scaler.scale(disc_loss).backward()     
            dis_scaler.step(opt_discriminator)
            dis_scaler.update()
        else:
            #scales loss and computes gradients 
            disc_loss.backward()                        
            opt_discriminator.step()

        #GENERATOR TRAINING
        with (torch.cuda.amp.autocast() if config.DEVICE == "cuda" else contextlib.nullcontext()):
            disc_A_fake = discriminator_A(fake_imgA)                            #gen wants to fool discriminator
            disc_B_fake = discriminator_B(fake_imgB)
            #calculate adversarial loss 
            loss_gen_A = mse(disc_A_fake,torch.ones_like(disc_A_fake))
            loss_gen_B = mse(disc_B_fake,torch.ones_like(disc_B_fake))
            #calculate cycle loss: take fake A and generate B out of it
            cycle_B = generator_B(fake_imgA)
            cycle_A = generator_A(fake_imgB) 
            cycle_A_loss = L1(imgA,cycle_A)
            cycle_B_loss = L1(imgB,cycle_B)
            #calculate identity loss 
            identity_A = generator_A(imgA)
            identity_B = generator_B(imgB)
            identity_A_loss = L1(imgA,identity_A)
            identity_B_loss = L1(imgB,identity_B)

            #total loss generators
            generator_loss = (
                loss_gen_A + loss_gen_B
                + cycle_A_loss * config.LAMBDA_CYCLE
                + cycle_B_loss * config.LAMBDA_CYCLE
                + identity_A_loss * config.LAMBDA_IDENTITY
                + identity_B_loss * config.LAMBDA_IDENTITY
            )

            #append the calculated losses 
            generator_losses.append(generator_loss.item())
            discriminator_A_losses.append(disc_A_loss.item())
            discriminator_B_losses.append(disc_B_loss.item())
            identity_losses.append((identity_A_loss.item() + identity_B_loss.item()) / 2)
            cycle_A_losses.append(cycle_A_loss.item())
                
            opt_generator.zero_grad()


            if config.DEVICE == 'cuda' or gen_scaler is not None:
                gen_scaler.scale(generator_loss).backward()
                gen_scaler.step(opt_generator)
                gen_scaler.update()
            else:
                generator_loss.backward()
                opt_generator.step()
        
            #double check this code
            if idx % 100 == 0:
                #add 0.5 to do inverse of what transform normalization did to image function 
                save_image(fake_imgA*0.5 + 0.5, f"{config.SAVE_DIR}/imgA_{idx}.png")       
                save_image(fake_imgB*0.5 + 0.5, f"{config.SAVE_DIR}/imgB_{idx}.png")

#MAIN FUNCTION 
def main():
# > > > initialization
    #instantiate Discriminators A and B - Default num residual layers is 9 for 256
    discriminator_A = Discriminator(in_channels=3).to(config.DEVICE)
    discriminator_B = Discriminator(in_channels=3).to(config.DEVICE)
    
    #instantiate Generator A and B  - RGB - Default num residual layers is 9 for 256
    generator_A = Generator(img_channels=3).to(config.DEVICE)
    generator_B = Generator(img_channels=3).to(config.DEVICE)

    #vectors to hold the losses values and log them later 
    generator_losses = []
    discriminator_A_losses = []
    discriminator_B_losses = []
    identity_losses = []
    cycle_A_losses = []
    

#initialize generator and discriminator optimizers -  Using ADAM
    opt_discriminator = optim.Adam( 
        list(discriminator_A.parameters()) + list(discriminator_B.parameters()),
        lr = config.LEARNING_RATE,
        betas = (0.5,0.999),            #beta terms for momentum and beta 2 specified in the paper
    )
    opt_generator = optim.Adam(
        list(generator_A.parameters()) + list(generator_B.parameters()),
        lr = config.LEARNING_RATE,
        betas = (0.5,0.999),            #beta terms for momentum and beta 2 specified in the paper
    )

#initialize loss functions 
    L1 = nn.L1Loss()                    #identity and cycle consistency loss
    mse = nn.MSELoss()                  #adversarial loss using mean square error 

#Load existing model if LOAD_MODEL = TRUE
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN_A,generator_A,opt_generator, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_GEN_B,generator_B,opt_generator, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC_A,discriminator_A,opt_generator, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC_B,discriminator_B,opt_generator, config.LEARNING_RATE)
    pass 

#initialize Dataset class, send paths of training images for each domain
    dataset = Dataset_AB(root_domainA= config.TRAIN_DIR + "/trainA",root_domainB= config.TRAIN_DIR + "/trainB")
    
#Instantiate a DataLoader from Pytorch
    loader = DataLoader(dataset, batch_size = config.BATCH_SIZE,shuffle = True, num_workers=config.NUM_WORKERS, pin_memory = True)
    
    #Check if there is CUDA 
    if config.DEVICE == 'cuda':
        gen_scaler = torch.cuda.amp.GradScaler()
        dis_scaler = torch.cuda.amp.GradScaler()
    else:
        gen_scaler = None
        dis_scaler = None

#Main loop training runs according to the number of epochs 
    for epoch in range(config.NUM_EPOCHS):
    #call the train function and send instantiations of gen, disc, data loader, optimizer, loss functions and scalers
        train_fn(discriminator_A,discriminator_B,generator_A,generator_B,loader,opt_discriminator,opt_generator,L1,mse,dis_scaler,gen_scaler,generator_losses,discriminator_A_losses, discriminator_B_losses,identity_losses, cycle_A_losses)

        if config.SAVE_MODEL:
            save_checkpoint(generator_A, opt_generator, filename=config.CHECKPOINT_GEN_A)
            save_checkpoint(generator_B, opt_generator, filename=config.CHECKPOINT_GEN_B)
            save_checkpoint(discriminator_A, opt_generator, filename=config.CHECKPOINT_DISC_A)
            save_checkpoint(discriminator_B, opt_generator, filename=config.CHECKPOINT_DISC_B)
        
        #save loss values 
        save_losses_to_file(zip(generator_losses, discriminator_A_losses, discriminator_B_losses, identity_losses, cycle_A_losses), "losses.csv")    
     

def save_losses_to_file(losses, filename):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Generator Loss", "Discriminator A Loss", "Discriminator B Loss", "Identity Loss", "Cycle Loss"])
        for row in losses:
            writer.writerow(row)


#run the train.py 
if __name__ == "__main__":
    main()

