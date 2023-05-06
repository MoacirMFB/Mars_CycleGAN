import torch
import torch.nn as nn

# Parameters for the discriminator model
KERNEL_SIZE_D = 4
PADDING = 1 
PADDING_MODE = "reflect"
OUT_CHANNELS = [64, 128, 256, 512]       #list of channels per layer for disc, as per paper

# > > > some parameters to be used that are defined in the original paper for the Generator
KERNEL_SIZE_G = 3
PADDING_RES = 1 
PADDING_GEN = 3 
PADDING_MODE = "reflect"
DOWNSAMPLE = True
NUM_RESIDUALS = 9               #6 if it is 128 or smaller
NUM_FEATURES = 64
# < < < some parameters to be used that are defined in the original paper  for the Generator


# > > > > GENERIC CLASSES FOR NEURAL NETWORK DEFINITION > > > > 

# Block class inherits from the Module Class in Torch
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        # Initialize parent class attributes for the Block class
        super().__init__()
        
        # Define the sequential block structure
        self.conv = nn.Sequential(
            # Conv2D layer with specified parameters and "reflect" padding mode to reduce artifacts
            nn.Conv2d(in_channels, out_channels, KERNEL_SIZE_D, stride, PADDING, bias=True, padding_mode=PADDING_MODE),
            # Instance normalization helps stabilize training and improve generated images' quality
            nn.InstanceNorm2d(out_channels),
            # LeakyReLU activation with a negative slope of 0.2
            nn.LeakyReLU(0.2),
        )

    # Forward pass for the Block class
    def forward(self, x):
        return self.conv(x)
# ConvBlock: a custom block containing a 2D convolutional layer or a transposed convolutional layer for ups
class ConvBlock(nn.Module):
        
    def __init__(self, in_channels, out_channels, down=DOWNSAMPLE, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode=PADDING_MODE, **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            # instance normalization and a ReLU activation function as per original source paper
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )

    def forward(self, input_tensor):
        return self.conv(input_tensor)
    

# ResidualBlock: a custom block consisting of two ConvBlocks - does not change num of inpt chan
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=KERNEL_SIZE_G, padding=PADDING_RES),
            ConvBlock(channels, channels, use_act=False, kernel_size=KERNEL_SIZE_G, padding=PADDING_RES)
            
        )

    def forward(self, input_tensor):
        return input_tensor + self.block(input_tensor)   

# < < < < GENERIC CLASSES FOR NEURAL NETWORK DEFINITION  < < < <  

# > > > > DISCRIMINATOR CLASS DEFINITION   > > > > 
class Discriminator(nn.Module): # Discriminator class inherits from the Module Class in Torch
    def __init__(self, in_channels=3, out_channels=OUT_CHANNELS):
        super().__init__()
        
        # Create discriminator layers
        disc_layers = []
        in_channels = 64  # Initial block changes it to 64 channels

        # > > >  Initial Conv2D layer followed by a LeakyReLU activation
        self.initial = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=KERNEL_SIZE_D, stride=2, padding=PADDING, padding_mode=PADDING_MODE),nn.LeakyReLU(0.2))
              
        
        #Create the discirminator layers iterativly for each layer
        for out_ch in out_channels[1:]:
            # Add Block instances to the layers list
            if out_ch == out_channels[-1]:
                disc_layers.append(Block(in_channels, out_ch, stride=1)) # Use a stride of 1 for the last layer
            else:
                disc_layers.append(Block(in_channels, out_ch, stride=2))
            
            in_channels = out_ch    #update input to match previous out # channels
        
        # > > > Final Conv2D layer to the layers list
        disc_layers.append(nn.Conv2d(in_channels, 1, kernel_size=KERNEL_SIZE_D, stride=1, padding=PADDING, padding_mode=PADDING_MODE))
        
        # Create the model by combining all layers in a sequential manner
        self.disc_model = nn.Sequential(*disc_layers)

    # Forward pass for the Discriminator class
    def forward(self, input_tensor):
        #pass incomign tensor into initial layer of discriminator
        input_tensor = self.initial(input_tensor)
        # Apply the model layers and use a sigmoid activation function for the final output
        return torch.sigmoid(self.disc_model(input_tensor))
# < < < < DISCRIMINATOR CLASS DEFINITION  < < < < 

# > > > > GENERATOR CLASS DEFINITION   > > > > 

class Generator(nn.Module):

    def __init__(self,img_channels,num_features = NUM_FEATURES, num_residuals = NUM_RESIDUALS):
        super().__init__()
        
        #The initial attribute is the conv block without the instanceNorm
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=PADDING_GEN, padding_mode=PADDING_MODE),
            nn.ReLU(inplace = True),
        )
        #The down blocks attribute is 2 conv blocks with stride of 2 to DOWNSAMPLE
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(num_features,num_features*2, kernel_size = KERNEL_SIZE_G, stride = 2, padding = 1),
                ConvBlock(num_features*2,num_features*4, kernel_size = KERNEL_SIZE_G, stride = 2, padding = 1),
            ]           
        )
        
        #Residual blocks that don't change input or number of channels
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features*4) for _ in range(num_residuals)]            
        )

        self.up_blocks = nn.ModuleList(
            [
            ConvBlock(num_features*4, num_features*2, down=False, kernel_size = KERNEL_SIZE_G, stride = 2, padding = 1, output_padding = 1 ),
            ConvBlock(num_features*2, num_features*1, down=False, kernel_size = KERNEL_SIZE_G, stride = 2, padding = 1, output_padding = 1 ),
            ]
        )

        self.last = nn.Conv2d(num_features*1, img_channels, kernel_size=7, stride=1, padding=3, padding_mode=PADDING_MODE)


    def forward(self,x):
        x = self.initial(x)

        for layer in self.down_blocks:
            x = layer(x)
        x = self.residual_blocks(x)

        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.last(x))     #last block to convert it to RGB and tanh to convert to +/- 1 range
    
# < < < < GENERATOR CLASS DEFINITION  < < < < 