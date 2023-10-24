class Generator(nn.Module):
    def __init__(self, latent_size, n_c = 3):
        super(Generator, self).__init__()
        self.latent_size = latent_size
        self.n_c = n_c # RGB so I defaulted it to 3

        self.dense_layer = nn.Linear(latent_size, 256 * 4 * 4)
        self.leaky_relu_1 = nn.LeakyReLU(0.2)
        self.reshape = nn.Unflatten(1, (256, 4, 4))
        self.upsample_1 = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv_1 = nn.ConvTranspose2d(256, 128, kernel_size=5, padding=2)
        self.leaky_relu_2 = nn.LeakyReLU(0.2)
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv_2 = nn.ConvTranspose2d(128, 64, kernel_size=5, padding=2)
        self.leaky_relu_3 = nn.LeakyReLU(0.2)
        self.upsample_3 = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv_3 = nn.ConvTranspose2d(64, 32, kernel_size=5, padding=2)
        self.leaky_relu_4 = nn.LeakyReLU(0.2)
        self.upsample_4 = nn.Upsample(scale_factor=2, mode='nearest')

        self.output_layer = nn.ConvTranspose2d(32, n_c, kernel_size=5, padding=2)

    def forward(self, input_layer):
        h = self.dense_layer(input_layer)
        h = self.leaky_relu_1(h)
        h = self.reshape(h)
        h = self.upsample_1(h)

        h = self.conv_1(h)
        h = self.leaky_relu_2(h)
        h = self.upsample_2(h)

        h = self.conv_2(h)
        h = self.leaky_relu_3(h)
        h = self.upsample_3(h)

        h = self.conv_3(h)
        h = self.leaky_relu_4(h)
        h = self.upsample_4(h)

        x = self.output_layer(h)
        return x
    

    class Discriminator(nn.Module):
        def __init__(self, image_size):
            super(Discriminator, self).__init__()
            
            self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
            self.leaky_relu_1 = nn.LeakyReLU(0.2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
            self.leaky_relu_2 = nn.LeakyReLU(0.2)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
            self.leaky_relu_3 = nn.LeakyReLU(0.2)
            self.flatten = nn.Flatten()
            self.dense1 = nn.Linear(128 * (image_size // 8) * (image_size // 8), 1)

        def forward(self, input_layer):
            h = self.conv1(input_layer)
            h = self.leaky_relu_1(h)
            h = self.conv2(h)
            h = self.leaky_relu_2(h)
            h = self.conv3(h)
            h = self.leaky_relu_3(h)
            h = self.flatten(h)
            output_layer = self.dense1(h)
            return output_layer