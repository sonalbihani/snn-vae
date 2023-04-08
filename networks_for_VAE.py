from layers import *
from spikingjelly.clock_driven import layer
import numpy as np

init_constrain = 0.2
from prior import *
from posterior import *

class VanillaVAE_GLIF(nn.Module):
    def __init__(self, lif_param:dict, tunable_lif=False, in_channels=3, hidden_dims=[32, 64, 128, 256, 512], 
                latent_dim=128,
                beta: int = 4,
                gamma:float = 10.,
                max_capacity: int = 25,
                Capacity_max_iter: int = 1e5,
                loss_type = "beta",
                kld_weight_corrector = 1.0):
        super(VanillaVAE_GLIF, self).__init__()
        
        self.choice_param_name = ['alpha', 'beta', 'gamma']
        self.lifcal_param_name = ['tau', 'Vth', 'leak', 'conduct', 'reVth']
        self.T = lif_param['t']
        self.lif_param = lif_param
        self.tunable_lif = tunable_lif
        self.gamma = gamma
        self.beta = beta
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter
        self.loss_type = loss_type
        self.kld_weight = kld_weight_corrector
        self.latent_dim = latent_dim

        image_channels = in_channels
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    layer.SeqToANNContainer(nn.Conv2d(in_channels, out_channels=h_dim, kernel_size= 3, stride= 2, padding  = 1), nn.BatchNorm2d(h_dim)),
                    LIFSpike_CW(h_dim, **self.lif_param)
                    # nn.LeakyReLU())
            ))
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = layer.SeqToANNContainer(nn.Linear(hidden_dims[-1]*4, latent_dim))
        self.fc_var = layer.SeqToANNContainer(nn.Linear(hidden_dims[-1]*4, latent_dim))

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1]*4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    layer.SeqToANNContainer(nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1])),
                    LIFSpike_CW(hidden_dims[i + 1], **self.lif_param)
            ))
            
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
                            layer.SeqToANNContainer(nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1])),
                            LIFSpike_CW(hidden_dims[-1], **self.lif_param),
                            layer.SeqToANNContainer(nn.Conv2d(hidden_dims[-1], out_channels=image_channels,
                                      kernel_size= 3, padding= 1)),
                            nn.Tanh())

        self._initialize_weights()
        print('steps:{}'.format(self.T),
              'init-tau:{}'.format(tau),
              'aa:{}'.format(aa),
              'Vth:{}'.format(Vth)
              )

    def encode(self, input):
        input = input.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=2)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        means = self.fc_mu(result)
        log_variances = self.fc_var(result)
        return [means, log_variances]

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(self.T, -1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, input):
        means, log_variances = self.encode(input)
        z = self.reparameterize(means, log_variances)
        return  [self.decode(z).mean(0), input, means.mean(0), log_variances.mean(0)]
    
    def reparameterize(self, means, log_variances):
        std = torch.exp(0.5 * log_variances)
        eps = torch.randn_like(std)
        return eps * std + means


    
    def loss_function(self, means, log_variances, output, target):
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_variances - means ** 2 - log_variances.exp(), dim = 1), dim = 0)
        reconstruction_loss = F.mse_loss(output, target)
        if self.loss_type == "beta":
          loss = reconstruction_loss + self.beta * self.kld_weight * kld_loss
        else:
          self.C_max = self.C_max.to(device)
          C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
          loss = reconstruction_loss + self.gamma * self.kld_weight* (kld_loss - C).abs()
        return {"loss":loss, "Reconstruction_loss": reconstruction_loss, "KLD_loss": kld_loss}

    def generate(self, x,):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]

    def sample(self, num_samples):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(device)

        samples = self.decode(z)
        return samples
    
    def randomize_gate(self):
        for name, m in self.named_modules():
            if all([hasattr(m, i) for i in self.choice_param_name]):
                for i in range(len(self.choice_param_name)):
                    setattr(m, self.choice_param_name[i],
                            nn.Parameter(
                                torch.tensor(init_constrain * (np.random.rand(m.plane) - 0.5)
                                             , dtype=torch.float)
                                        )
                            )
                    

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, tdBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class DIPVAE_GLIF(nn.Module):
    def __init__(self, lif_param:dict, tunable_lif=False, in_channels=3, hidden_dims=[32, 64, 128, 256, 512], 
                latent_dim=128,
                lambda_diag = 10.,
                lambda_offdiag= 5.,
                kld_weight_corrector = 1.0):
        super(DIPVAE_GLIF, self).__init__()
        
        self.choice_param_name = ['alpha', 'beta', 'gamma']
        self.lifcal_param_name = ['tau', 'Vth', 'leak', 'conduct', 'reVth']
        self.T = lif_param['t']
        self.lif_param = lif_param
        self.tunable_lif = tunable_lif

        self.lambda_diag = lambda_diag
        self.lambda_offdiag = lambda_offdiag
        self.latent_dim = latent_dim
        self.kld_weight = kld_weight_corrector

        image_channels = in_channels
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    layer.SeqToANNContainer(nn.Conv2d(in_channels, out_channels=h_dim, kernel_size= 3, stride= 2, padding  = 1), nn.BatchNorm2d(h_dim)),
                    LIFSpike_CW(h_dim, **self.lif_param)
                    # nn.LeakyReLU())
            ))
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = layer.SeqToANNContainer(nn.Linear(hidden_dims[-1]*4, latent_dim))
        self.fc_var = layer.SeqToANNContainer(nn.Linear(hidden_dims[-1]*4, latent_dim))

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1]*4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    layer.SeqToANNContainer(nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1])),
                    LIFSpike_CW(hidden_dims[i + 1], **self.lif_param)
            ))
            
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
                            layer.SeqToANNContainer(nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1])),
                            LIFSpike_CW(hidden_dims[-1], **self.lif_param),
                            layer.SeqToANNContainer(nn.Conv2d(hidden_dims[-1], out_channels=image_channels,
                                      kernel_size= 3, padding= 1)),
                            nn.Tanh())

        self._initialize_weights()
        print('steps:{}'.format(self.T),
              'init-tau:{}'.format(tau),
              'aa:{}'.format(aa),
              'Vth:{}'.format(Vth)
              )

    def encode(self, input):
        input = input.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=2)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        means = self.fc_mu(result)
        log_variances = self.fc_var(result)
        return [means, log_variances]

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(self.T, -1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, input):
        means, log_variances = self.encode(input)
        z = self.reparameterize(means, log_variances)
        return  [self.decode(z).mean(0), input, means.mean(0), log_variances.mean(0)]
    
    def reparameterize(self, means, log_variances):
        std = torch.exp(0.5 * log_variances)
        eps = torch.randn_like(std)
        return eps * std + means


    
    def loss_function(self, means, log_variances, output, target):
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_variances - means ** 2 - log_variances.exp(), dim = 1), dim = 0)
        reconstruction_loss = F.mse_loss(output, target, reduction = 'sum')

        #DIP Loss
        centered_means = means - means.mean(dim=1, keepdim = True) # [B x D]
        cov_means = centered_means.t().matmul(centered_means).squeeze() # [D X D]
        cov_z = cov_means + torch.mean(torch.diagonal((2. * log_variances).exp(), dim1 = 0), dim = 0) # [D x D]

        cov_diag = torch.diag(cov_z) # [D]
        cov_offdiag = cov_z - torch.diag(cov_diag) # [D x D]
        dip_loss = self.lambda_offdiag * torch.sum(cov_offdiag ** 2) + self.lambda_diag * torch.sum((cov_diag - 1) ** 2)


        loss = reconstruction_loss + self.kld_weight * kld_loss + dip_loss
        return {"loss":loss, "Reconstruction_loss": reconstruction_loss, "KLD_loss": kld_loss, 'DIP_Loss':dip_loss}

    def generate(self, x,):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]

    def sample(self, num_samples):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(device)

        samples = self.decode(z)
        return samples
    
    def randomize_gate(self):
        for name, m in self.named_modules():
            if all([hasattr(m, i) for i in self.choice_param_name]):
                for i in range(len(self.choice_param_name)):
                    setattr(m, self.choice_param_name[i],
                            nn.Parameter(
                                torch.tensor(init_constrain * (np.random.rand(m.plane) - 0.5)
                                             , dtype=torch.float)
                                        )
                            )
                    

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, tdBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    

class DIPVAE_GLIF_Bernoulli(nn.Module):
    def __init__(self, lif_param:dict, tunable_lif=False, in_channels=3, hidden_dims=[32, 64, 128, 256, 512], 
                latent_dim=128,
                k = 20,
                kld_weight_corrector = 6e-3):
        super(DIPVAE_GLIF_Bernoulli, self).__init__()
        
        self.choice_param_name = ['alpha', 'beta', 'gamma']
        self.lifcal_param_name = ['tau', 'Vth', 'leak', 'conduct', 'reVth']
        self.T = lif_param['t']
        self.lif_param = lif_param
        self.tunable_lif = tunable_lif
        self.k = k
        self.loss = []

        self.latent_dim = latent_dim
        self.kld_weight = kld_weight_corrector

        image_channels = in_channels
        modules = []
        is_first_conv = True
        for h_dim in hidden_dims:
            modules.append(
                tdConv(in_channels, out_channels=h_dim, kernel_size= 3, stride= 2, padding  = 1, 
                       bias=True,
                       bn=tdBatchNorm(h_dim),
                       spike=LIFSpike_CW_Mod(h_dim, **self.lif_param),
                       is_first_conv=is_first_conv)
            )
            in_channels = h_dim
            is_first_conv = False

        self.encoder = nn.Sequential(*modules)
        self.before_latent_layer = tdLinear(hidden_dims[-1]*4,
                                            latent_dim,
                                            bias=True,
                                            bn=tdBatchNorm(latent_dim),
                                            spike=LIFSpike_CW_Mod(latent_dim, **self.lif_param))
        
        self.prior = PriorBernoulliSTBP(lif_param = self.lif_param, k = self.k, T = self.T)
        
        self.posterior = PosteriorBernoulliSTBP(lif_param = self.lif_param, k = self.k, T = self.T)

        # Build Decoder
        modules = []

        self.decoder_input = tdLinear(latent_dim, 
                                        hidden_dims[-1] * 4, 
                                        bias=True,
                                        bn=tdBatchNorm(hidden_dims[-1] * 4),
                                        spike=LIFSpike_CW_Mod(hidden_dims[-1] * 4, **self.lif_param))

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                tdConvTranspose(hidden_dims[i],
                                    hidden_dims[i + 1],
                                    kernel_size=3,
                                    stride = 2,
                                    padding=1,
                                    output_padding=1,
                                    bias=True,
                                    bn=tdBatchNorm(hidden_dims[i+1]),
                                    spike=LIFSpike_CW_Mod(hidden_dims[i + 1], **self.lif_param))
                )

            
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(tdConvTranspose(hidden_dims[-1],
                                            hidden_dims[-1],
                                            kernel_size=3,
                                            stride=2,
                                            padding=1,
                                            output_padding=1,
                                            bias=True,
                                            bn=tdBatchNorm(hidden_dims[-1]),
                                            spike=LIFSpike_CW_Mod(hidden_dims[-1], **self.lif_param)),
                            tdConvTranspose(hidden_dims[-1], 
                                            out_channels=image_channels,
                                            kernel_size=3, 
                                            padding=1,
                                            bias=True,
                                            bn=None,
                                            spike=None),
                            nn.Tanh())

        self._initialize_weights()
        self.p = 0
        self.membrane_output_layer = MembraneOutputLayer(T=self.T)
        self.psp = PSP()


        print('steps:{}'.format(self.T),
              'init-tau:{}'.format(tau),
              'aa:{}'.format(aa),
              'Vth:{}'.format(Vth)
              )

    def encode(self, input, scheduled = False):
        input = input.unsqueeze(-1).repeat(1, 1, 1, 1,self.T)
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1, end_dim =3)
        latent_x = self.before_latent_layer(result)
        sampled_z, q_z = self.posterior(latent_x)
        p_z = self.prior(sampled_z, scheduled, self.p)
        return sampled_z, q_z, p_z

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2, self.T)
        result = self.decoder(result)
        result = self.final_layer(result)
        result = self.membrane_output_layer(result) 
        return result

    def forward(self, input):
        sampled_z, q_z, p_z = self.encode(input)
        x_recon = self.decode(sampled_z)
        return x_recon, q_z, p_z, sampled_z
    
    def reparameterize(self, means, log_variances):
        std = torch.exp(0.5 * log_variances)
        eps = torch.randn_like(std)
        return eps * std + means


    def loss_function_mmd(self, input, output, q_z, p_z):
        """
        q_z is q(z|x): (N,latent_dim,k,T)
        p_z is p(z): (N,latent_dim,k,T)
        """
        reconstruction_loss = F.mse_loss(output, input)
        q_z_ber = torch.mean(q_z, dim=2) # (N, latent_dim, T)
        p_z_ber = torch.mean(p_z, dim=2) # (N, latent_dim, T)

        #kld_loss = torch.mean((q_z_ber - p_z_ber)**2)
        mmd_loss = torch.mean((self.psp(q_z_ber)-self.psp(p_z_ber))**2)
        loss = reconstruction_loss + mmd_loss
        return {'loss': loss, 'Reconstruction_Loss':reconstruction_loss, 'Distance_Loss': mmd_loss}

    def loss_function(self, target, output, q_z, p_z):
        """
        q_z is q(z|x): (N,latent_dim,k,T)
        p_z is p(z): (N,latent_dim,k,T)
        """
        reconstruction_loss = F.mse_loss(output, target)
        prob_q = torch.mean(q_z, dim=2) # (N, latent_dim, T)
        prob_p = torch.mean(p_z, dim=2) # (N, latent_dim, T)
        kld_loss = prob_q * torch.log((prob_q+1e-2)/(prob_p+1e-2)) + (1-prob_q)*torch.log((1-prob_q+1e-2)/(1-prob_p+1e-2))
        kld_loss = torch.mean(torch.sum(kld_loss, dim=(1,2)))
        
        loss = reconstruction_loss + self.kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':reconstruction_loss, 'Distance_Loss': kld_loss}

    def generate(self, x,):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]

    def sample(self, num_samples):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        sampled_z = self.prior.sample(num_samples)
        sampled_z = sampled_z.to(device)
        sampled_x = self.decode(sampled_z)
        return sampled_x, sampled_z

    def update_p(self, epoch, max_epoch):
        init_p = 0.1
        last_p = 0.3
        self.p = (last_p-init_p) * epoch / max_epoch + init_p
    
    def randomize_gate(self):
        for name, m in self.named_modules():
            if all([hasattr(m, i) for i in self.choice_param_name]):
                for i in range(len(self.choice_param_name)):
                    setattr(m, self.choice_param_name[i],
                            nn.Parameter(
                                torch.tensor(init_constrain * (np.random.rand(m.plane) - 0.5)
                                             , dtype=torch.float)
                                        )
                            )
                    

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, tdBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                  



class DIPVAE_GLIF_Static_Bernoulli(nn.Module):
    def __init__(self, lif_param, tunable_lif=False, in_channels=3, hidden_dims=[32, 64, 128, 256, 512], 
                latent_dim=128,
                k = 20,
                kld_weight_corrector = 1e-4):
        super(DIPVAE_GLIF_Static_Bernoulli, self).__init__()
        
        self.choice_param_name = ['alpha', 'beta', 'gamma']
        self.lifcal_param_name = ['tau', 'Vth', 'leak', 'conduct', 'reVth']
        self.T = lif_param['t']
        self.lif_param = lif_param
        self.tunable_lif = tunable_lif
        self.k = k
        self.loss = []
        self.lif_model = LIFSpike_CW_softsimple_mod
        

        self.latent_dim = latent_dim
        self.kld_weight = kld_weight_corrector

        image_channels = in_channels
        modules = []
        is_first_conv = True
        for h_dim in hidden_dims:
            modules.append(
                tdConv(in_channels, out_channels=h_dim, kernel_size= 3, stride= 2, padding  = 1, 
                       bias=True,
                       bn=tdBatchNorm(h_dim),
                       spike=self.lif_model(h_dim, **self.lif_param),
                       is_first_conv=is_first_conv)
            )
            in_channels = h_dim
            is_first_conv = False

        self.encoder = nn.Sequential(*modules)
        self.before_latent_layer = tdLinear(hidden_dims[-1]*4,
                                            latent_dim,
                                            bias=True,
                                            bn=tdBatchNorm(latent_dim),
                                            spike=self.lif_model(latent_dim, **self.lif_param))
        
        self.prior = PriorBernoulliSTBP(lif_param = self.lif_param, k = self.k, T = self.T, lif_model = self.lif_model)
        
        self.posterior = PosteriorBernoulliSTBP(lif_param = self.lif_param, k = self.k, T = self.T, lif_model = self.lif_model)

        # Build Decoder
        modules = []

        self.decoder_input = tdLinear(latent_dim, 
                                        hidden_dims[-1] * 4, 
                                        bias=True,
                                        bn=tdBatchNorm(hidden_dims[-1] * 4),
                                        spike=self.lif_model(hidden_dims[-1] * 4, **self.lif_param))

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                tdConvTranspose(hidden_dims[i],
                                    hidden_dims[i + 1],
                                    kernel_size=3,
                                    stride = 2,
                                    padding=1,
                                    output_padding=1,
                                    bias=True,
                                    bn=tdBatchNorm(hidden_dims[i+1]),
                                    spike=self.lif_model(hidden_dims[i + 1], **self.lif_param))
                )

            
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(tdConvTranspose(hidden_dims[-1],
                                            hidden_dims[-1],
                                            kernel_size=3,
                                            stride=2,
                                            padding=1,
                                            output_padding=1,
                                            bias=True,
                                            bn=tdBatchNorm(hidden_dims[-1]),
                                            spike=self.lif_model(hidden_dims[-1], **self.lif_param)),
                            tdConvTranspose(hidden_dims[-1], 
                                            out_channels=image_channels,
                                            kernel_size=3, 
                                            padding=1,
                                            bias=True,
                                            bn=None,
                                            spike=None),
                            nn.Tanh())

        self._initialize_weights()
        self.p = 0
        self.membrane_output_layer = MembraneOutputLayer(T=self.T)
        self.psp = PSP()


        print('steps:{}'.format(self.T),
              'init-tau:{}'.format(tau),
              'aa:{}'.format(aa),
              'Vth:{}'.format(Vth)
              )

    def encode(self, input, scheduled = False):
        input = input.unsqueeze(-1).repeat(1, 1, 1, 1,self.T)
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1, end_dim =3)
        latent_x = self.before_latent_layer(result)
        sampled_z, q_z = self.posterior(latent_x)
        p_z = self.prior(sampled_z, scheduled, self.p)
        return sampled_z, q_z, p_z

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2, self.T)
        result = self.decoder(result)
        result = self.final_layer(result)
        result = self.membrane_output_layer(result) 
        return result

    def forward(self, input):
        sampled_z, q_z, p_z = self.encode(input)
        x_recon = self.decode(sampled_z)
        return x_recon, q_z, p_z, sampled_z
    
    def reparameterize(self, means, log_variances):
        std = torch.exp(0.5 * log_variances)
        eps = torch.randn_like(std)
        return eps * std + means


    def loss_function(self, target, output, q_z, p_z):
        """
        q_z is q(z|x): (N,latent_dim,k,T)
        p_z is p(z): (N,latent_dim,k,T)
        """
        reconstruction_loss = F.mse_loss(output, target)
        q_z_ber = torch.mean(q_z, dim=2) # (N, latent_dim, T)
        p_z_ber = torch.mean(p_z, dim=2) # (N, latent_dim, T)

        #kld_loss = torch.mean((q_z_ber - p_z_ber)**2)
        mmd_loss = torch.mean((self.psp(q_z_ber)-self.psp(p_z_ber))**2)
        loss = reconstruction_loss + mmd_loss
        return {'loss': loss, 'Reconstruction_Loss':reconstruction_loss, 'Distance_Loss': mmd_loss}

    def loss_function_kld(self, target, output, q_z, p_z):
        """
        q_z is q(z|x): (N,latent_dim,k,T)
        p_z is p(z): (N,latent_dim,k,T)
        """
        reconstruction_loss = F.mse_loss(output, target)
        prob_q = torch.mean(q_z, dim=2) # (N, latent_dim, T)
        prob_p = torch.mean(p_z, dim=2) # (N, latent_dim, T)
        kld_loss = prob_q * torch.log((prob_q+1e-2)/(prob_p+1e-2)) + (1-prob_q)*torch.log((1-prob_q+1e-2)/(1-prob_p+1e-2))
        kld_loss = torch.mean(torch.sum(kld_loss, dim=(1,2)))
        
        loss = reconstruction_loss + self.kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':reconstruction_loss, 'Distance_Loss': kld_loss}

    def generate(self, x,):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]

    def sample(self, num_samples):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        sampled_z = self.prior.sample(num_samples)
        sampled_z = sampled_z.to(device)
        sampled_x = self.decode(sampled_z)
        return sampled_x, sampled_z

    def update_p(self, epoch, max_epoch):
        init_p = 0.1
        last_p = 0.3
        self.p = (last_p-init_p) * epoch / max_epoch + init_p
    
    def randomize_gate(self):
        for name, m in self.named_modules():
            if all([hasattr(m, i) for i in self.choice_param_name]):
                for i in range(len(self.choice_param_name)):
                    setattr(m, self.choice_param_name[i],
                            nn.Parameter(
                                torch.tensor(init_constrain * (np.random.rand(m.plane) - 0.5)
                                             , dtype=torch.float)
                                        )
                            )
                    

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, tdBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                  

