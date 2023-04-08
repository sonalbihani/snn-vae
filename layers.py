import torch
import torch.nn as nn
import math
import torch.nn.functional as F

torch.pi = torch.acos(torch.zeros(1)).item() * 2
steps = 4
a = 0.25
Vth = 0.5  #  V_threshold
aa = Vth
tau = 0.25  # exponential decay coefficient
conduct = 0.5 # time-dependent synaptic weight
linear_decay = Vth/(steps * 2)  #linear decay coefficient

gamma_SG = 1.
class SpikeAct_extended(torch.autograd.Function):
    '''
    solving the non-differentiable term of the Heavisde function
    '''
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # if input = u > Vth then output = 1
        output = torch.gt(input, 0.)
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output.clone()

        # hu is an approximate func of df/du in linear formulation
        hu = torch.abs(input[0]) < 0.5
        hu = hu.float()

        # arctan surrogate function
        # hu =  1 / ((input * torch.pi) ** 2 + 1)

        # triangles
        # hu = (1 / gamma_SG) * (1 / gamma_SG) * ((gamma_SG - input.abs()).clamp(min=0))

        return grad_input * hu

class ArchAct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.gt(input, 0.5)
        return output.float()
    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input

class LIFSpike_CW(nn.Module):
    '''
    gated spiking neuron
    '''
    def __init__(self, inplace, **kwargs):
        super(LIFSpike_CW, self).__init__()
        self.T = kwargs['t']
        self.soft_mode = kwargs['soft_mode']
        self.static_gate = kwargs['static_gate']
        self.static_param = kwargs['static_param']
        self.time_wise = kwargs['time_wise']
        self.plane = inplace
        #c
        self.alpha, self.beta, self.gamma = [nn.Parameter(- math.log(1 / ((i - 0.5)*0.5+0.5) - 1) * torch.ones(self.plane, dtype=torch.float))
                                                 for i in kwargs['gate']]

        self.tau, self.Vth, self.leak = [nn.Parameter(- math.log(1 / i - 1) * torch.ones(self.plane, dtype=torch.float))
                              for i in kwargs['param'][:-1]]
        self.reVth = nn.Parameter(- math.log(1 / kwargs['param'][1] - 1) * torch.ones(self.plane, dtype=torch.float))
        #t, c
        self.conduct = [nn.Parameter(- math.log(1 / i - 1) * torch.ones((self.T, self.plane), dtype=torch.float))
                                   for i in kwargs['param'][3:]][0]

    def forward(self, x): #t, b, c, h, w
        u = torch.zeros(x.shape[1:], device=x.device)
        out = torch.zeros(x.shape, device=x.device)
        self.T = x.shape[-1]
        for step in range(self.T):
            u, out[step] = self.extended_state_update(u, out[max(step - 1, 0)], x[step],
                                                      tau=self.tau.sigmoid(),
                                                      Vth=self.Vth.sigmoid(),
                                                      leak=self.leak.sigmoid(),
                                                      conduct=self.conduct[step].sigmoid(),
                                                      reVth=self.reVth.sigmoid())
        return out

    #[b, c, h, w]  * [c]
    def extended_state_update(self, u_t_n1, o_t_n1, W_mul_o_t_n1, tau, Vth, leak, conduct, reVth):
        if(u_t_n1.ndim > 2):
            if self.static_gate:
                al, be, ga = self.alpha.view(1, -1, 1, 1).clone().detach().gt(0.).float(), self.beta.view(1, -1, 1, 1).clone().detach().gt(0.).float(), self.gamma.view(1, -1, 1, 1).clone().detach().gt(0.).float()
            else:
                al, be, ga = ArchAct.apply(self.alpha.view(1, -1, 1, 1).sigmoid()), ArchAct.apply(self.beta.view(1, -1, 1, 1).sigmoid()), ArchAct.apply(self.gamma.view(1, -1, 1, 1).sigmoid())
            I_t1 = W_mul_o_t_n1 * (1 - be * (1 - conduct[None, :, None, None]))
            u_t1_n1 = ((1 - al * (1 - tau[None, :, None, None])) * u_t_n1 * (1 - ga * o_t_n1.clone()) - (1 - al) * leak[None, :, None, None]) + \
            I_t1 - (1 - ga) * reVth[None, :, None, None] * o_t_n1.clone()
            o_t1_n1 = SpikeAct_extended.apply(u_t1_n1 - Vth[None, :, None, None])
        else:
            if self.static_gate:
                al, be, ga = self.alpha.view(1, -1).clone().detach().gt(0.).float(), self.beta.view(1, -1).clone().detach().gt(0.).float(), self.gamma.view(1, -1).clone().detach().gt(0.).float()
            else:
                al, be, ga = ArchAct.apply(self.alpha.view(1, -1).sigmoid()), ArchAct.apply(self.beta.view(1, -1).sigmoid()), ArchAct.apply(self.gamma.view(1, -1).sigmoid())
            I_t1 = W_mul_o_t_n1 * (1 - be * (1 - conduct[None, :]))
            u_t1_n1 = ((1 - al * (1 - tau[None, :])) * u_t_n1 * (1 - ga * o_t_n1.clone()) - (1 - al) * leak[None, :]) + \
            I_t1 - (1 - ga) * reVth[None, :] * o_t_n1.clone()
            o_t1_n1 = SpikeAct_extended.apply(u_t1_n1 - Vth[None, :])

        return u_t1_n1, o_t1_n1

    def _initialize_params(self, **kwargs):
        self.mid_gate_mode = True
        self.tau.copy_(torch.tensor(- math.log(1 / kwargs['param'][0] - 1), dtype=torch.float, device=self.tau.device))
        self.Vth.copy_(torch.tensor(- math.log(1 / kwargs['param'][1] - 1), dtype=torch.float, device=self.Vth.device))
        self.reVth.copy_(torch.tensor(- math.log(1 / kwargs['param'][1] - 1), dtype=torch.float, device=self.reVth.device))

        self.leak.copy_(- math.log(1 / kwargs['param'][2] - 1) * torch.ones(self.T, dtype=torch.float, device=self.leak.device))
        self.conduct.copy_(- math.log(1 / kwargs['param'][3] - 1) * torch.ones(self.T, dtype=torch.float, device=self.conduct.device))

class LIFSpike_CW_Mod(nn.Module):
    '''
    gated spiking neuron, with different output shape
    '''
    def __init__(self, inplace, **kwargs):
        super(LIFSpike_CW_Mod, self).__init__()
        self.T = kwargs['t']
        self.soft_mode = kwargs['soft_mode']
        self.static_gate = kwargs['static_gate']
        self.static_param = kwargs['static_param']
        self.time_wise = kwargs['time_wise']
        self.plane = inplace
        #c
        self.alpha, self.beta, self.gamma = [nn.Parameter(- math.log(1 / ((i - 0.5)*0.5+0.5) - 1) * torch.ones(self.plane, dtype=torch.float))
                                                 for i in kwargs['gate']]

        self.tau, self.Vth, self.leak = [nn.Parameter(- math.log(1 / i - 1) * torch.ones(self.plane, dtype=torch.float))
                              for i in kwargs['param'][:-1]]
        self.reVth = nn.Parameter(- math.log(1 / kwargs['param'][1] - 1) * torch.ones(self.plane, dtype=torch.float))
        #t, c
        self.conduct = [nn.Parameter(- math.log(1 / i - 1) * torch.ones((self.plane, self.T), dtype=torch.float))
                                   for i in kwargs['param'][3:]][0]

    def forward(self, x): #t, b, c, h, w
        u = torch.zeros(x.shape[:-1], device=x.device)
        out = torch.zeros(x.shape, device=x.device)
        self.T = x.shape[-1]
        for step in range(self.T):
            #  u, out[..., step] = self.state_update(u, out[..., max(step-1, 0)], x[..., step])
            u, out[..., step] = self.extended_state_update(u, out[..., max(step - 1, 0)], x[...,step],
                                                      tau=self.tau.sigmoid(),
                                                      Vth=self.Vth.sigmoid(),
                                                      leak=self.leak.sigmoid(),
                                                      conduct=self.conduct[...,step].sigmoid(),
                                                      reVth=self.reVth.sigmoid())
        return out

    #[b, c, h, w]  * [c]
    def extended_state_update(self, u_t_n1, o_t_n1, W_mul_o_t_n1, tau, Vth, leak, conduct, reVth):
        if(u_t_n1.ndim > 2):
            if self.static_gate:
                al, be, ga = self.alpha.view(1, -1, 1, 1).clone().detach().gt(0.).float(), self.beta.view(1, -1, 1, 1).clone().detach().gt(0.).float(), self.gamma.view(1, -1, 1, 1).clone().detach().gt(0.).float()
            else:
                al, be, ga = ArchAct.apply(self.alpha.view(1, -1, 1, 1).sigmoid()), ArchAct.apply(self.beta.view(1, -1, 1, 1).sigmoid()), ArchAct.apply(self.gamma.view(1, -1, 1, 1).sigmoid())
            I_t1 = W_mul_o_t_n1 * (1 - be * (1 - conduct[None, :, None, None]))
            u_t1_n1 = ((1 - al * (1 - tau[None, :, None, None])) * u_t_n1 * (1 - ga * o_t_n1.clone()) - (1 - al) * leak[None, :, None, None]) + \
            I_t1 - (1 - ga) * reVth[None, :, None, None] * o_t_n1.clone()
            o_t1_n1 = SpikeAct_extended.apply(u_t1_n1 - Vth[None, :, None, None])
        else:
            if self.static_gate:
                al, be, ga = self.alpha.view(1, -1).clone().detach().gt(0.).float(), self.beta.view(1, -1).clone().detach().gt(0.).float(), self.gamma.view(1, -1).clone().detach().gt(0.).float()
            else:
                al, be, ga = ArchAct.apply(self.alpha.view(1, -1).sigmoid()), ArchAct.apply(self.beta.view(1, -1).sigmoid()), ArchAct.apply(self.gamma.view(1, -1).sigmoid())
            I_t1 = W_mul_o_t_n1 * (1 - be * (1 - conduct[None, :]))
            u_t1_n1 = ((1 - al * (1 - tau[None, :])) * u_t_n1 * (1 - ga * o_t_n1.clone()) - (1 - al) * leak[None, :]) + \
            I_t1 - (1 - ga) * reVth[None, :] * o_t_n1.clone()
            o_t1_n1 = SpikeAct_extended.apply(u_t1_n1 - Vth[None, :])
        return u_t1_n1, o_t1_n1

    def _initialize_params(self, **kwargs):
        self.mid_gate_mode = True
        self.tau.copy_(torch.tensor(- math.log(1 / kwargs['param'][0] - 1), dtype=torch.float, device=self.tau.device))
        self.Vth.copy_(torch.tensor(- math.log(1 / kwargs['param'][1] - 1), dtype=torch.float, device=self.Vth.device))
        self.reVth.copy_(torch.tensor(- math.log(1 / kwargs['param'][1] - 1), dtype=torch.float, device=self.reVth.device))

        self.leak.copy_(- math.log(1 / kwargs['param'][2] - 1) * torch.ones(self.T, dtype=torch.float, device=self.leak.device))
        self.conduct.copy_(- math.log(1 / kwargs['param'][3] - 1) * torch.ones(self.T, dtype=torch.float, device=self.conduct.device))

        
class LIFSpike_CW_softsimple_mod(nn.Module):
    '''
        Coarsely fused LIF, referred to as GLIF_f in 'GLIF: A Unified Gated Leaky Integrate-and-Fire Neuron for Spiking Neural Networks'
    '''
    def __init__(self, inplace, **kwargs):
        super(LIFSpike_CW_softsimple_mod, self).__init__()
        self.T = kwargs['t']
        self.soft_mode = kwargs['soft_mode']
        self.static_gate = kwargs['static_gate']
        self.static_param = kwargs['static_param']
        self.time_wise = kwargs['time_wise']
        self.plane = inplace
        #c
        self.gamma = nn.Parameter(- math.log(1 / ((kwargs['gate'][-1] - 0.5)*0.5+0.5) - 1) * torch.ones(self.plane, dtype=torch.float))

        self.tau, self.Vth, self.leak = [nn.Parameter(- math.log(1 / i - 1) * torch.ones(self.plane, dtype=torch.float))
                              for i in kwargs['param'][:-1]]
        self.reVth = nn.Parameter(- math.log(1 / kwargs['param'][1] - 1) * torch.ones(self.plane, dtype=torch.float))
        #t, c
        self.conduct = [nn.Parameter(- math.log(1 / i - 1) * torch.ones((self.T, self.plane), dtype=torch.float))
                                   for i in kwargs['param'][3:]][0]

    def forward(self, x): #t, b, c, h, w
        u = torch.zeros(x.shape[:-1], device=x.device)
        out = torch.zeros(x.shape, device=x.device)
        self.T = x.shape[-1]
        for step in range(self.T):
             u, out[..., step] = self.extended_state_update(u, out[..., max(step - 1, 0)], x[...,step],
                                                      tau=self.tau.sigmoid(),
                                                      Vth=self.Vth.sigmoid(),
                                                      leak=self.leak.sigmoid(),
                                                      conduct=self.conduct[step].sigmoid(),
                                                      reVth=self.reVth.sigmoid())
        return out

    #[b, c, h, w]  * [c]
    def extended_state_update(self, u_t_n1, o_t_n1, W_mul_o_t_n1, tau, Vth, leak, conduct, reVth):
        if(u_t_n1.ndim > 2):
            I_t1 = W_mul_o_t_n1 * conduct[None, :, None, None]
            u_t1_n1 = ((tau[None, :, None, None]) * u_t_n1 * (1 - o_t_n1.clone()) - leak[None, :, None, None]) + \
                      I_t1 - \
                      reVth[None, :, None, None] * o_t_n1.clone()
            o_t1_n1 = SpikeAct_extended.apply(u_t1_n1 - Vth[None, :, None, None])
        else:
            I_t1 = W_mul_o_t_n1 * conduct[None, :]
            u_t1_n1 = ((tau[None, :]) * u_t_n1 * (1 - o_t_n1.clone()) - leak[None, :]) + \
                      I_t1 - \
                      reVth[None, :] * o_t_n1.clone()
            o_t1_n1 = SpikeAct_extended.apply(u_t1_n1 - Vth[None, :])
        return u_t1_n1, o_t1_n1

    def _initialize_params(self, **kwargs):
        self.mid_gate_mode = True
        self.tau.copy_(torch.tensor(- math.log(1 / kwargs['param'][0] - 1), dtype=torch.float, device=self.tau.device))
        self.Vth.copy_(torch.tensor(- math.log(1 / kwargs['param'][1] - 1), dtype=torch.float, device=self.Vth.device))
        self.reVth.copy_(torch.tensor(- math.log(1 / kwargs['param'][1] - 1), dtype=torch.float, device=self.reVth.device))

        self.leak.copy_(- math.log(1 / kwargs['param'][2] - 1) * torch.ones(self.T, dtype=torch.float, device=self.leak.device))
        self.conduct.copy_(- math.log(1 / kwargs['param'][3] - 1) * torch.ones(self.T, dtype=torch.float, device=self.conduct.device))

    def gumbel_on(self):
        self.static_gate = False

    def gumbel_off(self):
        self.static_gate = True

        
class MembraneOutputLayer(nn.Module):
    """
    outputs the last time membrane potential of the LIF neuron with V_th=infty
    """
    def __init__(self, T = 3) -> None:
        super().__init__()
        n_steps = T

        arr = torch.arange(n_steps-1,-1,-1)
        self.register_buffer("coef", torch.pow(0.8, arr)[None,None,None,None,:]) # (1,1,1,1,T)

    def forward(self, x):
        """
        x : (N,C,H,W,T)
        """
        out = torch.sum(x*self.coef, dim=-1)
        return out

class tdLinear(nn.Linear):
    def __init__(self, 
                in_features,
                out_features,
                bias=True,
                bn=None,
                spike=None):
        assert type(in_features) == int, 'inFeatures should not be more than 1 dimesnion. It was: {}'.format(in_features.shape)
        assert type(out_features) == int, 'outFeatures should not be more than 1 dimesnion. It was: {}'.format(out_features.shape)

        super(tdLinear, self).__init__(in_features, out_features, bias=bias)

        self.bn = bn
        self.spike = spike
        

    def forward(self, x):
        """
        x : (N,C,T)
        """        
        x = x.transpose(1, 2) # (N, T, C)
        y = F.linear(x, self.weight, self.bias)
        y = y.transpose(1, 2)# (N, C, T)
        
        if self.bn is not None:
            y = y[:,:,None,None,:]
            y = self.bn(y)
            y = y[:,:,0,0,:]
        if self.spike is not None:
            y = self.spike(y)
        return y

class tdConv(nn.Conv3d):
    def __init__(self, 
                in_channels, 
                out_channels,  
                kernel_size,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=True,
                bn=None,
                spike=None,
                is_first_conv=False):

        # kernel
        if type(kernel_size) == int:
            kernel = (kernel_size, kernel_size, 1)
        elif len(kernel_size) == 2:
            kernel = (kernel_size[0], kernel_size[1], 1)
        else:
            raise Exception('kernelSize can only be of 1 or 2 dimension. It was: {}'.format(kernel_size.shape))

        # stride
        if type(stride) == int:
            stride = (stride, stride, 1)
        elif len(stride) == 2:
            stride = (stride[0], stride[1], 1)
        else:
            raise Exception('stride can be either int or tuple of size 2. It was: {}'.format(stride.shape))

        # padding
        if type(padding) == int:
            padding = (padding, padding, 0)
        elif len(padding) == 2:
            padding = (padding[0], padding[1], 0)
        else:
            raise Exception('padding can be either int or tuple of size 2. It was: {}'.format(padding.shape))

        # dilation
        if type(dilation) == int:
            dilation = (dilation, dilation, 1)
        elif len(dilation) == 2:
            dilation = (dilation[0], dilation[1], 1)
        else:
            raise Exception('dilation can be either int or tuple of size 2. It was: {}'.format(dilation.shape))

        super(tdConv, self).__init__(in_channels, out_channels, kernel, stride, padding, dilation, groups,
                                        bias=bias)
        self.bn = bn
        self.spike = spike
        self.is_first_conv = is_first_conv

    def forward(self, x):
        x = F.conv3d(x, self.weight, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)
        if self.bn is not None:
            x = self.bn(x)
        if self.spike is not None:
            x = self.spike(x)
        return x
        

class tdConvTranspose(nn.ConvTranspose3d):
    def __init__(self, 
                in_channels, 
                out_channels,  
                kernel_size,
                stride=1,
                padding=0,
                output_padding=0,
                dilation=1,
                groups=1,
                bias=True,
                bn=None,
                spike=None):

        # kernel
        if type(kernel_size) == int:
            kernel = (kernel_size, kernel_size, 1)
        elif len(kernel_size) == 2:
            kernel = (kernel_size[0], kernel_size[1], 1)
        else:
            raise Exception('kernelSize can only be of 1 or 2 dimension. It was: {}'.format(kernel_size.shape))

        # stride
        if type(stride) == int:
            stride = (stride, stride, 1)
        elif len(stride) == 2:
            stride = (stride[0], stride[1], 1)
        else:
            raise Exception('stride can be either int or tuple of size 2. It was: {}'.format(stride.shape))

        # padding
        if type(padding) == int:
            padding = (padding, padding, 0)
        elif len(padding) == 2:
            padding = (padding[0], padding[1], 0)
        else:
            raise Exception('padding can be either int or tuple of size 2. It was: {}'.format(padding.shape))

        # dilation
        if type(dilation) == int:
            dilation = (dilation, dilation, 1)
        elif len(dilation) == 2:
            dilation = (dilation[0], dilation[1], 1)
        else:
            raise Exception('dilation can be either int or tuple of size 2. It was: {}'.format(dilation.shape))


        # output padding
        if type(output_padding) == int:
            output_padding = (output_padding, output_padding, 0)
        elif len(output_padding) == 2:
            output_padding = (output_padding[0], output_padding[1], 0)
        else:
            raise Exception('output_padding can be either int or tuple of size 2. It was: {}'.format(padding.shape))

        super().__init__(in_channels, out_channels, kernel, stride, padding, output_padding, groups,
                                        bias=bias, dilation=dilation)

        self.bn = bn
        self.spike = spike

    def forward(self, x):
        x = F.conv_transpose3d(x, self.weight, self.bias,
                        self.stride, self.padding, 
                        self.output_padding, self.groups, self.dilation)

        if self.bn is not None:
            x = self.bn(x)
        if self.spike is not None:
            x = self.spike(x)
        return x

class tdBatchNorm(nn.BatchNorm2d):
    """
        Implementation of tdBN. Link to related paper: https://arxiv.org/pdf/2011.05280. In short it is averaged over the time domain as well when doing BN.
    Args:
        num_features (int): same with nn.BatchNorm2d
        eps (float): same with nn.BatchNorm2d
        momentum (float): same with nn.BatchNorm2d
        alpha (float): an addtional parameter which may change in resblock.
        affine (bool): same with nn.BatchNorm2d
        track_running_stats (bool): same with nn.BatchNorm2d
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.1, alpha=1, affine=True, track_running_stats=True):
        super(tdBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha

    def forward(self, input):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3, 4])
            # use biased var in train
            var = input.var([0, 2, 3, 4], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = self.alpha * Vth * (input - mean[None, :, None, None, None]) / (torch.sqrt(var[None, :, None, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None, None] + self.bias[None, :, None, None, None]
        
        return input
class PSP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tau_s = 2

    def forward(self, inputs):
        """
        inputs: (N, C, T)
        """
        syns = None
        syn = 0
        n_steps = inputs.shape[-1]
        for t in range(n_steps):
            syn = syn + (inputs[...,t] - syn) / self.tau_s
            if syns is None:
                syns = syn.unsqueeze(-1)
            else:
                syns = torch.cat([syns, syn.unsqueeze(-1)], dim=-1)

        return syns

if __name__ == "__main__":
    test_data = torch.rand(1, 1, 3, 3)
    test_data, _ = torch.broadcast_tensors(test_data, torch.zeros((2,) + test_data.shape))
    # test_data = test_data.permute(1, 2, 3, 4, 0)
    print(test_data)

