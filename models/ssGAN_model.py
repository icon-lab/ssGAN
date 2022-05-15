import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class ssGANModel(BaseModel):
    def name(self):
        return 'ssGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt) 
        self.isTrain = opt.isTrain
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            
        self.input_channels = 2 # real and imag
        self.output_channels = 2 # real and imag

        # load/define networks
        self.netG = networks.define_G(self.input_channels, self.output_channels, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,opt.down_samp)
        if self.isTrain:
            self.netD = networks.define_D(self.output_channels, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)
  

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)

            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
            input_B = input_B.cuda(self.gpu_ids[0], async=True)
        self.input_A = input_A
        self.input_B = input_B
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        
        self.real_A = Variable(torch.unsqueeze(self.input_A[:,0,:,:],0))
        self.real_B = Variable(torch.unsqueeze(self.input_B[:,0,:,:],0))
      
        # masks should be iffshifted (central calibration region should stay in the edges)
        self.real_A_mask = Variable((torch.unsqueeze(Variable(torch.unsqueeze(self.input_A[:,1,:,:],0)),-1)+1)/2)
        self.real_B_mask = Variable((torch.unsqueeze(Variable(torch.unsqueeze(self.input_B[:,1,:,:],0)),-1)+1)/2)
        
        self.k_real_A = torch.fft(torch.cat((torch.unsqueeze(self.real_A,4),0*torch.unsqueeze(self.real_A,4)),4),2)
        self.k_real_B = torch.fft(torch.cat((torch.unsqueeze(self.real_B,4),0*torch.unsqueeze(self.real_B,4)),4),2)
        
        self.k_real_A_us = self.k_real_A * self.real_A_mask
        self.k_real_B_us = self.k_real_B * self.real_B_mask
        
        self.real_A_us = torch.ifft(self.k_real_A_us,2)
        self.real_B_us = torch.ifft(self.k_real_B_us,2)
        
        self.real_A_us = torch.squeeze(self.real_A_us.permute(0,4,2,3,1),4).detach()
        self.real_B_us = torch.squeeze(self.real_B_us.permute(0,4,2,3,1),4).detach()
        
        self.fake_B = self.netG(self.real_A_us)
        self.fake_B_fs = self.fake_B
        self.fake_B_fs_tmp = torch.unsqueeze(self.fake_B_fs.permute(0,2,3,1),0)

        self.k_fake_B_fs = torch.fft(self.fake_B_fs_tmp,2)
        
        self.k_fake_B_us = self.k_fake_B_fs * self.real_B_mask
        
        self.fake_B_us = torch.ifft(self.k_fake_B_us,2)
        self.fake_B_us = torch.squeeze(self.fake_B_us.permute(0,4,2,3,1),4)
        
        self.k_fake_B_us_tmp = self.k_fake_B_us.permute(0,4,2,3,1)
        self.k_fake_B_us_tanh = torch.tanh(torch.squeeze(self.k_fake_B_us_tmp,4)/5000)
        
        self.k_real_B_us_tanh = torch.tanh(torch.squeeze(self.k_real_B_us.permute(0,4,2,3,1),4)/5000)

        
    # no backprop gradients
    def test(self):
        
        self.real_A = Variable(torch.unsqueeze(self.input_A[:,0,:,:],0))
        self.real_B = Variable(torch.unsqueeze(self.input_B[:,0,:,:],0))
      
        # masks should be iffshifted (central calibration region should stay in the edges)
        self.real_A_mask = Variable((torch.unsqueeze(Variable(torch.unsqueeze(self.input_A[:,1,:,:],0)),-1)+1)/2)
        self.real_B_mask = Variable((torch.unsqueeze(Variable(torch.unsqueeze(self.input_B[:,1,:,:],0)),-1)+1)/2)
        
        self.k_real_A = torch.fft(torch.cat((torch.unsqueeze(self.real_A,4),0*torch.unsqueeze(self.real_A,4)),4),2)
        self.k_real_B = torch.fft(torch.cat((torch.unsqueeze(self.real_B,4),0*torch.unsqueeze(self.real_B,4)),4),2)
        
        self.k_real_A_us = self.k_real_A * self.real_A_mask
        self.k_real_B_us = self.k_real_B * self.real_B_mask
        
        self.real_A_us = torch.ifft(self.k_real_A_us,2)
        self.real_B_us = torch.ifft(self.k_real_B_us,2)
        
        self.real_A_us = torch.squeeze(self.real_A_us.permute(0,4,2,3,1),4).detach()
        self.real_B_us = torch.squeeze(self.real_B_us.permute(0,4,2,3,1),4).detach()
        
        self.fake_B = self.netG(self.real_A_us)
        self.fake_B_fs = self.fake_B
        self.fake_B_fs_tmp = torch.unsqueeze(self.fake_B_fs.permute(0,2,3,1),0)

        self.k_fake_B_fs = torch.fft(self.fake_B_fs_tmp,2)
        
        self.k_fake_B_us = self.k_fake_B_fs * self.real_B_mask
        
        self.fake_B_us = torch.ifft(self.k_fake_B_us,2)
        self.fake_B_us = torch.squeeze(self.fake_B_us.permute(0,4,2,3,1),4)
        
        self.k_fake_B_us_tmp = self.k_fake_B_us.permute(0,4,2,3,1)
        self.k_fake_B_us_tanh = torch.tanh(torch.squeeze(self.k_fake_B_us_tmp,4)/5000)
        
        self.k_real_B_us_tanh = torch.tanh(torch.squeeze(self.k_real_B_us.permute(0,4,2,3,1),4)/5000)
        
        
    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(self.fake_B_us.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        
        pred_real = self.netD(self.real_B_us.detach())
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 * self.opt.lambda_adv

        self.loss_D.backward()
    
    
        
    def backward_G(self):

        
        pred_fake = self.netD(self.fake_B_us)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)*self.opt.lambda_adv
        
        # Second, k_x(G(A)) = k_x(B)
        self.loss_G_k_space = 30*self.criterionL1(self.k_fake_B_us_tanh, self.k_real_B_us_tanh)
        self.loss_G_k_space.retain_grad()
        
        self.loss_G_image = self.criterionL1(self.fake_B_us, self.real_B_us)
        self.loss_G_image.retain_grad()
        
        self.loss_G = (self.loss_G_k_space+self.loss_G_image)* self.opt.lambda_A +self.loss_G_GAN
        self.loss_G.retain_grad()
        
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_kspace', self.loss_G_k_space.data[0]),('G_image', self.loss_G_image.data[0]),
                            ('D_real', self.loss_D_real.data[0]),
                            ('D_fake', self.loss_D_fake.data[0])])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        real_B = util.tensor2im(self.real_B.data)
        fake_B = util.tensor2im(torch.unsqueeze(self.fake_B_fs.norm(dim=1),0).data)
        mask_A = util.tensor2im(torch.unsqueeze(torch.unsqueeze(torch.squeeze(self.real_A_mask),0),0).data)
        mask_B = util.tensor2im(torch.unsqueeze(torch.unsqueeze(torch.squeeze(self.real_B_mask),0),0).data)

        return OrderedDict([('real_A', real_A),('real_B', real_B),('fake_B', fake_B),('mask_A', mask_A),('mask_B', mask_B)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)
