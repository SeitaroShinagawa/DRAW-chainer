from chainer import Chain,Variable,cuda
import chainer.functions as F
import chainer.links as L
from Filter_funcs import Filters
from util import XP
import numpy as np

class VAE_bernoulli_attention(Chain):
    def __init__(self,H_enc,z_dim,H_dec,Read_patch,Write_patch,height,width,C): #C: [color]:3 [mono]:1  need tobe set automatically
        super().__init__(
        enc_x_he = L.Linear(C*Read_patch**2,4*H_enc),
        enc_errx_he = L.Linear(C*Read_patch**2,4*H_enc),
        enc_he_he = L.Linear(H_enc,4*H_enc),
        enc_he_mu = L.Linear(H_enc,z_dim),
        enc_he_logsig2 = L.Linear(H_enc,z_dim), #In original paper: log_sig_2 = exp(mu), so commentout
        #attf_hd = L.Linear(att_enc,4*H_dec), #Attention forward hidden to decoder 
        #attb_hd = L.Linear(att_enc,4*H_dec), #Attention backward hidden to decoder
        dec_z_hd = L.Linear(z_dim,4*H_dec),
        dec_hd_hd = L.Linear(H_dec,4*H_dec),
        dec_hd_he = L.Linear(H_dec,4*H_enc),
        dec_hd_Rmeanx = L.Linear(H_dec,1),
        dec_hd_Rmeany = L.Linear(H_dec,1),
        dec_hd_Rlnvar = L.Linear(H_dec,1),
        dec_hd_Rlnstride = L.Linear(H_dec,1),
        dec_hd_Rlngamma = L.Linear(H_dec,1),
        dec_hd_Wmeanx = L.Linear(H_dec,1),
        dec_hd_Wmeany = L.Linear(H_dec,1),
        dec_hd_Wlnvar = L.Linear(H_dec,1),
        dec_hd_Wlnstride = L.Linear(H_dec,1),
        dec_hd_Wlngamma = L.Linear(H_dec,1),
        dec_hd_y = L.Linear(H_dec,C*Write_patch**2))
        self.H_enc = H_enc
        self.z_dim = z_dim
        self.H_dec = H_dec
        self.Read_patch = Read_patch
        self.Write_patch = Write_patch
        self.height = height
        self.width = width
        self.R_filter = Filters(C,self.height,self.width,self.Read_patch)
        self.W_filter = Filters(C,self.height,self.width,self.Write_patch)
        self.C = C

    def reset(self,image_batch):
        """
        initialization
        target image: x  BxHxW[mono] 3BxHxW[color] matrix (Variable)
        current canvas: canvas  BxHxW[mono] 3BxHxW[color] matrix (Variable)
        initial canvas: ini_ [normal] val=0.5 [whitened] val=0.0  BxHxW[mono] 3BxHxW[color] matrix (Variable)
        error : target - current canvas
        {encoder,decoder} {cell,hidden}: 0
        [attentional window patch]
          [position](meanx,meany): center of each image(0.5*width,0.5*height)
          [varience](ln_var):-6.9 (var=0.001)
          [stride](ln_stride):1.1 (stride=3.0)
          [gamma](ln_gamma):0.0 (gamma=1.0)
          t: the number of processed minibatch
        """
        self.zerograds()
        self.B = image_batch.shape[0]
        self.canvas = XP.fnonzeros((self.B*self.C,self.height,self.width),val=0.0)
        self.x = F.reshape(XP.farray(image_batch),(self.B*self.C,self.height,self.width)) 
        self.errx = self.x-F.sigmoid(XP.fnonzeros((self.B*self.C,self.height,self.width),val=0.0))
        self.c = XP.fzeros((self.B, self.H_enc)) #initialize encoder cell
        self.h = XP.fzeros((self.B, self.H_enc)) #initialize encoder hidden 
        self.c2 = XP.fzeros((self.B, self.H_dec)) #initialize decoder cell (decoder hidden is initialized in train_align_draw.py)
        self.h2 = XP.fzeros((self.B, self.H_dec))
        #Rmean_x = XP.fzeros((self.B,1))
        #Rmean_y = XP.fzeros((self.B,1))
        #Wmean_x = XP.fzeros((self.B,1))
        #Wmean_y = XP.fzeros((self.B,1))
        #ln_var = F.reshape(XP.fnonzeros(self.B,val=0.0),(self.B,1))      #initial_var:0.001 -> ln_var:-6.9
        #ln_stride = F.reshape(XP.fnonzeros(self.B,val=0.0),(self.B,1))    #initial_stride:3.0 -> ln_stride:1.1
        #ln_gamma = XP.fzeros((self.B,1))     #initial_gamma:1.0 -> ln_gamma:0.0
        h_dec = self.h2 
        Wmean_x = self.dec_hd_Wmeanx(h_dec)
        Wmean_y = self.dec_hd_Wmeany(h_dec)
        Wln_var = self.dec_hd_Wlnvar(h_dec)
        Wln_stride = self.dec_hd_Wlnstride(h_dec)
        Wln_gamma = self.dec_hd_Wlngamma(h_dec)
        Rmean_x = self.dec_hd_Rmeanx(h_dec)
        Rmean_y = self.dec_hd_Rmeany(h_dec)
        Rln_var = self.dec_hd_Rlnvar(h_dec)
        Rln_stride = self.dec_hd_Rlnstride(h_dec)
        Rln_gamma = self.dec_hd_Rlngamma(h_dec)

        self.R_filter.mkFilter(Rmean_x,Rmean_y,Rln_var,Rln_stride,Rln_gamma)
        self.W_filter.mkFilter(Wmean_x,Wmean_y,Wln_var,Wln_stride,Wln_gamma)
        self.t = 0
        #print("reset pass")
         
    def encode(self,c_enc,h_enc,x,errx,h_dec):
        h_in = self.enc_x_he(x) + self.enc_errx_he(errx) + self.enc_he_he(h_enc) + self.dec_hd_he(h_dec)
        c_enc,h_enc = F.lstm(c_enc,h_in)
        mu = self.enc_he_mu(h_enc)
        logsig2 = self.enc_he_logsig2(h_enc)*2
        return c_enc,h_enc,mu,logsig2

    def decode(self,c_dec,h_dec,z):#,aa,bb):
        """
        decode latent z --> h, enc/dec filter paremeter
        """
        h_in = self.dec_z_hd(z) + self.dec_hd_hd(h_dec) #+ self.attf_hd(aa) + self.attb_hd(bb)
        c_dec,h_dec = F.lstm(c_dec,h_in)
        
        Wmean_x = F.tanh(self.dec_hd_Wmeanx(h_dec))
        Wmean_y = F.tanh(self.dec_hd_Wmeany(h_dec))
        Wln_var = self.dec_hd_Wlnvar(h_dec)
        Wln_stride = self.dec_hd_Wlnstride(h_dec)
        Wln_gamma = self.dec_hd_Wlngamma(h_dec)
        Rmean_x = F.tanh(self.dec_hd_Rmeanx(h_dec))
        Rmean_y = F.tanh(self.dec_hd_Rmeany(h_dec))
        Rln_var = self.dec_hd_Rlnvar(h_dec)
        Rln_stride = self.dec_hd_Rlnstride(h_dec)
        Rln_gamma = self.dec_hd_Rlngamma(h_dec)
        y = self.dec_hd_y(h_dec)
        return c_dec,h_dec,y, Wmean_x,Wmean_y,Wln_var,Wln_stride,Wln_gamma,Rmean_x,Rmean_y,Rln_var,Rln_stride,Rln_gamma

    def free_energy_onestep(self):#,h2,aa,bb):
        """
        [input]
        x    :  BxHxW[mono] 3BxHxW[color] matrix (Variable)
        errx :  BxHxW[mono] 3BxHxW[color] matrix (Variable)
        
        """        
        
        B=self.B
        C=self.C
        rP=self.Read_patch
        wP=self.Write_patch
        
        x_patch = self.R_filter.Filter(self.x)
        #print("x_patch max",np.max(x_patch.data))
        errx_patch = self.R_filter.Filter(self.errx)
        #reshape 3BxHxW -> Bx3HW array
        x_patch_2D = F.reshape(x_patch,(B,C*rP**2))
        errx_patch_2D = F.reshape(errx_patch,(B,C*rP**2))
        
        self.c,self.h,enc_mu,enc_logsig2 = self.encode(self.c,self.h,x_patch_2D,errx_patch_2D,self.h2)
        kl = F.gaussian_kl_divergence(enc_mu,enc_logsig2)
        z = F.gaussian(enc_mu,enc_logsig2)
        
        self.c2,self.h2,inc_canvas,Wmean_x,Wmean_y,Wln_var,Wln_stride,Wln_gamma,Rmean_x,Rmean_y,Rln_var,Rln_stride,Rln_gamma = self.decode(self.c2,self.h2,z)#,aa,bb)
        self.W_filter.mkFilter(Wmean_x,Wmean_y,Wln_var,Wln_stride,Wln_gamma)
        self.R_filter.mkFilter(Rmean_x,Rmean_y,Rln_var,Rln_stride,Rln_gamma)
        inc_canvas = F.reshape(inc_canvas,(B*C,wP,wP))
        #print("Wfilter:",np.max(self.W_filter.Fx.data),np.min(self.W_filter.Fx.data),np.max(self.W_filter.Fy.data),np.min(self.W_filter.Fy.data))
        #print("Wmean:{} {}, Wlnvar:{}, Wln_stride:{}, Wln_gamma:{}".format(Wmean_x.data,Wmean_y.data,Wln_var.data,Wln_stride.data,Wln_gamma.data))
        inc_canvas = self.W_filter.InvFilter(inc_canvas)
        self.canvas += inc_canvas
        y = F.sigmoid(self.canvas) #F.relu(self.canvas+0.5)-F.relu(self.canvas-0.5) #[normal]:sigmoid, [whitened]:tanh
        self.errx = self.x-y
        self.t += 1
        return y,kl#,h2

    def generate_onestep(self):#,h2,aa,bb):
        """
        generate from middle layer
        #call reset() first, but no relation between img_array[input] and generated image[output] 
        [input]
        x    :  BxHxW[mono] 3BxHxW[color] matrix (Variable)
        errx :  BxHxW[mono] 3BxHxW[color] matrix (Variable)
        [output]
        y   :   BxHxW[mono] 3BxHxW[color] matrix (Variable)
                [normal]:sigmoid,relu [whitened]:tanh
        """		
        zero_mat = XP.fzeros((self.B,self.z_dim))
        z = F.gaussian(zero_mat,zero_mat) #F.gaussian(mean,ln_var)  
        self.c2,self.h2,inc_canvas,Wmean_x,Wmean_y,Wln_var,Wln_stride,Wln_gamma,Rmean_x,Rmean_y,Rln_var,Rln_stride,Rln_gamma = self.decode(self.c2,self.h2,z)#,aa,bb)
        self.W_filter.mkFilter(Wmean_x,Wmean_y,Wln_var,Wln_stride,Wln_gamma)
        inc_canvas = F.reshape(inc_canvas,(self.B*self.C,self.Write_patch,self.Write_patch))
        inc_canvas = self.W_filter.InvFilter(inc_canvas)
        self.canvas += inc_canvas
        y = F.relu(self.canvas+0.5)-F.relu(self.canvas-0.5)#F.sigmoid(self.canvas) #[normal]:sigmoid, [whitened]:tanh
        self.errx = self.x-y
        self.t += 1
        return y#,h2

    def reconstruct_onestep(self):
        """
        without randomness in middle layer
        [input]
        x    :  BxHxW[mono] Bx3HxW[color] matrix (Variable)
        errx :  BxHxW[mono] Bx3HxW[color] matrix (Variable)
        [output]
        y   :   BxHxW[mono] Bx3HxW[color] matrix (Variable)
                [normal]:sigmoid,relu [whitened]:tanh
        """        
        x_patch = self.R_filter.Filter(self.x)
        errx_patch = self.R_filter.Filter(self.errx)
        #reshape 3BxHxW -> Bx3HW array
        x_patch_2D = F.reshape(x_patch,(self.B,self.C*self.Read_patch**2))
        errx_patch_2D = F.reshape(errx_patch,(self.B,self.C*self.Read_patch**2))
        
        self.c,self.h,enc_mu,enc_logsig2 = self.encode(self.c,self.h,x_patch_2D,errx_patch_2D)
        z = enc_mu
        self.c2,self.h2,inc_canvas,Wmean_x,Wmean_y,Wln_var,Wln_stride,Wln_gamma,Rmean_x,Rmean_y,Rln_var,Rln_stride,Rln_gamma = self.decode(self.c2,self.h2,z)#,aa,bb)

        self.W_filter.mkFilter(Wmean_x,Wmean_y,Wln_var,Wln_stride,Wln_gamma)
        self.R_filter.mkFilter(Rmean_x,Rmean_y,Rln_var,Rln_stride,Rln_gamma)

        inc_canvas = F.reshape(inc_canvas,(self.B*self.C,self.Write_patch,self.Write_patch))
        inc_canvas = self.W_filter.InvFilter(inc_canvas)
        self.canvas += inc_canvas
        y = F.relu(self.canvas+0.5)-F.relu(self.canvas-0.5)#F.sigmoid(self.canvas) #[normal]:sigmoid, [whitened]:tanh
        self.errx = self.x-y
        self.t += 1
        return y#,h2

