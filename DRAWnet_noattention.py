from chainer import Chain 
import chainer.functions as F 
import chainer.links as L 
from util import XP 
 
class VAE_bernoulli_noattention(Chain): 
    def __init__(self,H_enc,z_dim,H_dec,height,width,C):  
        super().__init__( 
        enc_x_he = L.Linear(height*width*C,4*H_enc), 
        enc_errx_he = L.Linear(height*width*C,4*H_enc), 
        enc_he_he = L.Linear(H_enc,4*H_enc), 
        enc_he_mu = L.Linear(H_enc,z_dim), 
        enc_he_logsig2 = L.Linear(H_enc,z_dim), 
        dec_z_hd = L.Linear(z_dim,4*H_dec), 
        dec_hd_hd = L.Linear(H_dec,4*H_dec),
        dec_hd_he = L.Linear(H_dec,4*H_dec), 
        dec_hd_y = L.Linear(H_dec,height*width*C)) 
        self.H_enc = H_enc 
        self.z_dim = z_dim 
        self.H_dec = H_dec 
        self.height = height 
        self.width = width 
        self.C = C 
 
    def reset(self,image_batch): 
        """ 
        initialization 
        target image: x  BxHW[mono] Bx3HW[color] matrix (Variable) 
        current canvas: canvas  BxHW[mono] Bx3HW[color] matrix (Variable) 
        initial canvas: ini_ [normal] val=0.5 [whitened] val=0.0  BxHW[mono] Bx3HH[color] matrix (Variable) 
        error : target - current canvas 
        {encoder,decoder} {cell,hidden}: 0 
        [attentional window patch 
          [position](meanx,meany): center of each image(0.5*width,0.5*height) 
          [varience](ln_var):-6.9 (var=0.001) 
          [stride](ln_stride):1.1 (stride=3.0) 
          [gamma](ln_gamma):0.0 (gamma=1.0) 
          t: the number of processed minibatch 
        """ 
        self.zerograds() 
        self.B = image_batch.shape[0] 
        self.canvas = XP.fzeros((self.B,self.C*self.height*self.width)) 
        self.x = F.reshape(XP.farray(image_batch),(self.B,self.C*self.height*self.width)) 
        self.errx = self.x-XP.fnonzeros((self.B,self.C*self.height*self.width),val=0.0) 
        self.c = XP.fzeros((self.B, self.H_enc)) #initialize encoder cell 
        self.h = XP.fzeros((self.B, self.H_enc)) #initialize encoder hidden  
        self.c2 = XP.fzeros((self.B, self.H_dec)) #initialize decoder cell (decoder hidden is initialized in train_align_draw.py) 
        self.h2 = XP.fzeros((self.B, self.H_dec)) 
        self.t = 0 
 
    def encode(self,c_enc,h_enc,x,errx,h_dec): 
        h_in = self.enc_x_he(x) + self.enc_errx_he(errx) + self.enc_he_he(h_enc) + self.dec_hd_he(h_dec)
        c_enc,h_enc = F.lstm(c_enc,h_in) 
        mu = self.enc_he_mu(h_enc) 
        logsig2 = self.enc_he_logsig2(h_enc)*2 #original paper setting, but *2 is not necessary.
        return c_enc,h_enc,mu,logsig2 
 
    def decode(self,c_dec,h_dec,z): 
        h_in = self.dec_z_hd(z) + self.dec_hd_hd(h_dec) 
        c_dec,h_dec = F.lstm(c_dec,h_in) 
        return c_dec,h_dec,self.dec_hd_y(h_dec) 
 
    def free_energy_onestep(self): 
        """ 
        [input] 
        x    :  BxHxW[mono] Bx3HW[color] matrix (Variable) 
        errx :  BxHxW[mono] Bx3HW[color] matrix (Variable) 
        """ 
        self.c,self.h,enc_mu,enc_logsig2 = self.encode(self.c,self.h,self.x,self.errx,self.h2) 
        kl = F.gaussian_kl_divergence(enc_mu,enc_logsig2) 
        z = F.gaussian(enc_mu,enc_logsig2)
        z = enc_mu 
        self.c2,self.h2,inc_canvas = self.decode(self.c2,self.h2,z) 
         
        self.canvas += inc_canvas 
        y = F.sigmoid(self.canvas) 
        #y = F.relu(self.canvas+0.5)-F.relu(self.canvas-0.5)
        self.errx = self.x-y 
        self.t += 1         
        return y,kl 
 
    def generate_onestep(self): 
        """ 
        generate from middle layer 
        #call reset() first, but no relation between img_array[input] and generated image[output]  
        [input] 
        x    :  BxHW[mono] Bx3HW[color] matrix (Variable) 
        errx :  BxHW[mono] Bx3HW[color] matrix (Variable) 
        [output] 
        y   :   BxHW[mono] Bx3HW[color] matrix (Variable) 
                [normal]:sigmoid,relu [whitened]:tanh 
        """  
        zero_mat = XP.fzeros((self.B,self.z_dim)) 
        z = F.gaussian(zero_mat,zero_mat) #F.gaussian(mean,ln_var) 
        self.c2,self.h2,inc_canvas = self.decode(self.c2,self.h2,z) 
         
        self.canvas += inc_canvas 
        y = F.sigmoid(self.canvas)
        #y = F.relu(self.canvas+0.5)-F.relu(self.canvas-0.5)
        return y 
 
 
