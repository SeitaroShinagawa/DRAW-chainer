from chainer import functions as F
from chainer import Variable
from util import XP
import numpy as np

class Filters:
    def __init__(self,C,height,width,patchsize):
        """
        [input]
        height: vertical size of image (int)
        width : horizonal size of image(int)
        patchsize : Attentional window(square) patch edge length(width==height)(int)
        """
        self.C = C
        self.height = height
        self.width = width
        self.patchsize = patchsize
        self.L_edge = width if width>height else height
         
        self.Harray = XP.farray(np.arange(height))   # (1,H) Variable
        self.Warray = XP.farray(np.arange(width))   # (1,W) Variable
        self.Parray = XP.farray(np.arange(patchsize,dtype=np.float32).reshape(patchsize,1)-0.5*(patchsize+1.0))   # (P,1) Variable

    def mkFilter(self,mean_x,mean_y,ln_var,ln_stride,ln_gamma):
        eps = 1e-8
        """
        make Attention Filters, need B Filters for a minibatch(composed of B data), shared between each color map
        [input]
        C: 1[mono],3[color]
        mean_x: Bx1[mono] Bx1[color] (chainer.Variable) 
        mean_y: Bx1[mono] Bx1[color] (chainer.Variable)
        ln_var: Bx1[mono] Bx1[color] (chainer.Variable)
        ln_stride: Bx1[mono] Bx1[color] (chainer.Variable)
        ln_gamma: Bx1[mono] Bx1[color] (Variable)
        [output]
        Fx : BxPxW[mono] 3BxPxW[color] matrix (Variable)
        Fy : BxPxH[mono] 3BxPxH[color] matrix (Variable)
        Gamma BxHxW[mono] 3BxHxW[color] (Variable)
        """
        P = self.patchsize
        B = mean_x.data.shape[0]
        H = self.height
        W = self.width

        mean_x = 0.5*(W+1.0)*(mean_x+1.0)  # (B,1)
        mean_y = 0.5*(H+1.0)*(mean_y+1.0) # (B,1)
        var = F.exp(ln_var)
        stride = (self.L_edge-1.0)/(P-1.0)*F.exp(ln_stride)
        gamma = F.exp(ln_gamma)

        mu_x = F.broadcast_to(mean_x,(P,B,1))       # (B,1) -> (P,B,1)
        mu_x = F.transpose(mu_x,(1,0,2))            #       -> (B,P,1)          
        mu_y = F.broadcast_to(mean_y,(P,B,1))       # (B,1) -> (P,B,1)
        mu_y = F.transpose(mu_y,(1,0,2))            #       -> (B,P,1) 
        stride = F.broadcast_to(stride,(P,B,1))    # (B,1) -> (P,B,1)
        stride = F.transpose(stride,(1,0,2))        #       -> (B,P,1)  
        var_x = F.broadcast_to(var,(P,W,B,1))       # (B,1) -> (P,W,B,1)
        var_x = F.transpose(var_x,(2,0,1,3))        #       -> (B,P,W,1)
        var_y = F.broadcast_to(var,(P,H,B,1))       # (B,1) -> (P,H,B,1)
        var_y = F.transpose(var_y,(2,0,1,3))        #       -> (B,P,H,1)

        mu_x = mu_x + F.broadcast_to(self.Parray,(B,P,1))*stride # (B,P,1)
        mu_y = mu_y + F.broadcast_to(self.Parray,(B,P,1))*stride # (B,P,1)
        
        mu_x = F.transpose(F.broadcast_to(mu_x,(self.width,B,P,1)),(1,2,0,3))
        mu_x = F.broadcast_to(self.Warray,(B,P,W)) - F.reshape(mu_x,(B,P,W))
        mu_y = F.transpose(F.broadcast_to(mu_y,(self.height,B,P,1)),(1,2,0,3))
        mu_y = F.broadcast_to(self.Harray,(B,P,H)) - F.reshape(mu_y,(B,P,H))
        var_x = F.reshape(var_x,(B,P,W)) # (B,P,W) -> (B,P,W)
        var_y = F.reshape(var_y,(B,P,H)) # (B,P,H) -> (B,P,H)

        x_square = -0.5 * (mu_x/var_x)**2   # (B,P,W)
        y_square = -0.5 * (mu_y/var_y)**2   # (B,P,H)
        x_gauss = F.exp(x_square)
        y_gauss = F.exp(y_square)
        
        xsum = F.sum(x_gauss,2)    # (B,P) 
        ysum = F.sum(y_gauss,2)    # (B,P)
        Zx_prev = F.transpose(F.broadcast_to(xsum,(W,B,P)), (1,2,0))
        enable = Variable(Zx_prev.data > eps)
        Zx = F.where(enable,Zx_prev,XP.fnonzeros(Zx_prev.data.shape,val=1.0)*eps)
        Zy_prev = F.transpose(F.broadcast_to(ysum,(H,B,P)), (1,2,0))
        enable = Variable(Zy_prev.data > eps)
        Zy = F.where(enable,Zy_prev,XP.fnonzeros(Zy_prev.data.shape,val=1.0)*eps)
        Fx = x_gauss/Zx
        Fy = y_gauss/Zy

        gamma_ = F.broadcast_to(gamma,(P,P,self.C,B,1)) # (B,1) -> (H,W,C,B,1) 
        Gamma = F.reshape(F.transpose(gamma_,(4,3,2,0,1)),(self.C*B,P,P))  #       -> (C*B,H,W)

        Fx_ = F.broadcast_to(Fx,(self.C,B,P,W))
        Fy_ = F.broadcast_to(Fy,(self.C,B,P,H))
        Fx = F.reshape(F.transpose(Fx_,(1,0,2,3)),(self.C*B,P,W))
        Fy = F.reshape(F.transpose(Fy_,(1,0,2,3)),(self.C*B,P,H))

        self.Fx = Fx
        self.Fy = Fy
        self.Gamma = Gamma

    def Filter(self,X):
        """
        [input]
        X : BxHxW[mono] 3BxHxW[color] matrix (Variable)
        Fx : BxPxW[mono] 3BxPxW[color] matrix (Variable)
        Fy : BxPxH[mono] 3BxPxH[color] matrix (Variable)
        Gamma BxHxW[mono] 3BxHxW[color] (Variable)

        [output]
        X_patch : BxPxP[mono] 3BxPxP[color] matrix (Variable) 
        """
        return self.Gamma*F.batch_matmul(F.batch_matmul(self.Fy,X),self.Fx,transb=True)
            
        
    def InvFilter(self,X_patch): 
        """
        [input]
        X_patch : BxPxP[mono] 3BxPxP[color] matrix (Variable)
        Fx : BxPxW[mono] BxPxW[color] matrix (Variable)
        Fy : BxPxH[mono] BxPxH[color] matrix (Variable)
        Gamma BxHxW[mono] 3BxHxW[color] (Variable)
        
        [output]
        X : BxHxW[mono] 3BxHxW[color] matrix (Variable)
        """
        return F.batch_matmul(F.batch_matmul(self.Fy,X_patch/self.Gamma,transa=True),self.Fx)


