#!usr/bin/python3
# require chainer 1.7 or newer
# pip install chianer matplotlib pillow sklearn scipy
import os
import math
import sys
import time
import random
import json
from argparse import ArgumentParser
from chainer import functions as F
from chainer import optimizers,optimizer,serializers,Variable,cuda
from DRAWnet_noattention import VAE_bernoulli_noattention
import numpy as np
from data_util_mnist import MNIST
from util import XP
import matplotlib.pyplot as plt
from matplotlib import cm

def parse_args():
    vae_H_enc_def = 256 
    vae_z_dim_def = 100
    vae_H_dec_def = 256
    maxepoch_def = 20 
    batchsize_def = 100 
    lr_def = 0.001
    wdecay_def = 0.0
    gpu_def = -1
    times_def = 10 
    alpha_def = 1.0
    
    p = ArgumentParser(
    description='DRAW Deep Recurrent Attentive Writer',
    usage=
        '\n  e.g.) python train_draw_noattention.py save_path [option... -g 0 -T 32 etc.]'
        '\n  %(prog)s -h',
    )
    #p.add_argument('data_path', help='[in] training corpus')
    p.add_argument('save_path', help='[out] model file')
    p.add_argument('-M','--model_path',nargs='?',default=None,metavar=None,type=str,
        help='model_path(to continue saved model)')
    p.add_argument('-I', '--maxepoch', default=maxepoch_def, metavar='INT', type=int,
        help='the number of training epoch (default: %d)' % maxepoch_def)
    p.add_argument('-B', '--batchsize', default=batchsize_def, metavar='INT', type=int,
        help='minibatch size (default: %d)' % batchsize_def)
    p.add_argument('-L', '--lr', default=lr_def, metavar='FLOAT', type=float,
        help='learning rate (default: %f)' % lr_def)
    p.add_argument('-W', '--weightdecay', default=wdecay_def, metavar='FLOAT', type=float,
        help='weight decay (default: %f)' % wdecay_def)
    p.add_argument('-g', '--gpu',default=gpu_def, metavar='INT',type=int,
        help='gpu mode (-1:CPU 0:GPU) (default: %d)' % gpu_def)
    p.add_argument('-Ve', '--vae_enc',default=vae_H_enc_def, metavar='INT',type=int,
        help='VAE encoder hidden size (default: %d)' % vae_H_enc_def)
    p.add_argument('-Vz', '--vae_z',default=vae_z_dim_def, metavar='INT',type=int,
        help='VAE latent z dimension size (default: %d)' % vae_z_dim_def)
    p.add_argument('-Vd', '--vae_dec',default=vae_H_dec_def, metavar='INT',type=int,
        help='VAE decoder hidden size (default: %d)' % vae_H_dec_def)
    p.add_argument('-T', '--times',default=times_def, metavar='INT',type=int,
        help='DRAW sequence length (default: %d)' % times_def) 
    p.add_argument('-A', '--alpha',default=alpha_def, metavar='FLOAT',type=float,
        help='VAE KL coefficient (default: %f)' % alpha_def) 
    args = p.parse_args()

    # check args
    try:
        if (args.maxepoch < 1): raise ValueError('you must set --epoch >= 1')
        if (args.batchsize < 1): raise ValueError('you must set --batchsize >= 1')
        if (args.lr < 0): raise ValueError('you must set --lr >= 0')
        if (args.weightdecay < 0): raise ValueError('you must set --weightdecay >= 0')
        if (args.vae_enc < 1): raise ValueError('you must set --vae_enc >= 1')
        if (args.vae_dec < 1): raise ValueError('you must set --vae_dec >= 1')
        if (args.vae_z < 1): raise ValueError('you must set --vae_z >= 1')
        if (args.times < 2): raise ValueError('you must set --times >= 2')
        if (args.alpha < 0): raise ValueError('you must set --alpha >= 0')
    except Exception as ex:
        p.print_usage(file=sys.stderr)
        print(ex, file=sys.stderr)
        sys.exit()

    return args

def trace(path,src):
    with open(path,'a') as f:
        print(src,file=f)
    print(src)

def Bernoulli_nll_wesp(x,y,eps): #eps is epsilon
    return -F.sum(x*F.log(y+eps)+(1-x)*F.log(1-y+eps))
    
def main():
  args = parse_args()
  XP.set_library(args)
  date=time.localtime()[:6]
  D=[]
  for i in date:
    D.append(str(i))
  D="_".join(D)

  save_path=args.save_path
  if os.path.exists(save_path)==False:
    os.mkdir(save_path)

  if args.model_path!=None:
    print("continue existed model!! load recipe of {}".format(args.model_path))
    with open(args.model_path+'/recipe.json','r') as f:
      recipe=json.load(f)
    vae_enc=recipe["network"]["IM"]["vae_enc"]
    vae_z=recipe["network"]["IM"]["vae_z"]
    vae_dec=recipe["network"]["IM"]["vae_dec"]
    times=recipe["network"]["IM"]["times"]
    alpha=recipe["network"]["IM"]["KLcoefficient"]
    
    batchsize=recipe["setting"]["batchsize"]
    maxepoch=args.maxepoch
    weightdecay=recipe["setting"]["weightdecay"]
    grad_clip=recipe["setting"]["grad_clip"]
    cur_epoch=recipe["setting"]["cur_epoch"]+1
    ini_lr=recipe["setting"]["initial_learningrate"]
    cur_lr=recipe["setting"]["cur_lr"]            

    with open(args.model_path+"/../trainloss.json",'r') as f:
      trainloss_dic=json.load(f)
    with open(args.model_path+"/../valloss.json",'r') as f:
      valloss_dic=json.load(f)

  else:
    vae_enc=args.vae_enc
    vae_z=args.vae_z
    vae_dec=args.vae_dec
    times=args.times
    alpha=args.alpha
    batchsize=args.batchsize
    maxepoch=args.maxepoch
    weightdecay=args.weightdecay
    grad_clip=5
    cur_epoch=0
    ini_lr=args.lr
    cur_lr=ini_lr
    trainloss_dic={}
    valloss_dic={}

  print('this experiment started at :{}'.format(D))
  print('***Experiment settings***')
  print('[IM]vae encoder hidden size :{}'.format(vae_enc))
  print('[IM]vae hidden layer size :{}'.format(vae_z))
  print('[IM]vae decoder hidden layer size :{}'.format(vae_dec)) 
  print('[IM]sequence length:{}'.format(times)) 
  print('max epoch :{}'.format(maxepoch))
  print('mini batch size :{}'.format(batchsize))
  print('initial learning rate :{}'.format(cur_lr))
  print('weight decay :{}'.format(weightdecay))
  print("optimization by :{}".format("Adam"))
  print("VAE KL coefficient:",alpha)
  print('*************************') 
  
  vae = VAE_bernoulli_noattention(vae_enc,vae_z,vae_dec,28,28,1)
  opt = optimizers.Adam(alpha = cur_lr)
  opt.setup(vae)
  if args.model_path!=None:
    print('loading model ...')
    serializers.load_npz(args.model_path + '/VAEweights', vae)
    serializers.load_npz(args.model_path + '/optimizer', opt)
  else:
    print('making [[new]] model ...')
    for param in vae.params():
      data = param.data
      data[:] = np.random.uniform(-0.1, 0.1, data.shape)
  opt.add_hook(optimizer.GradientClipping(grad_clip))
  opt.add_hook(optimizer.WeightDecay(weightdecay))  

  if args.gpu >= 0 :
    vae.to_gpu()

  mnist=MNIST(binarize=True)
  train_size = mnist.train_size
  test_size = mnist.test_size
 
  eps = 1e-8
  for epoch in range(cur_epoch+1, maxepoch+1):
    print('\nepoch {}'.format(epoch))
    LX = 0.0
    LZ = 0.0
    counter = 0
    for iter,(img_array,label_array) in enumerate(mnist.gen_train(batchsize,Random=True)):
        B = img_array.shape[0]
        Lz = XP.fzeros(())
        vae.reset(img_array)
        
        #first to T-1 step
        for j in range(times-1):
            y,kl = vae.free_energy_onestep()
            Lz_i = alpha*kl
            Lz += Lz_i
        #last step
        j+=1
        y,kl = vae.free_energy_onestep()
        Lz_i = alpha*kl
        Lz += Lz_i
        Lx = Bernoulli_nll_wesp(vae.x,y,eps)
        
        LZ += Lz.data
        LX += Lx.data
 
        loss = (Lx+Lz)/batchsize
        loss.backward()
        opt.update()

        counter += B
        sys.stdout.write('\rnow training ...  epoch {}, {}/{}  '.format(epoch,counter,mnist.train_size))
        sys.stdout.flush()
        if (iter+1) % 100 == 0:
          print("({}-th batch mean loss) Lx:%03.3f Lz:%03.3f".format(counter) % (Lx.data/B,Lz.data/B))

    img_array = cuda.to_cpu(y.data)
    im_array = img_array.reshape(batchsize*28,28)
    img = im_array[:28*5]
    plt.clf()
    plt.imshow(img,cmap=cm.gray)
    plt.colorbar(orientation='horizontal')
    plt.savefig(save_path+"/"+"img{}.png".format(epoch))

    trace(save_path+"/trainloss.txt","epoch {} Lx:{} Lz:{} Lx+Lz:{}".format(epoch,LX/train_size,LZ/train_size,(LX+LZ)/train_size))            	
    trainloss_dic[str(epoch).zfill(3)]={
                    "Lx":float(LX/train_size),
                    "Lz":float(LZ/train_size),
                    "Lx+Lz":float((LX+LZ)/train_size)}
    with open(save_path+"/trainloss.json",'w') as f:
        json.dump(trainloss_dic,f,indent=4)   

    print('save model ...')
    prefix = save_path+"/"+str(epoch).zfill(3)
    if os.path.exists(prefix)==False:
        os.mkdir(prefix)        
    serializers.save_npz(prefix + '/VAEweights', vae) 
    serializers.save_npz(prefix + '/optimizer', opt)
    print('save recipe...')
    recipe_dic = {
    "date":D,
    "setting":{
        "maxepoch":maxepoch,
        "batchsize":batchsize,
        "weightdecay":weightdecay,
        "grad_clip":grad_clip,
        "opt":"Adam",
        "initial_learningrate":ini_lr,
        "cur_epoch":epoch,
        "cur_lr":cur_lr},
    "network":{
        "IM":{
            "x_size":784,
            "vae_enc":vae_enc,
            "vae_z":vae_z,
            "vae_dec":vae_dec,
            "times":times,
            "KLcoefficient":alpha},
            },
            }
    with open(prefix+'/recipe.json','w') as f:
      json.dump(recipe_dic,f,indent=4)
           
    if epoch % 1 == 0:
        print("\nvalidation step")
        LX = 0.0
        LZ = 0.0        
        counter = 0
        for iter,(img_array,label_array) in enumerate(mnist.gen_test(batchsize)):
            B = img_array.shape[0]
            Lz = XP.fzeros(())
            vae.reset(img_array)
            
            #first to T-1 step
            for j in range(times-1):
                y,kl = vae.free_energy_onestep()
                Lz_i = alpha*kl
                Lz += Lz_i           
            #last step
            j+=1
            y,kl = vae.free_energy_onestep()
            Lz_i = alpha*kl
            Lz += Lz_i  
            Lx = Bernoulli_nll_wesp(vae.x,y,eps)

            LZ += Lz.data.reshape(())
            LX += Lx.data.reshape(())

            counter += B
            sys.stdout.write('\rnow testing ...  epoch {}, {}/{}  '.format(epoch,counter,test_size))
            sys.stdout.flush()
        print("")
        trace(save_path+"/valloss.txt","epoch {} Lx:{} Lz:{} Lx+Lz:{}".format(epoch,LX/test_size,LZ/test_size,(LX+LZ)/test_size))                  		
        valloss_dic[str(epoch).zfill(3)]={
                        "Lx":float(LX/test_size),
						"Lz":float(LZ/test_size),
						"Lx+Lz":float((LX+LZ)/test_size)}
        with open(save_path+"/valloss.json",'w') as f:
            json.dump(valloss_dic,f,indent=4)

        img_array = cuda.to_cpu(y.data)
        im_array = img_array.reshape(batchsize*28,28)
        img = im_array[:28*5]
        plt.clf()
        plt.imshow(img,cmap=cm.gray)
        plt.colorbar(orientation='horizontal')
        plt.savefig(save_path+"/"+"img_test{}.png".format(epoch))
  print('finished.') 
    
if __name__ == '__main__':
  main()
