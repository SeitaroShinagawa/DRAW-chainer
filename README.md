# DRAW-chainer  
Reproduction of Deep Recurrent Attentive Writer (Recurrent VAE with attention) using chainer   

todo  
- [x] Minimum implementation  
- [x] Show result  
- [ ] Add visualizer  
- [ ] Complete description  

## Original Paper  
["DRAW: A Recurrent Neural Network For Image Generation"](http://jmlr.org/proceedings/papers/v37/gregor15.html)  

## Develop environment  
Python 3.5.2  
### (requirement libraries using pip)
chainer 1.16.0  
scipy    
pillow  
sklearn (only for downloading mnist)  

## How to run  
[no attention]
```  
python train_draw_noattention.py save_path [options]  
```   
[with attention]  
```  
python train_draw_withattention.py save_path [options]  
```  

(options example)  
with gpu, sequence time length = 32  
```  
python train_draw_withattention.py save_path -g 0 -T 32  
```
if you want to see more, use  
```
python train_draw_withattention.py --help  
```  

## Architecture  
![DRAWarchitecture](https://github.com/SeitaroShinagawa/DRAW-chainer/blob/master/imgs/DRAW_architecture.jpg)  

## results
### without attention (reconstruction of train data during learning)
![noA](https://github.com/SeitaroShinagawa/DRAW-chainer/blob/master/imgs/noA.png)  
epoch: 0 -> 20  
![noAfig](https://github.com/SeitaroShinagawa/DRAW-chainer/blob/master/imgs/learning_curve_noA.png)  

### with attention  (reconstruction of train data during learning)
![wA](https://github.com/SeitaroShinagawa/DRAW-chainer/blob/master/imgs/wA.png)  
epoch: 0 -> 20  
![wAfig](https://github.com/SeitaroShinagawa/DRAW-chainer/blob/master/imgs/learning_curve_wA.png)  
