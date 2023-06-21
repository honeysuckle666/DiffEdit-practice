# DiffEdit：基于扩散模型的图像编辑革命性成果

## 介绍

图像生成最近取得了巨大的进步，扩散模型允许为各种文本提示合成令人信服的图像。在本文中，我们提出了DiffEdit，这是一种利用文本条件扩散模型进行语义图像编辑任务的方法，其目标是基于文本查询编辑图像。语义图像编辑是图像生成的扩展，具有额外的约束，即生成的图像应尽可能与给定的输入图像相似。 当前基于扩散模型的编辑方法通常需要提供遮罩，通过将其视为条件修复任务，使任务变得更加容易。相比之下，我们的主要贡献是能够自动生成一个遮罩，突出显示输入图像中需要编辑的区域，方法是对比以不同文本提示为条件的扩散模型的预测。此外，我们依靠潜在推理来保留这些感兴趣区域中的内容，并与基于掩模的扩散显示出出色的协同作用。 DiffEdit在ImageNet上实现了最先进的编辑性能。此外，我们使用来自COCO数据集的图像以及基于文本的生成图像，在更具挑战性的环境中评估语义图像编辑。这本笔记本展示了论文["DIFFEDIT: 基于震荡的半透明图像编辑与遮罩引导"](*https://arxiv.org/abs/2210.11427#*)中所涉及的提示引导图像编辑方法的一个实验性变化。在这篇论文中，掩码是通过获取一个参考提示和一个目标提示来生成的。 例如，参考提示：'马'，目标：'斑马'）并使用noise_pred（马）-noise_pred（斑马）生成一个噪声掩码。



## 创建虚拟环境

我们创建一个名为torch的环境

打开anconda prompt创建名为torch的环境

conda create -n pachong python==3.8

进入虚拟环境

activate torch

安装torch

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

此处我们使用的的是coda11.7的版本

下载coda11.7并安装

安装opencv

```
conda install opencv
```



## 安装所需要的包

```
def install_dependencies():
    !pip install -qq numpy
    !pip install -qq matplotlib
    !pip install -qq fastai
    !pip install -qq accelerate
    !pip install -qq --upgrade transformers diffusers ftfy

 
install_dependencies()

```

## imports和安装

```
import os
import numpy as np

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from fastcore.all import concat
from fastai.basics import show_image,show_images
from fastdownload import FastDownload
from pathlib import Path

from PIL import Image
import torch, logging
from torch import autocast
from torchvision import transforms as tfms

from huggingface_hub import notebook_login
from transformers import CLIPTextModel,CLIPTokenizer
from transformers import logging
from diffusers import AutoencoderKL,UNet2DConditionModel,LMSDiscreteScheduler,StableDiffusionInpaintPipeline

import cv2

# 设置设备
torch_device = "cuda"

from torch.nn.functional import threshold

#总结张量
_s = lambda x: (x.shape,x.max(),x.min())



```

## 加载所需要的模型

有两种方式，第一种是通过hugging face认证，通过注册hugging face使用令牌来在线下载模型

但是这种方式可能会需要多次运行才能下载且连接不稳定模型较大

![]([readme\1.pn](https://github.com/honeysuckle666/DiffEdit-practice/blob/main/readme/1.png)



```
# 加载自动编码器模型，该模型将用于将潜伏物解码为图像空间。
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

#加载标记器和文本编码器，对文本进行标记和编码。
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# 用于生成潜像的UNet模型。
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

# 噪声调度器
# 超参数与训练模型时使用的参数一致
scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

# 到GPU去吧!
vae = vae.to(torch_device)
text_encoder = text_encoder.to(torch_device)
unet = unet.to(torch_device);
```

```
vae_magic = 0.18215 # 用比例项训练的vae模型，更接近单位方差
```

如果在线模式不能下载，可在本地创建文件夹，将所需要模型下载到本地

我们选用的是直接通过官网下载保存到本地

以下是模型网址

[CompVis/stable-diffusion-v-1-4-original at main (huggingface.co)](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/tree/main)

[openai/clip-vit-large-patch14 at main (huggingface.co)](https://huggingface.co/openai/clip-vit-large-patch14/tree/main)



## 在潜标和图像之间进行转换的函数
![a](https://github.com/honeysuckle666/DiffEdit-practice/blob/main/readme/a.png)
![b](https://github.com/honeysuckle666/DiffEdit-practice/blob/main/readme/b.png)
```
def image2latent(im):
    im = tfms.ToTensor()(im).unsqueeze(0)
    with torch.no_grad():
        latent = vae.encode(im.to(torch_device)*2-1);
    latent = latent.latent_dist.sample() * vae_magic      
    return latent
def decode_latent(latents):
    with torch.no_grad():
        return vae.decode(latents/vae_magic).sample
def latents2images(latents):
    latents = latents/vae_magic
    with torch.no_grad():
        imgs = vae.decode(latents).sample
    imgs = (imgs / 2 + 0.5).clamp(0,1)
    imgs = imgs.detach().cpu().permute(0,2,3,1).numpy()
    imgs = (imgs * 255).round().astype("uint8")
    imgs = [Image.fromarray(i) for i in imgs]
    return imgs
def get_embedding_for_prompt(prompt):
    max_length = tokenizer.model_max_length
    tokens = tokenizer([prompt],padding="max_length",max_length=max_length,truncation=True,return_tensors="pt")
    with torch.no_grad():
        embeddings = text_encoder(tokens.input_ids.to(torch_device))[0]
    return embeddings
def generate_noise_pred(prompts, im_latents, seed=32, g=0.15):
    height = 512                        # 稳定扩散的默认高度
    width = 512                         # 稳定扩散的默认宽度
    num_inference_steps = 30            # 去噪步骤的数量
    generator = torch.manual_seed(seed)   # 创建初始潜伏噪声的种子发生器

    uncond = get_embedding_for_prompt('')
    text = get_embedding_for_prompt(prompts)
    text_embeddings = torch.cat([uncond, text])

    #准备工作调度员
    scheduler.set_timesteps(num_inference_steps)

    # 预备潜伏
    if im_latents != None:
        # img2img
        #start_step = 10
        start_step = int(num_inference_steps * 0.5)
        timesteps = torch.tensor([scheduler.timesteps[-start_step]],device=torch_device)
        noise = torch.randn_like(im_latents)
        latents = scheduler.add_noise(im_latents,noise,timesteps=timesteps)
        latents = latents.to(torch_device).float()
    else:
        # 只是文本提示
        start_step = -1 # disable branching below
        latents = torch.randn((1,unet.in_channels,height//8,width//8))#,generator=generator)
        latents = latents.to(torch_device)
        latents = latents * scheduler.init_noise_sigma # scale to initial amount of noise for t0

    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, timesteps)
    with torch.no_grad():
        u,t = unet(latent_model_input, timesteps, encoder_hidden_states=text_embeddings).sample.chunk(2)
    pred_nonscaled= u + g*(t-u)/torch.norm(t-u)*torch.norm(u)
    pred = pred_nonscaled * torch.norm(u)/torch.norm(pred_nonscaled)
    return scheduler.step(pred, timesteps, latents).pred_original_sample
def generate_image_from_embedding(text_embeddings, im_latents, mask=None, seed=None, guidance_scale=0.15):
    height = 512                        # 稳定扩散的默认高度
    width = 512                         # 稳定扩散的默认宽度
    num_inference_steps = 30            # 去噪步骤的数量
    if seed is None: seed = torch.seed()
    generator = torch.manual_seed(seed)   # 创建初始潜伏噪声的种子发生器

    uncond = get_embedding_for_prompt('')
    text_embeddings = torch.cat([uncond, text_embeddings])

    # Prep Scheduler
    scheduler.set_timesteps(num_inference_steps)

    # Prep latents
    
    if im_latents != None:
        # img2img
        start_step = 10
        noise = torch.randn_like(im_latents)
        latents = scheduler.add_noise(im_latents,noise,timesteps=torch.tensor([scheduler.timesteps[start_step]]))
        latents = latents.to(torch_device).float()
    else:
        # just text prompts
        start_step = -1 # disable branching below
        latents = torch.randn((1,unet.in_channels,height//8,width//8))#,generator=generator)
        latents = latents.to(torch_device)
        latents = latents * scheduler.init_noise_sigma # scale to initial amount of noise for t0

    noisy_latent = latents.clone()
    # Loop
    noise_pred = None
    for i, tm in tqdm(enumerate(scheduler.timesteps),total=num_inference_steps,desc='Generating Masked Image for Prompt'):
        if i > start_step:
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, tm)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, tm, encoder_hidden_states=text_embeddings)["sample"]

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            
            u = noise_pred_uncond
            g = guidance_scale
            t = noise_pred_text

            if g > 0:
                pred_nonscaled= u + g*(t-u)/torch.norm(t-u)*torch.norm(u)
                pred = pred_nonscaled * torch.norm(u)/torch.norm(pred_nonscaled)
            else:
                pred = u

            noise_pred = pred
            
            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, tm, latents).prev_sample
            if mask is not None: 
                latents = latents*mask+im_latents*(1.0-mask)

    noise_pred = noisy_latent-latents
    return latents2images(latents)[0],noise_pred
def image2latentmask(im):
    im = tfms.ToTensor()(im).permute(1,2,0)
    m = im.mean(-1) # convert to grayscale
    m = (m > 0.5).float() # binarize to 0.0 or 1.0
    m = cv2.resize(m.cpu().numpy(),(64,64),interpolation=cv2.INTER_NEAREST)
    m = torch.tensor(m).to(torch_device)
    return m
```

## DiffEdit掩码生成的变化

这种DiffEdit掩蔽方法的变种不是采取参考提示和目标提示，而是采取单一的提示，然后使用同一提示的正向和 "负向指导刻度 "来创建一个对比性/负向的噪声预测。
Step1 计算图像编辑 mask  当对图像去噪时，在不同的文本条件下，扩散模型将产生不同的噪声估计值。我们可以根据估计值的不同之处，得出哪些图像区域与条件文本变化有关的信息。因此，噪声估计值之间的差异可以用来推断出一个 mask，以确定图像上哪些部分需要改变以达到文本要求。去除噪声预测中的极值，并通过对一组噪声的空间差异进行平均来稳定效果，然后将结果重新缩放到[0, 1]的范围内，并用一个阈值进行二进制化，就得到了图像编辑 mask。

Step2 DDIM 编码  使用 DDIM 编码 Er 对输入图像 xo 在时间步 r 上编码。这是在无条件模型下进行的，即使用条件文本为ɵ，在这一步没有使用文本输入。此步骤的中间结果 xt  将在 Step3 中用到。

Step3 基于 mask 的解码  首先，使用目标文本引导的扩散模型做采样得到 yt，并用 mask 来引导扩散过程。对于在 maskM 外的图像部分，编辑后原则上应与输入图像相同，所以通过用 DDIM 编码推断出的中间结果 xt 替换 M 外的像素值来指导扩散模型，这将通过解码自然地映射回原始像素。
```
# 一个来自diffedit论文的变体
# 用一个提示生成一个掩码

def generate_mask_variant(encoded,prompt1):
    masks=[]
    n = 20
    for i in tqdm(range(n),desc='Generating Mask for Prompts'):
        s = torch.seed() # 对两个噪声样本使用相同的种子很重要
        n1 = generate_noise_pred(prompt1,encoded,seed=s,g=0.15)
        n2 = generate_noise_pred(prompt1,encoded,seed=s,g=-0.15)

        i = threshold(decode_latent(n1-n2),0,0)+threshold(decode_latent(n2-n1),0,0)
        masks.append(i.squeeze().mean(axis=0).unsqueeze(dim=0))

#         plt.imshow(masks[-1].squeeze().cpu().numpy(),cmap='gray')
#         plt.show()

    all_masks = torch.cat(masks)
    all_masks = all_masks.sum(axis=0)
    all_masks = all_masks/3
    all_masks = all_masks.clamp(0.,1.)
    return all_masks

def diffedit_variant(im_encoded,from_prompt,to_prompt,seed=None):
    if seed is None: seed = torch.seed()
            
    all_masks = generate_mask_variant(im_encoded,from_prompt)
    
#     plt.imshow(all_masks.cpu().numpy(),cmap='gray')
#     plt.show()
    
    scaled_mask = torch.tensor(cv2.resize((all_masks>=0.5).float().cpu().numpy(),(64,64),interpolation=cv2.INTER_NEAREST),
                              device=torch_device)
    
#     plt.imshow(scaled_mask.cpu().numpy(),cmap='gray')
#     plt.show()
    
    torch.manual_seed(seed)
    from_emb = get_embedding_for_prompt(from_prompt)
    from_image,from_latent = generate_image_from_embedding(from_emb,im_encoded,seed=seed)
#     plt.imshow(from_image)
#     plt.show()
    
    torch.manual_seed(seed)
    to_emb = get_embedding_for_prompt(to_prompt)
    to_image,to_latent = generate_image_from_embedding(to_emb,im_encoded,scaled_mask,seed=seed)
    
    return to_image
```





## 示范掩码生成

使用 "negative guidance scale"方法

#加载一个图像

```
#Load the image
img = Image.open('./images/mario_scaled.jpg').resize((512,512));img
```

![][(readme\2.png)](https://github.com/honeysuckle666/DiffEdit-practice/blob/main/readme/2.png)

```
encoded = image2latent(img); encoded.shape
```

```
mario_mask = generate_mask_variant(encoded,'Mario')
plt.imshow(mario_mask.cpu().numpy(),cmap='gray')
```

生成灰度图像![](https://github.com/honeysuckle666/DiffEdit-practice/blob/main/readme/3.png)

显示二值化的图像比例掩码（512,512）。

```
# 显示二值化的图像比例掩码（512,512）。
plt.imshow((mario_mask>0.5).float().cpu().numpy(),cmap='gray')
```

![4](https://github.com/honeysuckle666/DiffEdit-practice/blob/main/readme/4.png)

缩放并显示二值化的潜标掩码（64,64）。

```
scaled_mask = cv2.resize((mario_mask>=0.5).float().cpu().numpy(),(64,64),interpolation=cv2.INTER_NEAREST)
plt.imshow(scaled_mask,cmap='gray')
```

![5](https://github.com/honeysuckle666/DiffEdit-practice/blob/main/readme/5.png)

用瓦里奥取代马里奥
从提示中画出带有掩码生成的变体

```
seed = 16321159919113515480
#seed=torch.seed()
print('seed:',seed)
mario_im = Image.open('./images/mario_scaled.jpg')
plt.imshow(mario_im)
plt.show()
mario_en = image2latent(mario_im)
plt.imshow(diffedit_variant(mario_en,'Mario','Wario',seed=seed))
plt.show()
```

![6](https://github.com/honeysuckle666/DiffEdit-practice/blob/main/readme/6.png)

用斑马代替马
从提示中画出带有掩码生成的变体

```
seed = 11794812278352456374
horse_im = Image.open('./images/horse_scaled.jpg').resize((512,512))
plt.imshow(horse_im)
plt.show()
horse_en = image2latent(horse_im)
plt.imshow(diffedit_variant(horse_en,'a horse image','a zebra image',seed=seed))
plt.show()
```

![7](https://github.com/honeysuckle666/DiffEdit-practice/blob/main/readme/7.png)

![8](https://github.com/honeysuckle666/DiffEdit-practice/blob/main/readme/8.png)

\##用 "长颈鹿 "替换 "马"。
从提示中画出带有掩码生成的变体

```
seed = 5327248292640123939
diffedit_variant(horse_en,'a horse image','a giraffe image',seed=seed)
```

![9](https://github.com/honeysuckle666/DiffEdit-practice/blob/main/readme/9.png)

用苹果代替草莓

从提示中画出带有掩码生成的变体      

```
seed = 5327248292640123939
berry_im = Image.open('./images/bowloberries_scaled.jpg')
plt.imshow(berry_im)
plt.show()
berry_en = image2latent(berry_im)
plt.imshow(diffedit_variant(berry_en,'strawberries','apples',seed=seed))
plt.show()
```

![10](https://github.com/honeysuckle666/DiffEdit-practice/blob/main/readme/10.png)

![11](https://github.com/honeysuckle666/DiffEdit-practice/blob/main/readme/11.png)
