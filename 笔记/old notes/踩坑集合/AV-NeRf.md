## 驱动级别
### 关于cuda版本问题
![[Pasted image 20240912142352.png]]
这里的`cuda version`仅代表驱动发布时支持的最高版本。在[官方网站](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
中有提及cuda工具包对驱动版本的要求
### cuda，cuda toolkit ，与cudnn的关系
![[Pasted image 20240923143211.png]]
**广义上**CUDA指的是架构平台，通过`nvidia-smi`所查询到的cuda version指的是当前的driver api 版本。通常来说安装cuda toolkit时会默认安装cuda driver从而使其版本一致。
**但是！** 在conda中安装的cuda toolkit仅包含**runtime cuda**不包含**driver cuda**。此时就会导致版本不一致
![[Pasted image 20240923144120.png]]
[讲明白CUDA、CUDA toolkit、cuDNN、和NVCC关系 - 程序猿实验室-程序猿实验室 (sodebug.com)](https://www.sodebug.com/AI/cudatoolkit.html)
[显卡，显卡驱动,nvcc, cuda driver,cudatoolkit,cudnn到底是什么？这篇最清楚](https://www.cnblogs.com/marsggbo/p/11838823.html)
[1. Why CUDA Compatibility — CUDA Compatibility r555 documentation (nvidia.com)](https://docs.nvidia.com/deploy/cuda-compatibility/index.html)
## 系统级别

### **非root怎么装软件** （贼重要）

[[转载]Linux下非root用户如何安装软件 - 别再闹了 - 博客园 (cnblogs.com)](https://www.cnblogs.com/jiading/p/12055862.html)
[linux非root用户安装软件入门 - tlanyan (itlanyan.com)](https://itlanyan.com/work-with-linux-without-root-permission/)

### ~/.bashrc
本用户每次连接时默认运行的文件，一般存储环境变量
### ln
创建软连接

### source
使linux用户配置文件生效

### 如何让服务器翻墙？
1. 首先打开clash的局域网连接权限
2. 设置服务器连接代理
```shell
export https_proxy=http://10.11.8.73:7890 http_proxy=http://10.11.8.73:7890 all_proxy=socks5://10.11.8.73:7890
```

### pip显示可用版本问题
在安装nerfstudio的时候，他有一个`open3d`的依赖。安装他的时候出现了很多奇怪现象：
![[Pasted image 20240915190354.png]]
在pypi官网能够查询到其最新版本为0.18，但是pip指令就是无法安装
![[Pasted image 20240915190446.png]]
主要原因是系统不满足库的使用要求，可以在
![[Pasted image 20240915190534.png]]
查看到版本要求！！！

### **==关于pip软件列表中软件版本的说明==**

pypi中open3d的glibc必须 >=2.27，困扰了我好久。而后李老师发给我一个[page](https://github.com/quantaji/open3d-manylinux2014/tree/main)让我醍醐灌顶：在pypi中的都是已经编译好的程序，其各种要求本质上是在**链接** 阶段时使用到的工作做了要求。而这与编译时系统有关！


## Exploration！

### How pytorch save the model?
---
==reference linkings==
[Saving and Loading Models — PyTorch Tutorials 2.5.0+cu124 documentation](https://pytorch.org/tutorials/beginner/saving_loading_models.html?highlight=load_state_dict)

---

### Embedder-- help machine to imagin
For human, position imformation like (0,0) or (3,7) is clear and elegent.In the contrarary,machine can't understand what is coadinate main.So we need embedder to transform the  compact imfo in to a complex situaton.
```python
B = x["pos"].shape[0]                            
pos = self.pos_embedder(x["pos"]) # [B, 42]
pos = mydropout(pos, p=self.p, training=self.training)
freq =torch.linspace(-0.99,0.99,self.freq_num,device=x["pos"].device).unsqueeze(1) 
freq = self.freq_embedder(freq) # [F, 21]
```

```python
class embedding_module_log(nn.Module):
    def __init__(self, funcs=[torch.sin, torch.cos], num_freqs=20, max_freq=10, ch_dim=-1, include_in=True):
   
        super().__init__()
        self.functions = funcs
        self.num_functions = list(range(len(funcs)))
        self.freqs =torch.nn.Parameter(2.0**torch.from_numpy(np.linspace(start=0.0,stop=max_freq, num=num_freqs).astype(np.single)), requires_grad=False)

        self.ch_dim = ch_dim
        self.funcs = funcs
        self.include_in = include_in

    def forward(self, x_input):
        if self.include_in:
            out_list = [x_input]
        else:
            out_list = []
        for func in self.funcs:
            for freq in self.freqs:
                out_list.append(func(x_input*freq))
        return torch.cat(out_list, dim=self.ch_dim)
```

freq is a standard which represent the process of the embedder.

### What's the different between real-world audio-visual scene synthesis and sparse-view novel view synthesis?

