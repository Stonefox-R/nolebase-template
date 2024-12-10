# 基础指令
---
## commit 
提交修改
## branch 
新建分支
## merge A 
将当前分支与A分支融合
## rebase A
将当前分支的父节点设置为A分支所值节点
### -f 参数
```cmd
git branch -f A B 
```
将A分支强制切换到B分支
## switch ==X== 
切换至名字为X的分支

## 远程仓库
---
### git clone
从云端复制仓库至本地
## git fetch
从云端更新所有修改状态，但不改变本地分支状态
## git pull
=git fetch+git merge
## git push
将本地修改推送至云端
# 高级功能
---
## checkout ==X== 
将HEAD指针指向X节点
## reset
本地仓库返回至指定数量的父节点
## revert
功能与reset类似，与reset的区别为可影响云端分支


以上功能已经可以应对日常开发需求，如有进一步的开发需要就去[这里](Learn Git Branching](https://learngitbranching.js.org/?locale=zh_CN)的链接继续学习一下
