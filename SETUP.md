# Colortina V2.8.1 安装与发布前验证

## 1. Python 环境

推荐 Python 3.10–3.12。

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

### Windows

优先双击 `Start_Colortina.bat`。脚本会在项目内创建 `runtime/` 独立环境并安装依赖。

手动安装：

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

## 2. 设备选择

默认选择顺序：CUDA → MPS → CPU。

```bash
COLORTINA_DEVICE=mps python main.py
COLORTINA_DEVICE=cpu python main.py
```

Windows PowerShell：

```powershell
$env:COLORTINA_DEVICE="cpu"
python main.py
```

## 3. 首次运行

首次自动上色会检查并下载 mc-v2 权重。下载失败时，界面和终端会显示具体错误。

权重目录：

```text
models/weights/
```

该目录不会提交到 Git。

## 4. 发布前检查

```bash
python -m compileall -q .
python -m unittest discover -s tests -v
```

检查 Git 中没有缓存和权重：

```bash
git status --short
git ls-files | grep -E '(__pycache__|\.pyc$|models/weights|runtime/)'
```

最后一条命令应没有输出。

## 5. 手动验收清单

1. 导入一张黑白漫画页并完成自动上色。
2. 确认自动上色按钮位于“上色”页面最底部。
3. 在编辑页面确认当前颜色文字和颜色块同步更新。
4. 切换“点采集”和“区域采集”两个勾选项，确认始终只有一个被选中。
5. 用区域画笔修改局部，确认画布立即刷新。
6. 用区域上色修改封闭区域和轻微断线区域，确认不会整页扩散。
7. 应用图片滤镜，确认手动补色仍保留。
8. 执行撤销和重做。
9. 保存并重新打开 `.ccproject`，确认结果、滤镜基础层和参数恢复。
10. 导出当前页和全部页面。

## 6. 通用图像验证脚本

```bash
python validate_effect_controls.py page1.png page2.png --output validation.md
```

脚本不依赖固定本地路径，也不需要 mc-v2 权重；它使用确定性的代理彩色图验证风格细调和手动提示强度。

## 7. 公共仓库发布注意事项

- 不提交模型权重、运行环境、缓存、用户图片和项目文件。
- 顶层项目许可证需要由仓库所有者决定；上游许可证不能自动替代你自己修改部分的许可声明。
- 不要上传无授权的漫画样本页。
