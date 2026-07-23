<p align="center">
  <img src="assets/icon.png" width="128" alt="Colortina icon">
</p>

<h1 align="center">Colortina V2.8.1</h1>

<p align="center">
  本地运行的黑白漫画 AI 自动上色、自然调色与手动修色工具
</p>

## 当前功能

- 基于 manga-colorization-v2 的本地自动上色。
- 支持图片、文件夹、PDF 和拖拽导入，可批量处理并跳过已上色页面。
- 内置三种风格：
  - 原始 mc-v2
  - 淡彩水墨
  - 淡彩水墨（极淡）
- “上色”界面内提供风格细调：颜色量、亮度、冷暖、亮部保色、柔化、块面感，以及一键重置。
- 自然图片滤镜采用保边缘明暗基底与细节层分离，支持亮度、对比度、饱和度、冷暖、阴影和高光。
- 编辑工具：
  - 区域画笔
  - 区域上色
  - 吸管点采集（邻域中值）
  - 吸管区域采集（区域中值）
  - 当前颜色与实际风格匹配补色预览
  - 撤销 / 重做
- 手动编辑会同步当前结果与滤镜基础层，重新应用滤镜不会覆盖补色。
- `.ccproject` 保存页面顺序、上色结果、滤镜基础层、手动提示和界面参数。
- Windows 一键启动脚本会创建独立运行环境，不污染系统 Python。

## 推荐环境

- Python 3.10–3.12
- macOS Apple Silicon、Windows 10/11 或 Linux
- 16 GB 内存可运行普通 mc-v2 工作流
- Apple Silicon 默认优先使用 MPS；NVIDIA GPU 默认优先使用 CUDA；否则使用 CPU

## 安装

### macOS / Linux

```bash
git clone <你的仓库地址>
cd Colortina
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

### Windows

可直接双击：

```text
Start_Colortina.bat
```

也可以手动安装：

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

首次自动上色会下载 mc-v2 权重。权重保存于 `models/weights/`，该目录已加入 `.gitignore`。

## 基本使用流程

1. 导入黑白漫画图片、文件夹或 PDF。
2. 在“上色”页面选择风格和细调参数。
3. 在上色页面最底部点击“自动上色”。
4. 打开“编辑”页面：
   - 选择区域画笔或区域上色；
   - 点击颜色块选择颜色，或使用吸管采色；
   - 点采集和区域采集通过两个可见勾选项切换；
   - 当前选择色和实际补色会直接显示。
5. 需要整体调整时，在“图片滤镜”中设置参数并选择当前页或全部已上色页。
6. 保存项目或导出页面。

## 测试

```bash
python -m compileall -q .
python -m unittest discover -s tests -v
```

当前仓库定义 128 项自动化检查。没有安装 PySide6 的纯计算测试环境会跳过 2 项离屏 Qt 窗口测试，其余核心、编辑、滤镜、项目保存和静态 UI 检查均可运行。

可用自有图片执行通用效果验证：

```bash
python validate_effect_controls.py page1.png page2.png --output validation.md
```

## 模型与缓存

以下内容不应提交到 Git：

- `models/weights/`
- `runtime/`
- `__pycache__/`
- `*.pyc`
- 用户项目与导出结果

## 已知边界

- 自动上色质量仍受 mc-v2 模型能力和原始线稿质量影响。
- 断线严重、线稿极淡或大面积无边界区域时，区域上色会使用受限回退，优先避免整页扩散。
- 完整 GPU/MPS 推理、真实模型权重下载和桌面视觉效果，需要在安装完整依赖的目标电脑上最终验收。

## 许可

仓库包含上游组件的许可证文件：

- `LICENSE_ColorComic_MIT.txt`
- `LICENSE_lbpcascade_animeface_MIT.txt`

公开发布前，请为你自己的修改部分选择并添加顶层 `LICENSE`，同时遵守模型权重条款和所处理漫画内容的版权要求。
