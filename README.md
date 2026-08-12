<p align="center">
  <img src="assets/readme-banner.jpg" width="100%" alt="Colortina — Manga-Colorization-v2">
</p>

<h1 align="center">Colortina</h1>

<p align="center">基于 <strong>manga-colorization-v2</strong> 的本地黑白漫画 AI 自动上色桌面工具，面向 macOS、Windows 与 Linux。</p>

<p align="center">
  <img alt="macOS" src="https://img.shields.io/badge/macOS-Apple%20Silicon%20%2F%20Intel-black?logo=apple">
  <img alt="Windows" src="https://img.shields.io/badge/Windows-x64-0078D4?logo=windows11&logoColor=white">
  <img alt="Linux" src="https://img.shields.io/badge/Linux-x64-FCC624?logo=linux&logoColor=black">
  <a href="https://github.com/qweasdd/manga-colorization-v2"><img alt="manga-colorization-v2" src="https://img.shields.io/badge/upstream-manga--colorization--v2-2f6feb"></a>
  <img alt="PySide6" src="https://img.shields.io/badge/UI-PySide6-41CD52?logo=qt&logoColor=white">
  <img alt="Local AI" src="https://img.shields.io/badge/AI-Local%20Inference-2f6feb">
</p>

> 本地运行 · AI 自动上色 · 手动颜色提示 · 局部重绘 · 头发纠色 · 批量处理 · Apple Silicon MPS / CUDA / CPU

## 下载

优先从仓库 **Releases** 下载对应平台的最新构建：

- macOS Apple Silicon
- macOS Intel
- Windows x64
- Linux x64
- SHA-256 校验文件

当前发布核心版本：**v5.13.27**

## 主要特性

- **一键自动上色**：基于 [manga-colorization-v2](https://github.com/qweasdd/manga-colorization-v2) 的生成器与降噪流程，首次使用按需准备模型权重。
- **智能自动提示**：使用 CLIP 零样本区域识别，根据高光、中间调和阴影生成颜色提示。
- **手动颜色引导**：支持画笔、吸色、区域填充、撤销/重做，并可将人工颜色提示重新送入模型。
- **AI 选区重上色**：对选定区域恢复黑白后重新推理，只覆盖目标区域，减少叠色。
- **头发纠色工作区**：独立头发选区、增减笔、完整页面上下文检测与局部纠色。
- **输出版本管理**：原图 / AI 结果 / 编辑后三段式预览，支持恢复与分别导出。
- **批量处理**：支持图片、文件夹和 PDF 导入，文件夹自然排序，支持拖拽。
- **跨平台硬件加速**：Apple Silicon 使用 MPS，NVIDIA 可使用 CUDA，显存不足时可降级到 CPU。
- **中英文界面**：可在应用内切换语言。
- **本地处理**：图片、项目和导出结果默认只在本机处理，不上传到第三方服务。

## v5.13.27 更新重点

- **输出页重构**：原图 / AI 结果 / 编辑后三段式预览切换；不可用版本会明确禁用，不再静默回退到原图。
- **输出操作分区**：预览、版本恢复和导出采用独立卡片；“导出全部”根据整个项目是否存在结果启用，而不是只看当前页。
- **头发纠色工作区**：底部增加固定的“自动上色（应用头发纠色）”入口，并纳入统一 busy-state 锁定与可用状态刷新。
- **中央预览精简**：移除预览区左上角当前文件名，减少无效视觉占用。
- **右侧 UI 精装修**：统一设置卡片、间距、控件尺寸、主次按钮层级和紧凑输出布局。
- **稳定性加强**：后台任务期间锁定结构操作；增强 Qt 控件重建/销毁后的状态更新安全；同步页选择统计、任务状态和工作线程生命周期。
- **AI 头发解析与局部纠色**：保留多后端、独立头发选区、增减笔、完整页面上下文检测后裁剪和缓存复用等能力。
- **macOS / Apple Silicon**：继续支持 MPS；macOS 打包版本将模型权重和可写数据放入用户应用数据目录，不写入只读应用包。

## 效果展示

| 原图 | 自动上色 | 手动干预 |
|:---:|:---:|:---:|
| <img width="260" src="https://github.com/user-attachments/assets/818eae63-ad50-41a8-a487-a22eb5728441" /> | <img width="260" src="https://github.com/user-attachments/assets/3a10885f-bdc9-497e-9286-34ae8b94fb3d" /> | <img width="260" src="https://github.com/user-attachments/assets/09f4cfa7-2b2f-4f8e-a7fb-719e70d1a8b2" /> |

## 安装与启动

### macOS

```bash
git clone https://github.com/Amster-Ilvil/Colortina.git
cd Colortina
python3 -m venv venv
source venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python3 main.py
```

Apple Silicon 会自动优先使用 MPS GPU 加速。

### Windows

```powershell
git clone https://github.com/Amster-Ilvil/Colortina.git
cd Colortina
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
python main.py
```

也可以直接运行：

```text
Start_Colortina.bat
```

NVIDIA 显卡请按照 PyTorch 官方说明安装匹配 CUDA 的 torch 版本。

### 模型权重

首次实际运行相关 AI 功能时，会按需准备模型权重。mc-v2 官方 generator/denoiser 权重保存在本机模型目录，不作为公开仓库源码提交。

## 使用流程

1. 导入图片、PDF 或图片文件夹。
2. 运行自动上色。
3. 对需要修正的区域使用颜色提示、区块工具或头发纠色。
4. 必要时运行局部 / AI 选区重上色。
5. 在输出页比较原图、AI 结果与编辑后版本。
6. 导出当前页或全部页面。

## 平台与加速

| 能力 | macOS Apple Silicon | macOS Intel | Windows x64 | Linux x64 |
|---|:---:|:---:|:---:|:---:|
| 主界面 | ✓ | ✓ | ✓ | ✓ |
| 自动上色 | ✓ | ✓ | ✓ | ✓ |
| Apple MPS | ✓ | — | — | — |
| NVIDIA CUDA | — | — | ✓ | ✓ |
| CPU 回退 | ✓ | ✓ | ✓ | ✓ |
| 发布构建 | ✓ | ✓ | ✓ | ✓ |

## 构建与发布

现有 GitHub Actions 发布流程覆盖：

- Windows x64
- macOS Apple Silicon
- macOS Intel
- Linux x64
- 发布文件 SHA-256 校验

公开仓库不应包含模型权重、用户图片、项目文件、导出结果、缓存、凭据、令牌、私钥或真实用户主目录路径。

## 隐私说明

- 漫画图片、颜色提示和生成结果默认在本机处理。
- 模型权重和可写运行数据保存在用户应用数据目录或本地模型目录。
- 不把用户素材作为项目源码提交。
- 发布流程继续执行隐私审计，检查常见密钥、本机路径和不应进入公开仓库的运行数据。

## 致谢与许可

Colortina 是一个整合型本地桌面工具，项目中的部分代码、模型结构、权重格式兼容逻辑和工程思路参考或改写自多个开源项目。完整第三方清单见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

主要致谢：

- [qweasdd/manga-colorization-v2](https://github.com/qweasdd/manga-colorization-v2) — 核心漫画自动上色网络、推理流程与官方 generator/denoiser 权重来源。
- [qweasdd/manga-colorization](https://github.com/qweasdd/manga-colorization) — 手动颜色提示工作流与早期交互式上色思路。
- [vikast908/ColorComic](https://github.com/vikast908/ColorComic) — 提示点 API、引导式自动提示、分格/分块推理等工程思路；其 MIT 许可证文本保留在 `LICENSE_ColorComic_MIT.txt`。
- [xiaogdgenuine/Manga-Colorization-FJ](https://github.com/xiaogdgenuine/Manga-Colorization-FJ) — 跳过已上色页面、权重格式兼容、发布适配等思路。
- [ljsabc/MangaLineExtraction_PyTorch](https://github.com/ljsabc/MangaLineExtraction_PyTorch) — 矩形/套索区域的漫画结构线提取网络，MIT License。
- [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) 与 [XPixelGroup/BasicSR](https://github.com/XPixelGroup/BasicSR) — anime6B 超分模型与 RRDBNet/BasicSR 架构参考，分别采用 BSD-3-Clause 与 Apache-2.0。
- [Matias Tassano 的 FFDNet](https://github.com/cszn/FFDNet) / IPOL FFDNet 相关实现 — mc-v2 denoiser 中包含的 FFDNet 代码声明为 GPLv3-or-later；分发或再利用时请一并遵守其许可证。
- [nagadomi/lbpcascade_animeface](https://github.com/nagadomi/lbpcascade_animeface) — OpenCV 动漫脸检测级联文件来源，MIT 许可证文本保留在 `LICENSE_lbpcascade_animeface_MIT.txt`。

运行依赖包括 PyTorch / torchvision、PySide6、OpenCV、NumPy、Pillow、Transformers、ONNX Runtime、scikit-image、gdown、PyMuPDF、Pydantic 等；这些依赖均保留各自许可证。尤其是 PyMuPDF 同时提供 AGPL/commercial 授权，重新分发或商业使用前请自行确认合规性。

## 免责声明

本工具仅供个人学习与研究使用。请勿将上色结果用于侵犯原作者版权的用途。

## About 推荐内容

**Description**

> 本地黑白漫画 AI 自动上色工具｜manga-colorization-v2｜手动颜色提示与局部重绘｜macOS / Windows / Linux｜Apple Silicon MPS / CUDA

**Topics**

`manga` `manga-colorization` `comic` `colorization` `ai` `pytorch` `pyside6` `macos` `apple-silicon` `mps` `windows` `linux` `cuda` `local-ai` `image-processing`
