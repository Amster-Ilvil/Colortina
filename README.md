<p align="center">
  <img src="assets/icon.png" width="128" alt="Colortina icon">
</p>

<h1 align="center">Colortina</h1>

<p align="center">
  本地运行的黑白漫画 AI 自动上色桌面工具<br>
  A fully-local desktop app for AI manga colorization
</p>

---

## 功能特性

- **一键自动上色**：基于 [manga-colorization-v2](https://github.com/qweasdd/manga-colorization-v2)（U-Net + SEResNeXt），首次运行自动下载权重
- **智能自动提示**：CLIP 零样本区域识别 → 按高光/中间调/阴影分层生成颜色提示，上色结果更自然
- **手动引导**：画布上直接涂抹颜色提示，局部重新生成；支持吸色、区域填充、撤销/重做
- **批量处理**：导入整个文件夹（自然排序）、PDF 自动拆页、拖拽导入图片/PDF/文件夹
- **硬件友好**：支持 CUDA / Apple Silicon (MPS) / CPU，显存不足自动降级；CUDA 上 fp16 加速
- **中英文界面**，可随时切换
## 效果展示

| 原图 | 自动上色 | 手动干预 |
|:---:|:---:|:---:|
| <img width="260" src="https://github.com/user-attachments/assets/818eae63-ad50-41a8-a487-a22eb5728441" /> | <img width="260" src="https://github.com/user-attachments/assets/3a10885f-bdc9-497e-9286-34ae8b94fb3d" /> | <img width="260" src="https://github.com/user-attachments/assets/09f4cfa7-2b2f-4f8e-a7fb-719e70d1a8b2" /> |

## 🚀 安装与启动

### 🍎 macOS / 🪟 Windows

以下步骤在 macOS 与 Windows 上的说明结构一致，针对各系统给出对应命令：

1. 进入项目目录

   macOS:
   ```bash
   cd ~/Colortina
   ```
   Windows (CMD):
   ```cmd
   cd %USERPROFILE%\Colortina
   ```
   Windows (PowerShell):
   ```powershell
   Set-Location (Join-Path $HOME 'Colortina')
   ```
   也可以直接运行 `Start_Colortina.bat`。

2. 创建并激活虚拟环境

   macOS:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
   Windows (CMD):
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```
   Windows (PowerShell):
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. 安装依赖（可选：使用国内镜像加速）

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

4. 启动程序

macOS:
```bash
python3 main.py
```
Windows:
```cmd
python main.py
```
也可以直接运行 `Start_Colortina.bat`。

首次上色时会自动下载模型权重（约 400 MB，来自 mc-v2 官方发布），存放在 `models/weights/`。

macOS（Apple Silicon）无需额外配置，自动使用 MPS GPU 加速；NVIDIA 显卡请按 [PyTorch 官网](https://pytorch.org/get-started/locally/) 安装对应 CUDA 版本的 torch。

## 使用

1. 导入图片 / PDF / 文件夹（或直接拖入窗口）
2. 点击「自动上色」
3. 对不满意的区域涂抹颜色提示后「重新生成」
4. 导出单页或全部页面

## 致谢与许可

Colortina 是一个整合型本地桌面工具，项目中的部分代码、模型结构、
权重格式兼容逻辑和工程思路参考或改写自多个开源项目。完整第三方清单
见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

主要致谢：

- [qweasdd/manga-colorization-v2](https://github.com/qweasdd/manga-colorization-v2) — 核心漫画自动上色网络、推理流程与官方 generator/denoiser 权重来源。
- [qweasdd/manga-colorization](https://github.com/qweasdd/manga-colorization) — 手动颜色提示工作流与早期交互式上色思路。
- [vikast908/ColorComic](https://github.com/vikast908/ColorComic) — 提示点 API、引导式自动提示、分格/分块推理等工程思路；其 MIT 许可证文本保留在 `LICENSE_ColorComic_MIT.txt`。
- [xiaogdgenuine/Manga-Colorization-FJ](https://github.com/xiaogdgenuine/Manga-Colorization-FJ) — 跳过已上色页面、权重格式兼容、发布适配等思路。
- [ljsabc/MangaLineExtraction_PyTorch](https://github.com/ljsabc/MangaLineExtraction_PyTorch) — 矩形/套索区域的漫画结构线提取网络，MIT License。
- [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) 与 [XPixelGroup/BasicSR](https://github.com/XPixelGroup/BasicSR) — anime6B 超分模型与 RRDBNet/BasicSR 架构参考，分别采用 BSD-3-Clause 与 Apache-2.0。
- [Matias Tassano 的 FFDNet](https://github.com/cszn/FFDNet) / IPOL FFDNet 相关实现 — mc-v2 denoiser 中包含的 FFDNet 代码声明为 GPLv3-or-later；分发或再利用时请一并遵守其许可证。
- [nagadomi/lbpcascade_animeface](https://github.com/nagadomi/lbpcascade_animeface) — OpenCV 动漫脸检测级联文件来源，MIT 许可证文本保留在 `LICENSE_lbpcascade_animeface_MIT.txt`。

运行依赖包括 PyTorch / torchvision、PySide6、OpenCV、NumPy、Pillow、
Transformers、ONNX Runtime、scikit-image、gdown、PyMuPDF、Pydantic 等；
这些依赖均保留各自许可证。尤其是 PyMuPDF 同时提供 AGPL/commercial 授权，
重新分发或商业使用前请自行确认合规性。

## 免责声明

本工具仅供个人学习与研究使用。请勿将上色结果用于侵犯原作者版权的用途。
