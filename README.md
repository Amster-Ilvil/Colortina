<p align="center">
  <img src="assets/icon.png" width="128" alt="Colortina icon">
</p>

<h1 align="center">Colortina</h1>

<p align="center">
  本地运行的黑白漫画 AI 自动上色桌面工具（无云端、无 API）<br>
  A fully-local desktop app for AI manga colorization — no cloud, no API keys.
</p>

---

## 功能特性

- **一键自动上色**：基于 [manga-colorization-v2](https://github.com/qweasdd/manga-colorization-v2)（U-Net + SEResNeXt），首次运行自动下载权重
- **智能自动提示**：CLIP 零样本区域识别 → 按高光/中间调/阴影分层生成颜色提示，上色结果更自然
- **手动引导**：画布上直接涂抹颜色提示，局部重新生成；支持吸色、区域填充、撤销/重做
- **批量处理**：导入整个文件夹（自然排序）、PDF 自动拆页、拖拽导入图片/PDF/文件夹
- **跳过已上色页面**：自动检测彩页并跳过，混合黑白/彩色的整本漫画可直接批量处理
- **跨页颜色一致性**：CharacterMemory 让同一角色在整本书中保持发色等一致
- **自定义风格**：从参考图提取风格档案（.ccstyle），保存/加载/复用
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
   cd ~/ComicColorerAI     # 改成你的实际路径
   ```
   Windows (CMD 或 PowerShell):
   ```powershell
   cd C:\Users\你的用户名\ComicColorerAI
   ```

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

首次上色时会自动下载模型权重（约 400 MB，来自 mc-v2 官方发布），存放在 `models/weights/`。

macOS（Apple Silicon）无需额外配置，自动使用 MPS GPU 加速；NVIDIA 显卡请按 [PyTorch 官网](https://pytorch.org/get-started/locally/) 安装对应 CUDA 版本的 torch。



## 使用

1. 导入图片 / PDF / 文件夹（或直接拖入窗口）
2. 点击「自动上色」
3. 对不满意的区域涂抹颜色提示后「重新生成」
4. 导出单页或全部页面

## 致谢与许可

本项目基于以下开源工作构建：

- [qweasdd/manga-colorization-v2](https://github.com/qweasdd/manga-colorization-v2) — 上色核心模型
- [vikast908/ColorComic](https://github.com/vikast908/ColorComic)（MIT）— 提示点 API、引导式自动提示、分格/分块推理逻辑
- [xiaogdgenuine/Manga-Colorization-FJ](https://github.com/xiaogdgenuine/Manga-Colorization-FJ) — 跳过已上色页面、权重格式兼容等思路




## 免责声明

本工具仅供个人学习与研究使用。请勿将上色结果用于侵犯原作者版权的用途。
