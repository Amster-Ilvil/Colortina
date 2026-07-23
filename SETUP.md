# Colortina v5 安装与验证

## 1. 建立虚拟环境

### macOS

```bash
cd /你的/Colortina-optimized-v5
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows

```powershell
cd C:\你的\Colortina-optimized-v5
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. 启动

```bash
python main.py
```

Windows 也可以运行 `Start_Colortina.bat`。

首次自动上色会下载 mc-v2 权重；首次启用区域语义、参考画风或角色参考提取时会下载 CLIP 权重。权重下载完成后可离线运行。

## 3. Apple Silicon 建议

- 默认 `COLORTINA_DEVICE=auto`，选择顺序为 CUDA → MPS → CPU。
- MPS 不支持的少量算子会通过 `PYTORCH_ENABLE_MPS_FALLBACK=1` 自动回退 CPU。
- 16GB 机器建议关闭浏览器大型标签页、视频软件和其他模型进程。
- 普通使用优先 Draft/Normal；高质量 tiled 会额外运行一次整页低分辨率颜色先验，因此更慢但跨块颜色更稳定。

强制设备：

```bash
COLORTINA_DEVICE=mps python main.py
COLORTINA_DEVICE=cpu python main.py
```

## 4. 首次功能检查

按以下顺序验证：

1. 导入一张黑白漫画页并运行普通自动上色。
2. 打开提示/人物覆盖层，确认自动提示很少且人物框有绿色、黄色或红色状态。
3. 普通彩页可先使用“自动提取身份配色”；复杂封面、倒置或遮挡人物使用“手动添加参考角色”。
4. 为同一人物从多张参考图使用相同名称重复录入，再次自动上色并检查发色、肤色和瞳色是否区分。
5. 对黄色歧义人物使用“当前页人物绑定与禁用锁色”。
6. 从另一张彩页提取画风，确认画风改变但现有角色库没有被覆盖。
7. 保存项目，关闭并重新打开，确认页面绑定、角色库、场景色和诊断均恢复。

## 5. 命令行烟雾测试

```bash
python test_pipeline.py path/to/page.png
```

可选参数以脚本帮助为准：

```bash
python test_pipeline.py --help
```

## 6. 自动测试

```bash
python -m compileall -q .
python -m unittest discover -s tests -v
```

## 7. 真实页面回归评估

Colortina 不附带受版权保护的漫画测试页。将你有权使用的页面放入 `tests/golden/`，参照 `tests/golden/manifest.example.json` 建立清单：

```bash
python tools/evaluate_golden.py tests/golden/manifest.json \
  --output tests/golden/report.json
```

## 8. 常见问题

### 出现彩色小圆点或色块

v5 会自动检测并降级重跑一次。若仍出现：

- 查看页面 tooltip 中的 `hint_blob_score` 和 `hint_retry`。
- 暂时降低角色身份色强度。
- 检查是否有过大的手动画笔提示；手动画笔由用户控制，不会在自动降级中被删除。
- 打开提示覆盖层确认提示是否落在错误连通区域。

### 所有人物又变成接近同色

- 确认角色色是通过“角色身份配色”提取，而不是只提取了画风。
- 不要把场景彩图当角色参考图。
- 打开人物覆盖层；黄色歧义人物默认不锁色，可人工绑定。
- 旧 v3 `.ccstyle` 的主色只用于预览，不应继续作为角色配色来源。

### 瞳色没有生效

- 眼睛必须在参考图中足够清晰且有可见虹膜颜色。
- 小眼睛区域会使用更严格安全门，置信度不足时宁可不锁色。
- 可使用参考图—目标页对应点进行精确修正。

### CLIP 无法下载或加载

无 CLIP 时仍可运行普通 mc-v2，但角色匹配、语义场景色和参考区域分析不可用。检查网络、磁盘空间及 `transformers` 安装。

### MPS 报错或内存不足

- 关闭其他高内存程序。
- 使用普通质量而非 tiled/Ultra。
- 更新 PyTorch。
- 临时使用 `COLORTINA_DEVICE=cpu` 验证是否属于 MPS 算子问题。

## 9. 环境限制说明

源码可在无模型权重环境下完成语法和单元测试；真正的桌面视觉效果、MPS 推理速度和真实漫画质量仍应在安装完整 requirements 和权重的目标 Mac/Windows 机器上验收。


## 10. 动漫脸检测器

首次自动提取角色身份或在无 CLIP 情况下分析人物时，程序会下载并缓存小型 `lbpcascade_animeface.xml`。

- 默认缓存：`~/.cache/colortina/lbpcascade_animeface.xml`
- 完全离线：设置 `COLORTINA_OFFLINE=1`，并改用“手动添加参考角色”。
- 不再默认启用普通 Haar 正脸分类器，因为它在漫画封面上误报率过高。

## 11. 本轮样本回归

```bash
python tools/reference_sample_regression.py manifest.json --output sample_regression
```

该工具会先检查 `models/weights/`。只有 generator、extractor 和 denoiser 都存在时，才允许将运行描述为完整模型推理；否则输出明确标记为分析模式。

## 自动上色没有启动时

v5.1 会在右侧自动上色区域显示当前阶段。若首次下载失败：

1. 确认已经执行 `pip install -r requirements.txt`，尤其是 `gdown`。
2. 确认网络可以访问 Google Drive。
3. 删除 `models/weights` 中大小为 0 或明显不完整的文件后重试；程序通常会自动清理失败残留。
4. 错误弹窗会显示具体异常，终端中保留完整 traceback。

CLIP 不再是普通自动上色的硬依赖。没有本地 CLIP 缓存时，程序直接使用 mc-v2；不会因为下载 CLIP而长时间无界面反馈。
