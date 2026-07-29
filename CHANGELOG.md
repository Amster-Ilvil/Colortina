## V5.4.3 — 独立自定义颜色倾向画笔

- 新增“颜色倾向画笔”，使用 `core/bias_brush.py` 独立通道，只在实际笔迹范围内调用感知颜色倾向算法。
- 新画笔拥有独立颜色、笔刷大小、0–200% 强度、明暗范围以及肤色/线稿/高饱和保护开关。
- 不调用 mc-v2，不写入或修改 Hint，不复用普通画笔 `apply_brush_edit`，也不读取/修改整页自定义颜色倾向的开关和参数。
- 每条笔画只生成一次倾向候选图，拖动过程中通过累计柔边 Mask 实时合成；笔迹外像素逐值保持不变。
- 普通画笔、模型 Hint 画笔、区块/套索/矩形上色、局部 AI 重上色和整页颜色倾向路径保持原样。

## V5.4.3 — mixed Hint 补边尺寸修复

- 修复非 32 倍数页面在 mc-v2 缩放后产生内容区与补边区尺寸差异时，整区 Hint 布尔索引崩溃。
- 整区 Hint 仅写入未补边的有效内容视图，兼容 1009x712 页面对应的 817→832 高度补边。

# Changelog

## V5.4.4 — 修复局部 Hint 无论选什么都偏黄

- 修复局部 AI 重上色仍把手动 Hint 转成 `manual_region` 并拆成稀疏明暗点的问题。标准局部路线现在保留用户的原始 `manual` RGB，并使用 `mixed` Hint 渲染，把精确颜色写满 Hint 所在的闭合线稿区域。
- 模型 Hint 画笔不再经过“匹配当前风格”二次改色；色板选中的 RGB 会原样进入 mc-v2。
- 新增生成后手动颜色锁：仅在本次新生成图中，对 Hint 命中的闭合区锁定用户色相与饱和度，同时保留 mc-v2 生成的明暗、纹理和线稿。即使模型忽略 Hint 并固定输出黄色，目标区域也会恢复为指定颜色。
- 继续保留 V5.4.1 修复：选区内手动 Hint 压制旧 auto hint；成功后自动消费本次选区内手动/吸管 Hint。
- 用户样图测试使用“无论 Hint 都返回黄色”的替身模型，红、蓝、绿、紫四种 Hint 均得到对应不同色相。
- 全量回归：274 passed，6 skipped。

## V5.4.3 — 上色前 Hint 直达模型：一个点 = 一整块区域

- 画笔圆点直涂保留为画笔专属通道（`manual_paint` 来源）：只属于画笔、
  永不进入模型；模型 Hint 与画笔直涂两条路径完全独立。

- 修复根本问题：手动画笔 Hint 此前被刻意排除在 mc-v2 输入之外，只作为
  上色后的圆点补丁贴回（这就是"打了 hint 只剩圆点、和事后涂抹一样"的原因）。
  现在手动/吸管 Hint 全部进入模型 hint 通道。
- 新增 mixed 混合渲染（整页生成默认）：手动点自动填满其所在的封闭线稿
  区域（满强度、二值），模型按自己的明暗渲染整块颜色；自动提示保持柔和。
  点落在线条上时就近吸附区域；区域超过画布 35% 时回退为硬点，防止整页被淹。
- 实测：黑白页上脸部/衣服各点一下 → 整块区域精确呈现指定颜色，未打点的
  背景不受影响。

## V5.4.2 — 吸收三项外部能力（全部本地推理，遵守网络锁定）

- **文字气泡保护**（comic-text-detector，ONNX 本地推理）：自动上色后把检测到的
  文字像素还原为原稿黑白，对话框不再被染色。上色页新增开关，默认开启；
  模型缺失时自动跳过。
- **真实角色分割**（SkyTNT anime-segmentation，ONNX 本地推理）：自定义颜色倾向的
  「影响范围 = 角色 / 背景」改用真分割模型，替代旧的边缘密度启发式；
  分割覆盖率退化时自动回退启发式。
- **Real-ESRGAN anime6B 接通**：Ultra 质量的 4x 超分不再依赖已损坏的
  realesrgan/basicsr 包，改用自带最小 RRDBNet（vendor/realesrgan_min）直载权重。
- 新依赖：onnxruntime。新增权重：comictextdetector.pt.onnx (90MB)、
  isnetis.onnx (168MB)、realesrgan_anime6b.pth (17MB)，均已放入 models/weights。

## V5.4.0 — 移除 MC v1 旧版引擎，回归单一 mc-v2 路线

- 以 V5.2.0（编辑工具单一路径）为主体，保留 V5.3.x 中与旧版引擎无关的有益改进。
- 删除 MC v1（旧版 colorizer.zip）风格、独立 TorchScript 引擎、CPU 隔离逻辑与模型文件。
- 删除 “AI 选区重上色（原版 Hint）” 独立引擎按钮；普通 **AI 选区重上色**（mc-v2）与
  “原版单点 Hint 传播（独立模式）”开关全部保留。
- 保留：选区聚焦推理（框外弱化）、局部 Hint 筛选与缓冲带、选区黑白预览、
  模型构建加锁（预加载与上色线程不再重复加载）。
- 保留 macOS App 依赖锁定文件 requirements-macos.lock.txt。
- 风格「MC v2 (原始 mc-v2)」更名为「MC v2」。
- 默认关闭：套索选区边界自动贴线、选区恢复黑白预览（仅填充选区内闭合区域本就默认关闭）。
- 重修自定义颜色倾向：
  - 修复多重门控连乘导致的强度塌缩（默认 35% 实际只生效约 15%）；
  - 可上色判定改为资格饱和门，淡色皮肤/浅背景不再永久半强度；
  - 修复 100% 以上强度会把纸面白染色的问题（低 alpha 硬归零）；
  - 保护项移到归一化之后，均匀画面上不再被静默抵消；
  - 降低材质保留强度，特色区域（脸部等）也能真实响应目标色。

## V5.3.6 — MC v1 风格集成 + 旧版引擎修正

- 在风格列表新增 **MC v1（旧版 colorizer.zip）**。
- 选择 MC v1 后，整页 **自动上色 / 重新生成** 会独立走旧版 TorchScript 引擎。
- 修正旧版引擎局部重上色时的聚焦输入问题：MC v1 不再使用框外弱化图，只看原始整页黑白页，最后只做局部合成。
- 强化对 `colorizer.zip` 返回值的解析，优先提取真正的 3 通道 RGB 输出，避免把辅助张量错误贴回选区。
- 选区界面的 **AI 选区重上色（原版 Hint）** 按钮现在会随选区启用/禁用。

## V5.3.5 — 原版 TorchScript 引擎 CPU 隔离修复

- 原版 `colorizer.zip` 引擎固定使用 CPU，避免 Apple MPS 的 `Placeholder storage has not been allocated` 运行时错误。
- 标准 mc-v2 仍可继续使用 MPS/CUDA；CPU 限制只作用于“AI 选区重上色（原版 Hint）”。
- 原版模型加载后执行 `eval()`，可用时再执行 `torch.jit.freeze()`；推理前再次确认模型输入位于 CPU。
- 设备切换不会再把原版 TorchScript 引擎移动回 MPS。

## V5.3.4 — 原版 TorchScript 引擎加载与 6 通道输入修复

- 修复原项目 `colorizer.zip` 被错误当作 state_dict 加载的问题；原版按钮现在直接使用 `torch.jit.load`。
- 原版交互式引擎改为独立 6 通道输入：BW + DFM + RGB Hint × Mask + Mask。
- 新增原版 512 短边缩放与独立 Hint 坐标几何，避免提示点位置错位。
- 新增 OpenCV 实现的 XDoG 线稿与签名距离场 DFM，不新增 Snowy 运行依赖。
- 标准 mc-v2 缓存、画笔、选区及自动上色路径保持不变。

## V5.3.2 — 原版 Hint 独立按钮 + 双引擎入口

- 新增独立按钮 **AI 选区重上色（原版 Hint）**。
- 新增双引擎局部重上色入口：保留现有 mc-v2 路线，同时支持加载原项目 `colorizer.zip` 的交互式引擎。
- 原版按钮会强制使用原版单点 Hint 传播逻辑与 plain pipeline，不影响普通 mc-v2 路线。
- 若未提供 `models/weights/colorizer.zip`，会给出明确提示，不会影响其它正常功能。

## V5.3.1 — 选区外弱化的局部聚焦 AI 重上色

- 新增“框外弱化后再 AI 重上色”：运行 AI 选区重上色时，可将蓝色选区外的黑白图逐步弱化到白色，只保留一圈上下文带，让 mc-v2 更聚焦蓝色区域内部。
- 新增“上下文保留 / 外层弱化”参数，分别控制完整保留的上下文宽度与向白色渐变的距离。
- 默认开启局部聚焦模式；关闭后仍可回到原来的整页黑白输入模式。
- 保持原有原则：最终仍只把蓝色选区内的结果合成回当前彩图，选区外像素严格不变。

# Changelog

## V5.3.6 — MC v1 风格集成 + 旧版引擎修正

- 在风格列表新增 **MC v1（旧版 colorizer.zip）**。
- 选择 MC v1 后，整页 **自动上色 / 重新生成** 会独立走旧版 TorchScript 引擎。
- 修正旧版引擎局部重上色时的聚焦输入问题：MC v1 不再使用框外弱化图，只看原始整页黑白页，最后只做局部合成。
- 强化对 `colorizer.zip` 返回值的解析，优先提取真正的 3 通道 RGB 输出，避免把辅助张量错误贴回选区。
- 选区界面的 **AI 选区重上色（原版 Hint）** 按钮现在会随选区启用/禁用。

## V5.3.5 — 原版 TorchScript 引擎 CPU 隔离修复

- 原版 `colorizer.zip` 引擎固定使用 CPU，避免 Apple MPS 的 `Placeholder storage has not been allocated` 运行时错误。
- 标准 mc-v2 仍可继续使用 MPS/CUDA；CPU 限制只作用于“AI 选区重上色（原版 Hint）”。
- 原版模型加载后执行 `eval()`，可用时再执行 `torch.jit.freeze()`；推理前再次确认模型输入位于 CPU。
- 设备切换不会再把原版 TorchScript 引擎移动回 MPS。

## V5.3.4 — 原版 TorchScript 引擎加载与 6 通道输入修复

- 修复原项目 `colorizer.zip` 被错误当作 state_dict 加载的问题；原版按钮现在直接使用 `torch.jit.load`。
- 原版交互式引擎改为独立 6 通道输入：BW + DFM + RGB Hint × Mask + Mask。
- 新增原版 512 短边缩放与独立 Hint 坐标几何，避免提示点位置错位。
- 新增 OpenCV 实现的 XDoG 线稿与签名距离场 DFM，不新增 Snowy 运行依赖。
- 标准 mc-v2 缓存、画笔、选区及自动上色路径保持不变。

## V5.3.2 — 原版 Hint 独立按钮 + 双引擎入口

- 新增独立按钮 **AI 选区重上色（原版 Hint）**。
- 新增双引擎局部重上色入口：保留现有 mc-v2 路线，同时支持加载原项目 `colorizer.zip` 的交互式引擎。
- 原版按钮会强制使用原版单点 Hint 传播逻辑与 plain pipeline，不影响普通 mc-v2 路线。
- 若未提供 `models/weights/colorizer.zip`，会给出明确提示，不会影响其它正常功能。

## V5.3.0 — 原始黑白整页推理与 AI 选区重上色

- 新增“AI 选区重上色”：始终将原始黑白整页送入 mc-v2，不再裁剪局部，也不把半彩图半黑白图送入模型。
- 新增严格局部合成：mc-v2 生成完整新彩图后，只在蓝色选区内以内向羽化合成；选区外像素逐值保持不变。
- 当前编辑层、最近 AI 结果层和滤镜基础层同步更新，撤销、恢复 AI 结果及后续滤镜不会丢失局部模型修色。
- 新增非破坏“选区恢复黑白预览”，只改变画布显示，不覆盖当前彩图像素。
- 新增局部 Hint 筛选与缓冲带：默认只允许选区及附近提示参与本次推理，降低远处旧提示、错误肤色提示继续污染结果的概率。
- 新增“模型 Hint 画笔”：可在保留蓝色选区时切换工具，写入 mc-v2 手动提示而不直接涂改彩图；普通手动提示在局部推理时会转换为模型可读取的 `manual_region`。
- 新增“清除选区内手动/吸管 hint”，支持先删除错误绿色、紫色或灰色提示，再写入正确颜色。
- 工具切换不再自动取消蓝色选区，可按“圈定 → 写入/清除 Hint → AI 选区重上色”的完整流程操作。
- 当模型输出与当前选区完全相同时会明确提示“没有变化”，并引导新增、移动、更换或清除 Hint，不再表现为无响应。
- 新增完整核心回归：原图输入验证、Hint 过滤/转换、整页 pipeline 路由、选区外零改动、内向羽化、黑白预览非破坏。

## V5.2.0 — 编辑工具单一路径与动态参数面板

- 区域画笔收敛为单一路径：按真实笔迹实时上色，鼠标松开只提交撤销历史，不再贴线、扩区或执行二次模型处理。
- 从主界面、设置保存和编辑事件链移除画笔松开贴线、瞳色自然融合和局部模型上色；同时删除画笔贴线与局部模型重绘的废弃模块和工作线程。
- 编辑页根据当前工具动态切换标题、说明和参数；区块封口长度仅在区块、套索和矩形工具下显示。
- 新增“恢复当前工具默认参数”，只重置当前工具相关设置，不影响其他工具。
- 套索/矩形的手动扩大与擦除控件只在已经建立蓝色选区且开启手动修复时显示。
- 删除“显示提示点和区域边界”界面选项；提示点预览保留，但区域边界永远不在主界面显示。
- 清理隐藏选区下拉框、局部重绘工作线程和重复忙碌锁，降低主窗口事件链复杂度。

## V5.0.0 — 区域画笔可靠上色与区块上色命名

- 区域画笔开启“松开后贴线”时，若线稿吸附结果为空或仅剩极少像素，会自动回退到受线稿保护的原始笔迹，不再出现画完完全不上色。
- 回退只保留画笔实际经过的区域，并保护实心黑色线稿；不会扩大成整块或整页。
- “线条区块上色”统一更名为“区块上色”，功能仍为点击选择一个闭合区块并立即调色。
- 应用版本标识和更新包名称升级为 V5。

## V3.3.7 — Selection Line Snap

- Added post-release lasso/rectangle snapping to nearby manga line-art regions.
- Added configurable 1–40 px snap distance with RegionMap leakage guards.
- Integrated snapping with closed-area AI previews and manual blue-mask correction.
- Manual add/erase corrections now become authoritative and cannot be overwritten on Apply.
- 244 tests passed, 2 skipped.


## V3.3.2 — 狭长闭合区过滤与稳定性增强

- 新增“狭长区过滤”滑块和数值框（0–100 px），实时剔除面积较大但有效宽度很小的长条、弯曲细缝和线状误选区域。
- 过滤不再只看水平/垂直外接框：使用旋转外接框、面积/主轴宽度、轮廓水力宽度、长宽比、紧致度和填充率联合判断。
- 紧凑的小眼睛、纽扣、圆点等不会仅因尺寸小而被狭长过滤删除。
- 狭长过滤在外扩前执行，避免误选细条被外扩放大；滑块变化继续复用缓存模型结果并实时刷新蓝色预览。
- 新参数支持工程保存/载入、界面重建与中英文切换。
- 修复面积过滤设为 0 时仍被内部强制为 1 的语义不一致，并补充异常几何与零周长保护。
- 全量自动测试：229 项通过，2 项跳过。

### V3.3.1 矩形闭合区过滤与小区域外扩修复

- 新增“小闭合区过滤”滑块与数值框，可按面积阈值过滤矩形内误选的碎点、网点空隙和杂散闭合区。
- 过滤阈值与蓝色闭合区预览实时联动，不会重复运行 MangaLineExtraction；设置会随项目保存、载入及语言界面重建保留。
- 修复外扩算法可能删除/冻结原始小闭合区种子的问题，外扩现在严格单调递增，调大数值不会反而缩小。
- 收紧外扩停止边界，只把真实深色墨线核心视为硬墙，不再让灰色抗锯齿、阴影或网点提前阻断外扩。
- 全量自动测试：225 项通过，2 项跳过。

### V3 闭合区域实时外扩完善

- 闭合区外扩滑块改为实时刷新蓝色预览，无需重新画矩形。
- 缓存 MangaLineExtraction 概率图，拖动滑块不会重复运行模型。
- 外扩改为朝原始黑白图的真实墨线推进，数值变化更加连续有效。
- 支持替换、加选、减选时保留原选区并重新计算当前矩形。
- 点击应用前强制同步最新滑块数值，避免数值与实际 mask 不一致。


## V3 MangaLineExtraction 漫画结构线 AI 接入

- 仅接入官方 `ljsabc/MangaLineExtraction_PyTorch`，没有加入 Anime2Sketch、ControlNet 或其他线稿模型。
- 将官方 MIT 网络结构以独立 vendor 模块接入，兼容原始 `erika.pth` state dict；许可证随包保留。
- 矩形勾选“仅填充选区内闭合区域”后，在后台线程对原始黑白图的矩形局部运行模型，首次使用自动下载约 170 MB 权重。
- AI 结构线概率成为闭合拓扑的主要依据；OpenCV 只保留极深墨线与短 Gap 修补，避免重新把网点、阴影和颜色纹理当成边界。
- 模型下载、加载或推理失败时直接取消闭合填充，不再静默回退到普通整框矩形。
- 支持 CUDA、Apple MPS 与 CPU；加速器推理异常时自动回退 CPU。
- 新增模型网络键兼容、输出方向、AI 主导结构线、矩形闭合区域和异步 UI 调用链测试。

## V3 矩形闭合主体区域修复（蓝色碎点问题）

- 修复矩形“仅填充选区内闭合区域”把真正的衣服、领带、头发等最大闭合主体误删，只留下装饰纹/缝隙小碎块的问题。
- 矩形闭合路径不再启用 `reject_dominant`；只排除与矩形边缘连通的开放区域，保留矩形内部所有真实线稿闭合区域。
- 结构线检测移除 `gray <= 185` 的过宽中灰障碍规则，避免网点、阴影和已有颜色纹理被误识别成漫画线条。
- 新增 Canny 淡线轮廓通道，并降低全局膨胀强度；浅灰抗锯齿轮廓仍可闭合，同时不会把主体区域切成大量小岛。
- 加强圆形网点过滤，真实长线/闭合轮廓保留，孤立印刷点不再成为填充边界。
- 新增紧贴衣服主体矩形、网点纹理和最大有效闭合区域回归测试。

## V3 矩形闭合区域结构线识别修复

- 新增独立 `core/structural_line_detector.py`，融合多尺度暗线、局部对比、Scharr 边缘、自适应阈值和短 Gap 修补，替代矩形闭合过滤中的固定灰度阈值。
- 矩形勾选“仅填充选区内闭合区域”后，会在鼠标松开时立即计算并预览最终闭合掩膜；确认上色只使用该掩膜，不再重新传入原始矩形。
- 排除与选区边界连通的开放区域，并拒绝占据选区主体的面板/背景大腔体，避免“看似闭合、实际整框上色”。
- 增加可选外部线稿概率图接口，后续可接 MangaLineExtraction、Anime2Sketch 或 ControlNet Aux，而无需改动矩形上色流程。
- 新增淡灰抗锯齿线、面板背景排除、权威掩膜不回扩和 UI 调用链测试。

## V3 瞳色边界与画笔独立性修复

- 瞳色自然融合新增独立线稿边界守卫，只在光标所在的单一瞳孔/虹膜闭合区域内生效，不再越过瞳孔或虹膜线条。
- “画笔自动吸附附近线稿”和“瞳色自然融合”默认均改为关闭；项目中已有明确设置仍可正常保存和恢复。
- 修复关闭画笔吸附后自由区域画笔访问空 RegionMap 导致失效的问题，两项增强功能不再是普通画笔的依赖。
- 矩形/套索“仅填充选区内闭合区域”改用抗锯齿线稿障碍判断，灰色细线也能阻止越界。

## V3 颜色倾向增强 + 自然颜色滤镜

- 新增独立 `core/natural_tint.py` 感知调色模块，不改动 mc-v2、提示点、区域图或撤销流程。
- 自定义颜色倾向由弱 HLS 推色改为 LAB 低频色场迁移，保留局部色彩纹理和明暗；强度范围扩展到 0–200%。
- 颜色滤镜改为保明暗、保纹理的自然调色，避免透明色层式统一染色；强度范围扩展到 0–150%。
- 画笔松开后线稿吸附与矩形“仅封闭区域”沿用独立模块，不与新的全局调色代码耦合。
- 新增强度递进、线稿保护、纹理保留和 UI 范围回归测试。

## V3 自然色相恢复 + 区域大小控制 + 瞳色自然融合

- 新增“闭合区外扩”滑块：当 MangaLineExtraction 识别区域略偏内时，可在不跨越线条的前提下把闭合 mask 向外均匀扩展。
- 自然色相迁移退回稳定的 LAB/HLS 后处理路径，不再依赖局部 mc-v2 重绘或复杂色场重定向。
- 优化 Gap 与区域分割，加入边界安全带和 35% 页面面积上限，点击页外背景或疑似整页连通区时直接拒绝填充。
- 新增“瞳色自然融合”开关（该版本当时默认开启；最新修复已改为默认关闭）；画笔会保留瞳孔、睫毛和高光，只让虹膜中间调自然吸收所选颜色。
- 普通画笔与松开后线稿吸附画笔共用同一瞳色融合逻辑，设置会随项目保存和载入。

## V2.8.11 画笔松开后线稿吸附与单页 UI 恢复

- 画笔吸附从“每个笔点即时裁剪”改为“整笔松开后统一吸附”，绘制阶段只记录轻量原始笔迹，降低拖动延迟与断续。
- 新增闭合笔迹识别：以笔迹形成的封闭空洞为准保留圈内区域，伸到圈外又折回的多余线段不会进入最终掩膜。
- 闭合圈会映射到漫画线稿的连通区域并扩展到附近墨线边界，同时严格排除线稿像素与圈外远端区域。
- 非闭合笔迹保持局部，只在笔迹附近吸附线稿，不会整块泛洪到整个连通区域。
- 恢复上一版单页编辑界面，取消“画笔与颜色 / 选区 / 操作与历史”嵌套子标签，减少切换步骤。
- 编辑主区域改为占据剩余垂直空间并延伸到中下部；历史按钮保持底部固定高度，避免顶部拥挤和中底部空白。
- 修正选区模式与选区羽化的网格行冲突；五种工具保留在同一页面并以两行排列。
- “画笔自动吸附附近线稿”固定显示在笔刷大小下方（该版本当时默认开启；最新修复已改为默认关闭），项目保存与载入继续保留该状态。
- 全量自动测试 184 项通过，2 项因当前环境缺少 PySide6/真实模型运行条件而跳过。

## V2.8.11 编辑页布局修复

- 编辑页重新拆分为“画笔与颜色 / 选区 / 操作与历史”三个子标签，修复顶部拥挤、底部空白。
- 五种编辑工具改为两行网格排列，避免窄窗口文字挤压。
- “画笔自动吸附附近线稿”固定显示在笔刷大小下方。
- 修复选区模式、局部上色模式和选区羽化共用网格行造成的控件冲突。
- 工具切换时自动展示相应设置页。
- 保持所有编辑、上色、撤销和项目保存功能不变。

## V2.8.11

- 修复选色窗口在 macOS 等环境打开后无法可靠关闭：改为主窗口持有的非原生模态 `QColorDialog`，取消、确定和标题栏关闭均有明确生命周期。
- 修复三种局部上色模式本质都只是 OpenCV 色层覆盖的问题。
- “自然色相迁移”改为局部 mc-v2 引导重绘：目标色提示 + 周围颜色上下文提示 + 自适应邻色收敛，强调自然融入。
- “统一色相”改为局部 mc-v2 强提示重绘，再统一目标色相与饱和度，保留模型生成的材质、褶皱、阴影和高光。
- “完全统一纯色”明确跳过模型并直接写入精确 RGB，与前两种模型模式形成真正不同的效果和运行路径。
- 画笔改为整条笔画合并掩膜后只运行一次局部模型推理；区域、套索和矩形选区共用安全掩膜与局部模型任务。
- 增加局部模型异步线程、冲突编辑锁定、失败回滚、撤销记录保护、线稿保护和严格掩膜外不变保证。
- 保留 V2.8.11 原有工程持久化、界面状态和旧 LAB/HLS 后处理 API，避免破坏兼容测试及其他功能。
- 全量测试 175 项通过，2 项因当前环境无 PySide6/模型权重而跳过。

## V2.8.10

- 修复替换 / 加选 / 减选三种选区合成模式实际未独立工作的运行时问题。
- 三种模式改为互斥单选，绘制时使用不同预览色，并保留累计选区显示。
- 画圈与矩形选区统一使用同一套选区代数。
- 选区上色开始遵循区域上色模式；“统一色相，保留明暗”可统一同套衣服的断开区域。
- 纯色模式在选区内部严格匹配用户选择的 RGB。
- 全量测试 166 项通过，2 项跳过。

## V2.8.7
- Fixed custom color bias ignoring most mc-v2-colourized manga interiors whose original source pixels were white.
- Reworked custom bias paper detection to protect genuinely blank neutral paper while allowing pale skin, clothing and backgrounds to receive a visible colour tendency.
- Fixed lasso and rectangle fills shifting saturated reds/blues/violets toward the wrong hue after LAB gamut clipping.
- Manual lasso/rectangle recolor now follows the selected hue directly while preserving the existing lightness/shading field and never painting outside the selection.
- Style-based manual colour adaptation is now opt-in and explicitly warns that it changes the selected colour.
- Added colour-intent, source-white bias, paper-protection, lightness-retention and locality regression tests.

## V2.8.6
- Added inward-only selection feathering (0-30 px).
- Added skin-tone, line-art, and high-saturation protection for custom color bias.
- Fixed custom color bias not being applied by the actual colorize pipeline.
- Fixed BatchColorizeWorker missing custom color bias state and removed duplicate single-worker assignment.
- Improved combined selection preview to display subtraction holes.

## V2.8.5
- Added selection combine mode: replace / add / subtract.
- Added custom color bias tone range: all / highlights / midtones / shadows.
- Added mask-overlay preview for combined pending selections.

## V2.8.4
- Added a closed-region-only option for lasso/rectangle selection fill.
- Added custom color bias scope: whole page / characters first / background first.
- Added tests for closed-region selection masks and bias scope behavior.

## V2.8.2
- Added lasso fill and rectangle fill selection tools in Edit.
- Added custom global color bias in the Auto Colorize panel.
- Added tests for selection masks and global color bias behaviour.

V2.8.1

- 发布前清理：删除 Python 缓存、用户样本验证图片和过期版本文档。
- 更新 README、安装说明和 `.gitignore`，与当前 V2.8 界面及工作流保持一致。
- 修复中英文翻译字典中的重复键和缺失键。
- 当前颜色区域补齐正常悬浮提示。
- 将效果验证脚本改为通用命令行工具，不再依赖固定 `/mnt/data` 路径。
- 增加翻译完整性和发布包卫生检查。

## V2.8

- 编辑界面显示当前选择色与实际风格匹配补色。
- 吸管的点采集和区域采集改为直接可见的互斥勾选项。
- 保留 V2.7 的画笔即时刷新、区域上色安全回退和滤镜基础层同步修复。

## v2.8.11 reference brush/gap/no-blocks fix
- Restored reference-package brush radius mapping and dab spacing; open-stroke snapping can no longer enlarge brush width.
- Fixed false closed-loop detection that made small brush strokes fill oversized areas.
- Restored reference-style tone/chroma refinement after line-gap region segmentation.
- Larger line-gap values now close qualifying breaks and isolate the intended smaller line block.
- Manual natural/uniform/flat recolor now runs at full page resolution on the current AI result, eliminating local mc-v2 crop/grid blocks and latency.


## V5.4.2 — 局部重上色与自定义颜色倾向隔离

- 局部 AI 重上色现在在 `core/local_model_recolor.py` 内独立处理“手动 Hint 压制旧 auto hint”“黄色先验修正”“成功后消费手动 Hint”。
- 自定义颜色倾向不再被这些局部修复链路覆盖：局部重上色时会在手动颜色锁定之后重新单独应用颜色倾向。
- 普通整页上色 / 重新生成的 `pipeline.py` 路径保持不变，避免局部修复影响其它功能。


## V5.4.3 — 颜色倾向画笔与自定义颜色同步

- 颜色倾向画笔与“自定义颜色倾向”现在共享同一颜色。修改任一方的颜色，另一方会同步。
- 同步保持单向兼容：吸管 / 当前选中色更新时，允许同步到自定义颜色；但手动修改“自定义颜色倾向”颜色时，不会反向改写普通画笔 / 吸管当前色。
- 自定义颜色倾向计算现在读取 `_custom_color_bias_color` 本身，不再间接依赖普通画笔颜色。


## V5.4.6 — 颜色倾向画笔最终效果恢复

- 拖动期间继续使用 ROI 增量预览，保留流畅度和连续插值。
- 鼠标松开后，使用上一版完整笔迹合成算法重新计算一次最终结果，恢复原有颜色强度、软边缘和重叠区效果。
- 最终结果与优化前的多 dab 全页合成逐像素一致；性能优化仅影响拖动预览，不再改变成品。


## V5.4.8 — 颜色倾向画笔整块填充优化

- 颜色倾向画笔在鼠标释放后的最终合成阶段，新增“cohesive fill”处理。
- 对明确闭合的区域，会把笔迹扩展到完整的精修线稿区域，而不是只在局部留下碎片色块。
- 对纹理复杂、线条未完全闭合的区域，会在局部范围内执行受线稿约束的保守 flood+close 扩展，尽量把颜色连成整块，同时避免跨越大块墨线泄漏到整页。
- 拖动画时仍保留快速预览；最终结果再用整块填充后的 alpha 重新完整合成，因此不会牺牲连画流畅度。


## V5.4.9 — 复杂纹理整块填充增强

- 颜色倾向画笔在抬笔后的最终整合阶段，新增“outer contour fill”候选：先在局部简化线稿，只保留较强外轮廓，再从笔迹种子向内部做连通填充。
- 对复杂纹理 / 筛网点 / 未完全闭合线条，不再只依赖精确闭合区，因此不会轻易退化成零碎色块。
- 新增基于 seed-tone 的限制：即使采用外轮廓整块填充，也只保留与笔迹附近明暗接近的局部区域，减少扩散到无关背景。
- 候选区域上限改为“局部窗口上限 + 页面上限”联合约束，允许中等大小的整块区域被接受，同时继续阻止整页泄漏。
- 已用用户提供的黑白测试图做本地回归检查，局部笔刷最终 alpha 从狭窄条带扩展为连贯大块区域。


## V5.5.0 — 颜色倾向画笔整块填充档位

- 为颜色倾向画笔新增“整块填充”档位：保守 / 标准 / 强整块。
- 保守：更偏向局部和精细，尽量少扩张。
- 标准：默认模式，在整块填充与边界谨慎之间平衡。
- 强整块：更积极选择较大的连贯候选区域，适合复杂纹理、筛网点和未完全闭合线条。
- 整块填充档位只作用于颜色倾向画笔抬笔后的最终整合阶段，不影响普通画笔、Hint、mc-v2 和整页自定义颜色倾向。


## V5.5.1 — 整块档位与笔刷范围实质分离

- 修复“保守 / 标准 / 强整块实际范围相同”：三个档位现在分别选择最小、中位、最大合理候选，并使用不同的笔刷外扩倍率。
- 最终候选区域统一通过“距真实笔迹的硬范围包络”裁切，哪怕检测到很大的闭合区，也不能越过当前笔刷大小允许的范围。
- 移除固定 24 px 最小扩张造成的小笔刷大范围影响；2–3 px 小笔刷现在只影响紧邻笔迹的区域。
- 改动仍只属于颜色倾向画笔最终整合，不影响普通画笔、Hint、mc-v2 或整页颜色倾向。


## V5.5.1 — 强整块恢复固定 24 px 最小扩张

- 按要求，仅“强整块”模式恢复固定 24 px 最小扩张，并走更接近旧版标准档位的候选上限逻辑。
- “保守”和“标准”继续保留新版的笔刷大小敏感范围控制。
- 这样小笔刷下，保守/标准仍然收敛，而强整块会更积极地扩展到较大的连贯块。


## V5.5.2 — 颜色倾向专用消除笔

- 新增独立工具“颜色倾向消除笔”。
- 只对颜色倾向画笔效果生效，沿当前笔迹局部恢复，不做整块扩张。
- 新增颜色倾向基准引用层：非颜色倾向编辑会刷新基准，便于消除笔恢复到最近一次非颜色倾向状态。
- 已纳入 undo/redo 快照。


## V5.5.3 — mc-v2 偶发局部黑白修复

- 原因定位：文字气泡保护会在少数页误检，导致检测区域被重新合成为原始黑白稿，表现为“整页已上色，但局部区域又变回黑白”。
- 修复：文字气泡保护新增“极端明暗门控”，只允许恢复接近白纸与黑字的像素，不再把中灰漫画内容（头发、衣服、网点、阴影）恢复成黑白。
- 保留原有文字/气泡保护开关，不影响其它上色流程。


## V5.5.4 — 颜色倾向画笔回退到 complex-texture-fill 版本

- 以当前版本为主体，仅回退颜色倾向画笔逻辑到 `bias-brush-complex-texture-fill-fix` 的实现。
- 取消颜色倾向画笔中的“保守 / 标准 / 强整块”三档 UI 和对应逻辑。
- 其它功能保持当前版本不变，包括 mc-v2 局部黑白修复、颜色倾向消除笔及其引用层。


## V5.5.5 — 颜色倾向画笔/消除笔合并

- 将“颜色倾向画笔”和“颜色倾向消除笔”合并为一个工具，通过工具内模式切换按钮在“上色 / 消除”之间切换。
- 吸管工具调整为工具栏第二个位置。
- 仅调整颜色倾向工具的 UI 组织方式，不改动原有上色/消除逻辑，也不影响其它功能。
