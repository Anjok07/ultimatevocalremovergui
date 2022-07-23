# Ultimate Vocal Remover GUI v5.4.0
<img src="https://raw.githubusercontent.com/Anjok07/ultimatevocalremovergui/master/img/UVR_v54.png?raw=true" />

[![Release](https://img.shields.io/github/release/anjok07/ultimatevocalremovergui.svg)](https://github.com/anjok07/ultimatevocalremovergui/releases/latest)
[![Downloads](https://img.shields.io/github/downloads/anjok07/ultimatevocalremovergui/total.svg)](https://github.com/anjok07/ultimatevocalremovergui/releases)

[English](README.md) | 简体中文

## 关于

这个应用程序使用最先进的音源分离模型来去除音频文件中的人声。UVR 的核心开发人员训练了这个软件包中提供的所有模型（除了 Demucs 的辅助模型）。

- **核心开发者**
    - [Anjok07](https://github.com/anjok07)
    - [aufr33](https://github.com/aufr33)

## 安装

### Windows 安装

该安装包包含 UVR 接口、Python、PyTorch 和其他有效运行应用程序所需的依赖项。不需要任何先决条件，即装即用。

- 请注意：
    - 该安装程序适用于运行 Windows 10 或更高版本。
    - 不保证在 Windows 7 或更低版本时的应用功能
    - 不保证英特尔奔腾和赛扬 CPU 的应用功能。

- 通过以下链接下载UVR安装程序：
    - [主要下载链接](https://uvr.uvr.workers.dev/UVR_v5.4_setup.exe)
- 为已经安装了 UVR 的用户更新软件包的说明：
    - 如果从 UVR v5.3 更新 - [更新包](https://github.com/Anjok07/ultimatevocalremovergui/releases/download/v5.4.0/UVR_v5.4_Update_Package.exe)
    - 如果从 UVR v5.2 更新 - [更新包](https://github.com/Anjok07/ultimatevocalremovergui/releases/download/v5.4.0/UVR_v5.4_Update_From52_Package.exe)

- **可选**
    - 额外的模型和应用程序补丁可以通过应用程序内的 "设置" 菜单下载。

- **请注意：** 更多最新更新请见最新发布页面 [Releases](https://github.com/Anjok07/ultimatevocalremovergui/releases/tag/v5.4.0)

### 其他平台

这个应用程序可以通过执行手动安装在Mac和Linux上运行（更多信息请参见下面的**手动开发者安装**部分）。有些功能在非Windows平台上可能无法使用。

## 应用手册

**一般选项**

<img src="https://github.com/Anjok07/ultimatevocalremovergui/blob/master/img/gen_opt.png?raw=true" />

**VR 架构选项**

<img src="https://github.com/Anjok07/ultimatevocalremovergui/blob/master/img/vr_opt.png?raw=true" />

**MDX-Net 选项**

<img src="https://github.com/Anjok07/ultimatevocalremovergui/blob/master/img/mdx_opt.png?raw=true" />

**Demucs v3 选项**

<img src="https://github.com/Anjok07/ultimatevocalremovergui/blob/master/img/demucs_opt.png?raw=true" />

**合奏选项**

<img src="https://github.com/Anjok07/ultimatevocalremovergui/blob/master/img/ense_opt.png?raw=true" />

**用户合奏**

<img src="https://github.com/Anjok07/ultimatevocalremovergui/blob/master/img/user_ens_opt.png?raw=true" />

### 其他应用说明

- 建议使用至少有 8GB V-RAM 的 Nvidia GPU。
- 该应用程序只兼容 64 位平台。
- 该应用程序依赖于 Sox - Sound Exchange 的降噪。
- 该应用程序依靠 FFmpeg 来处理非 wav 音频文件。
- 关闭时，应用程序将自动记住您的设置。
- 转换时间将在很大程度上取决于你的硬件。
- 这些模型是计算密集型的。请谨慎行事，并注意你的电脑，确保它不会过热。***我们不对任何硬件损坏负责。***

## 变更日志

- **v4 对比 v5**
   - v5 模型的表现明显优于 v4 模型。
   - 萃取的攻击性可以通过 "Aggression Setting." 来调整。默认值为10，对大多数轨道来说是最佳的。
   - 所有 V2 和 V4 模型已被删除。
   - 增加了合奏模式--这使用户能够从每个模型中获得最有力的结果。
   - 堆积的模型已被完全删除。
     新的攻击性设置和模型组合已经取代了叠加模型功能。
   - NFFT、HOP_SIZE 和 SR 值现在都是内部设置。
   - MDX-NET 人工智能引擎和模型已被添加。
     - 这是一个添加到 UVR GUI 的全新功能。
     - 4个MDX-Net 型号包括在这个包中。
     - 提供的 MDX-Net 模型是由 UVR 的核心开发人员训练的。
     - 这个网络的资源密集度较低，但功能无比强大。
     - MDX-Net是一个混合波形/频谱网络。
   - 加入了 Demucs v3 人工智能引擎和模型。
   - The ability to separate all 4 stems through Demucs v3.

## 故障排除

### 常见问题

- 如果没有安装 FFmpeg，如果用户试图转换一个非 WAV 文件，应用程序将抛出一个错误。
- 内存分配错误通常可以通过降低 "Chunk Size" 来解决。

### 问题报告

在发布新问题时，请尽可能详细。

如果可能的话，请点击 "Start Processing" 按钮左边的 "Settings Button"，并点击 "Error Log" 按钮，以获得可提供给我们的详细错误信息。

## 手动安装（针对开发者）

这些说明只适用于**手动**安装UVR v5.2.0 的人。

1. 下载并安装Python 3.9或更低版本（但不低于3.6）[Python](https://www.python.org/downloads/)
    - **注意:** 确保 *"Add Python to PATH"* 选框被选中
2. 下载源代码 [Github](https://github.com/Anjok07/ultimatevocalremovergui/archive/refs/heads/master.zip)
3. 通过应用程序内的 "Settings" 菜单下载模型。
4. 提取 ultimatevocalremovergui-master.zip 中的 *ultimatevocalremovergui-master* 文件夹至任意位置。
5. 下载 SoX 文件 [SoX](https://sourceforge.net/projects/sox/files/sox/14.4.2/sox-14.4.2-win32.zip/download) 并将其内容提取至 *ultimatevocalremovergui-master/lib_v5/sox* 目录。
6. 从ultimatevocalremovergui-master目录打开命令提示符，分别运行以下命令
```
pip install --no-cache-dir -r requirements.txt
```
```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

从这里你应该能够打开并运行 UVR.py 文件

- FFmpeg 

    - FFmpeg 必须被安装和配置，以便应用程序能够处理任何不是 *.wav* 文件的轨道。你需要查找关于如何在你的操作系统上配置它的说明。

## 许可证

**Ultimate Vocal Remover GUI** 的代码采用 [MIT-licensed](LICENSE). 

- **请注意：** 对于所有希望使用我们的模型的第三方应用程序开发人员，请通过向 UVR 及其开发人员致谢来尊重 MIT 许可。

## 致谢

- [DilanBoskan](https://github.com/DilanBoskan) - 在该项目开始时所做的贡献对于 UVR 的成功至关重要。
- [Bas Curtiz](https://www.youtube.com/user/bascurtiz) - 设计了官方的 UVR 标志、图标、横幅和启动画面。
- [tsurumeso](https://github.com/tsurumeso) - 开发了原始的 VR 架构代码。
- [Kuielab & Woosung Choi](https://github.com/kuielab) - 开发了原始的 MDX-Net AI 代码。
- [Adefossez & Demucs](https://github.com/facebookresearch/demucs) - 开发了原始的 Demucs AI 代码。
- [Hv](https://github.com/NaJeongMo/Colab-for-MDX_B) - 帮助实现了MDX-Net AI代码中的大块内容

## 贡献

- 对于对 **Ultimate Vocal Remover GUI** 的持续开发感兴趣的任何人，请向我们发送 PR，我们将对其进行审核。
- 这个项目是100%开源的，任何人都可以按照自己的意愿免费使用和修改。
- 我们只对 **Ultimate Vocal Remover GUI** 和为其提供的模型进行维护开发和支持。

## 参考文献
- [1] Takahashi et al., "Multi-scale Multi-band DenseNets for Audio Source Separation（用于音源分离的多尺度多波段密集网络）", https://arxiv.org/pdf/1706.09588.pdf
