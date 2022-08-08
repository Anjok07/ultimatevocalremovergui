# Ultimate Vocal Remover GUI v5.4.0
<img src="https://raw.githubusercontent.com/Anjok07/ultimatevocalremovergui/master/img/UVR_v54.png?raw=true" />

[![Release](https://img.shields.io/github/release/anjok07/ultimatevocalremovergui.svg)](https://github.com/anjok07/ultimatevocalremovergui/releases/latest)
[![Downloads](https://img.shields.io/github/downloads/anjok07/ultimatevocalremovergui/total.svg)](https://github.com/anjok07/ultimatevocalremovergui/releases)

[English](README.md) | 简体中文

## 关于

本程序使用了最先进的音源分离模型，以去除音频文件中的人声。UVR 的核心开发人员训练了这个软件包中提供的所有模型（除了 Demucs 的辅助模型）。

- **核心开发者**
    - [Anjok07](https://github.com/anjok07)
    - [aufr33](https://github.com/aufr33)

## 安装

### Windows 安装

该安装包包含 UVR 接口、Python、PyTorch 和其他有效运行应用程序所需的依赖项。不需要任何先决条件，即装即用。

- 请注意：
    - 该安装程序适用于 Windows 10 或更高版本。
    - 不保证在 Windows 7 或更低版本时的应用功能。
    - 不保证英特尔奔腾和赛扬 CPU 的应用功能。

- 通过以下链接下载UVR安装程序：
    - [主程序下载链接](https://uvr.uvr.workers.dev/UVR_v5.4.0_setup.exe)
    - [主程序镜像链接](https://www.mediafire.com/file/nrakuh8t8p993y8/UVR_v5.4.0_setup.exe)
- 为已经安装了 UVR 的用户更新软件包的说明：
    - 从 UVR v5.4 更新 - [更新包](https://github.com/Anjok07/ultimatevocalremovergui/releases/download/v5.4.0/UVR_v5.4_Update_Package.exe)

- **可选项**
    - 额外的模型和程序补丁可以通过程序内的 "Settings" 菜单下载。

- **请注意：** 最新发布版本请见“最新发布”页面 [Releases](https://github.com/Anjok07/ultimatevocalremovergui/releases/tag/v5.4.0)

### 其他平台

本程序可以在Mac和Linux上手动安装并运行（更多信息请参见下面的**手动开发者安装**部分）。有些功能在非Windows平台上可能无法使用。

## 应用手册

**一般选项**

<img src="https://github.com/miria0/ultimatevocalremovergui/blob/master/img/gen_opt_CN.png?raw=true" />

**VR 架构选项**

<img src="https://github.com/miria0/ultimatevocalremovergui/blob/master/vr_opt_CN.png?raw=true" />

**MDX-Net 选项**

<img src="https://github.com/Anjok07/ultimatevocalremovergui/blob/master/img/mdx_opt.png?raw=true" />

**Demucs v3 选项**

<img src="https://github.com/Anjok07/ultimatevocalremovergui/blob/master/img/demucs_opt.png?raw=true" />

**合奏选项**

<img src="https://github.com/Anjok07/ultimatevocalremovergui/blob/master/img/ense_opt.png?raw=true" />

**手动合奏**

<img src="https://github.com/Anjok07/ultimatevocalremovergui/blob/master/img/user_ens_opt.png?raw=true" />

### 其他应用说明

- 建议使用至少有 8GB 显存的 nVidia GPU。
- 该应用程序只兼容 64 位平台。
- 该应用程序依赖于 Sox - Sound Exchange 的降噪。
- 该应用程序依赖于 FFmpeg 来处理非 wav 格式的音频文件。
- 应用程序将在关闭时自动记住你的设置。
- 转换时间将在很大程度上取决于你的硬件。
- 程序所采用的模型是计算密集型的，请谨慎行事，并在程序运行时时刻关注你的电脑，确保它不会过热。***我们不对任何硬件损坏负责。***

## 更新日志

- **v4 对比 v5**
   - v5 模型的表现明显优于 v4 模型。
   - 音频提取的力度 (Agreesiveness) 可以通过 "Aggression Setting." 来调整。默认值为10，对大多数音频来说是最佳的。
   - 所有 V2 和 V4 模型已被移除。
   - 增加了“混合模式” -- 这使用户能够从每个模型中得到最为稳健的结果。
   - “模型堆叠”选项已被完全移除，以新的“力度选项”以及模型混合模式取代之。
   - 现在NFFT、HOP_SIZE 和 SR 等值已均能在程序内设置。
   - 添加了 MDX-NET 人工智能引擎和模型。
     - 这是 UVR GUI 的新增的功能。
     - Package内包括了4个 MDX-Net 模型。
     - 内嵌的 MDX-Net 模型是由 UVR 的核心开发人员训练的。
     - 该神经网络无需大量计算资源，但其无比强大。
     - MDX-Net 是一个 Hybrid Waveform/Spectrogram network (混合型波形/频谱网络)。
   - 加入了 Demucs v3 人工智能引擎和模型。
   - 可以通过 Demucs v3 分离音乐的四个器乐部分了。

## 故障排除

### 常见问题

- 若没有正确安装并配置 FFmpeg，并试图转换一个非 WAV 文件，本程序将抛出一个错误。
- 内存分配错误通常可以通过降低 "Chunk Size" （分块大小）来解决。

### 问题报告

在发布新 Issue 时，请尽可能详细描述。

如果可以的话，请点击 "Start Processing" 按钮左边的 "Settings Button"，再点击 "Error Log" 按钮，以获得可提供给我们的详细错误信息。

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

    - 必须预先安装并配置好FFmpeg ，以便本程序能够处理非 *.wav* 文件的轨道。请查阅相关资料，并在你所使用的操作系统上配置好该工具库。

## 许可证

**Ultimate Vocal Remover GUI** 的代码采用 [MIT-licensed](LICENSE). 

- **请注意：** 所有希望使用我们的模型的第三方应用程序开发人员，请尊重 MIT 许可，向 UVR 及其开发人员致谢。

## 致谢

- [DilanBoskan](https://github.com/DilanBoskan) - 在该项目伊始时为UVR做出了至关的重要贡献。
- [Bas Curtiz](https://www.youtube.com/user/bascurtiz) - 设计了 UVR 的官方标志、图标、横幅和启动画面。
- [tsurumeso](https://github.com/tsurumeso) - 开发了原始的 VR 架构代码。
- [Kuielab & Woosung Choi](https://github.com/kuielab) - 开发了原始的 MDX-Net AI 代码。
- [Adefossez & Demucs](https://github.com/facebookresearch/demucs) - 开发了原始的 Demucs AI 代码。
- [Hv](https://github.com/NaJeongMo/Colab-for-MDX_B) - 帮助实现了MDX-Net AI代码中的大量内容

## 向本项目贡献

- 对 **Ultimate Vocal Remover GUI** 的持续开发感兴趣的任何人，请向我们发送 PR，我们将对其进行审核。
- 这个项目是100%开源的，任何人都可以按照自己的意愿免费使用和修改。
- 我们只对 **Ultimate Vocal Remover GUI** 和为其提供的模型进行维护开发和支持。

## 参考文献
- [1] Takahashi et al., "Multi-scale Multi-band DenseNets for Audio Source Separation（用于音源分离的多尺度多波段密集网络）", https://arxiv.org/pdf/1706.09588.pdf
