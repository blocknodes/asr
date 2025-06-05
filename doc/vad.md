### 接口说明

#### AutoModel 定义
```python
model = AutoModel(model=[str], device=[str], ncpu=[int], output_dir=[str], batch_size=[int], hub=[str], **kwargs)
```
- `model`(str): [模型仓库](https://github.com/alibaba-damo-academy/FunASR/tree/main/model_zoo) 中的模型名称，或本地磁盘中的模型路径
- `device`(str): `cuda:0`（默认gpu0），使用 GPU 进行推理，指定。如果为`cpu`，则使用 CPU 进行推理。`mps`：mac电脑M系列新品试用mps进行推理。`xpu`：使用英特尔gpu进行推理。
- `ncpu`(int): `4` （默认），设置用于 CPU 内部操作并行性的线程数
- `output_dir`(str): `None` （默认），如果设置，输出结果的输出路径
- `batch_size`(int): `1` （默认），解码时的批处理，样本个数
- `hub`(str)：`ms`（默认），从modelscope下载模型。如果为`hf`，从huggingface下载模型。
- `**kwargs`(dict): 所有在`config.yaml`中参数，均可以直接在此处指定，例如，vad模型中最大切割长度 `max_single_segment_time=6000` （毫秒）。

#### AutoModel 推理
```python
res = model.generate(input=[str], output_dir=[str])
```
- `input`: 要解码的输入，可以是：
  - wav文件路径, 例如: asr_example.wav
  - pcm文件路径, 例如: asr_example.pcm，此时需要指定音频采样率fs（默认为16000）
  - 音频字节数流，例如：麦克风的字节数数据
  - wav.scp，kaldi 样式的 wav 列表 (`wav_id \t wav_path`), 例如:
  ```text
  asr_example1  ./audios/asr_example1.wav
  asr_example2  ./audios/asr_example2.wav
  ```
  在这种输入 `wav.scp` 的情况下，必须设置 `output_dir` 以保存输出结果
  - 音频采样点，例如：`audio, rate = soundfile.read("asr_example_zh.wav")`, 数据类型为 numpy.ndarray。支持batch输入，类型为list：
  ```[audio_sample1, audio_sample2, ..., audio_sampleN]```
  - fbank输入，支持组batch。shape为[batch, frames, dim]，类型为torch.Tensor，例如
- `output_dir`: None （默认），如果设置，输出结果的输出路径
- `**kwargs`(dict): 与模型相关的推理参数，例如，`beam_size=10`，`decoding_ctc_weight=0.1`。


###vad示例代码：
```python
from funasr import AutoModel
import sys

#python3 vad.py  /cloudfs-data/db/models/speech_fsmn_vad_zh-cn-16k-common-pytorch/ ../audio.mp3 cuda:0
#python3 vad.py  /cloudfs-data/db/models/speech_fsmn_vad_zh-cn-16k-common-pytorch/ ../audio.mp3 cpu
#cuda:rtf_avg: 0.008
#cpu:rtf_avg: 0.004


# model: vad模型路径
# device : cuda or cpu
# disable_update: 是否自动检测模型更新并下载
# fsmn当前支持batch_size=1
# vad模型相关参数可以通过kwargs传入，例如max_single_segment_time=3000，所有vad参数请参考下面表格。正常情况下默认值即可
model = AutoModel(model=sys.argv[1],device=sys.argv[3],disable_update=True)

wav_file = sys.argv[2]

#输入：支持单条音频文件识别，也支持文件列表，列表为kaldi风格wav.scp：wav_id wav_path
#输出：VAD模型输出格式为：[{'key':key1, 'value':[[beg1, end1], [beg2, end2], .., [begN, endN]]}，
#{'key':key2, 'value':[[beg1, end1], [beg2, end2], .., [begN, endN]]}，] 如果输入是wav.scp，则key与wav_id对应
#其中begN/endN表示第N个有效音频片段的起始点/结束点， 单位为毫秒
res = model.generate(input=wav_file)
print(res)
```


以下是FSMN模型相关参数的解释表格：

| 参数名称                         | 类型     | 作用说明                                                                 | 默认值/取值 | 备注 |
|----------------------------------|----------|--------------------------------------------------------------------------|-------------|------|
| **sample_rate**                  | int      | 采样率（Hz）                                                             | 16000       | 通常为16kHz |
| **detect_mode**                  | int      | 检测模式（1: 启用端点检测，0: 禁用）                                    | 1           | 1表示启用端点检测 |
| **snr_mode**                     | int      | 信噪比模式（0: 不使用SNR，1: 使用SNR）                                 | 0           | 0表示不启用SNR |
| **max_end_silence_time**         | int      | 最大结束静音时间（ms）                                                 | 800         | 静音段最大持续时间 |
| **max_start_silence_time**       | int      | 最大开始静音时间（ms）                                                 | 3000        | 前导静音最大持续时间 |
| **do_start_point_detection**     | bool     | 是否启用起始点检测                                                      | True        | True表示启用 |
| **do_end_point_detection**       | bool     | 是否启用结束点检测                                                      | True        | True表示启用 |
| **window_size_ms**              | int      | 窗口大小（ms），用于特征提取                                           | 200         | 200ms窗口 |
| **sil_to_speech_time_thres**     | int      | 静音到语音切换时间阈值（ms）                                           | 150         | 切换敏感度 |
| **speech_to_sil_time_thres**     | int      | 语音到静音切换时间阈值（ms）                                           | 150         | 切换敏感度 |
| **speech_2_noise_ratio**         | float    | 语音与噪声的比率（用于噪声抑制）                                       | 1.0         | 比率阈值 |
| **do_extend**                    | int      | 是否启用扩展（1: 启用，0: 禁用）                                       | 1           | 1表示启用扩展 |
| **lookback_time_start_point**    | int      | 起始点检测的回溯时间（ms）                                             | 200         | 回溯窗口 |
| **lookahead_time_end_point**     | int      | 结束点检测的前瞻时间（ms）                                             | 100         | 前瞻窗口 |
| **max_single_segment_time**      | int      | 单个语音段最大时间（ms）                                               | 60000       | 1小时限制 |
| **snr_thres**                    | float    | 信噪比阈值（dB）                                                       | -100.0      | 低阈值用于噪声抑制 |
| **noise_frame_num_used_for_snr** | int      | 用于计算SNR的帧数                                                      | 100         | 帧数影响SNR计算 |
| **decibel_thres**                | float    | 分贝阈值（dB）                                                          | -100.0      | 与SNR阈值相关 |
| **speech_noise_thres**           | float    | 语音噪声阈值（用于区分语音/噪声）                                      | 0.6         | 阈值范围 |
| **speech_noise_thresh_low**      | float    | 语音噪声低阈值                                                          | -0.1        | 阈值范围下限 |
| **speech_noise_thresh_high**     | float    | 语音噪声高阈值                                                          | 0.3         | 阈值范围上限 |
| **fe_prior_thres**               | float    | 特征先验阈值（用于过滤低概率特征）                                    | 0.0001      | 过滤噪声 |
| **silence_pdf_num**              | int      | 静音PDF数量                                                             | 1           | 静音概率分布数量 |
| **sil_pdf_ids**                  | list     | 静音PDF ID列表                                                          | [0]         | 静音模型ID |
| **output_frame_probs**           | bool     | 是否输出帧概率                                                          | False       | False表示不输出 |
| **frame_in_ms**                  | int      | 帧间隔（ms）                                                            | 10          | 每10ms一帧 |
| **frame_length_ms**              | int      | 帧长度（ms）                                                            | 25          | 25ms帧长度 |

### 关键说明：
1. **端点检测**：通过 `do_start_point_detection` 和 `do_end_point_detection` 控制，结合静音时间阈值（`sil_to_speech_time_thres`、`speech_to_sil_time_thres`）判断语音段边界。
2. **噪声抑制**：`speech_2_noise_ratio` 和 `speech_noise_thres` 用于区分语音与噪声，避免误检。
3. **SNR相关参数**：`snr_mode` 控制是否启用信噪比分析，`snr_thres` 和 `decibel_thres` 共同作用于噪声抑制。
4. **扩展功能**：`do_extend` 启用后，允许语音段跨越大时间窗口（`max_single_segment_time`）。
5. **帧粒度**：`frame_in_ms` 和 `frame_length_ms` 定义语音处理的粒度，影响时序分析精度。

此配置适用于语音识别、语音活动检测（VAD）等场景，需根据实际需求调整阈值和参数。