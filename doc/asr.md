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
- `输出`(list(dict)): [{'key': 'audio', 'text': '<|zh|><|HAPPY|><|Speech|><|withitn|>喂，你好，嗯，，你有什么事？呃，我这边是汽车咨询之家，呃，现在呢是联合各大汽车品牌给您带来最新购车活动力度。呃，我长话短说啊，给您简单介绍一下可以吗？ <|zh|><|NEUTRAL|><|Speech|><|withitn|>性价比最高的车型是哪款？简单介绍一下它的性能没？呃，咱们各大汽车品牌都有，您是比较喜欢哪个品牌，哪个车型呢？ <|zh|><|NEUTRAL|><|Speech|><|withitn|>啊，您是在杭州买车方便，对吗
？ <|zh|><|NEUTRAL|><|Speech|><|withitn|>您继续。'}]。text内需要将特殊含义字符去掉



###asr示例代码（包含vad）
```python
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

model_dir = "iic/SenseVoiceSmall"


model = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    remote_code="./model.py",  
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
)

# en
res = model.generate(
    input=f"{model.model_path}/example/en.mp3",
    cache={},
    language="zh",  # "zh", "en", "yue", "ja", "ko", "nospeech"
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,
    merge_length_s=15,
)
text = rich_transcription_postprocess(res[0]["text"])
print(text)
```

注意：
- `model_dir`：模型名称，或本地磁盘中的模型路径。
- `trust_remote_code`：
  - `True` 表示 model 代码实现从 `remote_code` 处加载，`remote_code` 指定 `model` 具体代码的位置（例如，当前目录下的 `model.py`），支持绝对路径与相对路径，以及网络 url。
  - `False` 表示，model 代码实现为 [FunASR](https://github.com/modelscope/FunASR) 内部集成版本，此时修改当前目录下的 `model.py` 不会生效，因为加载的是 funasr 内部版本，模型代码 [点击查看](https://github.com/modelscope/FunASR/tree/main/funasr/models/sense_voice)。
- `vad_model`：表示开启 VAD，VAD 的作用是将长音频切割成短音频，此时推理耗时包括了 VAD 与 SenseVoice 总耗时，为链路耗时，如果需要单独测试 SenseVoice 模型耗时，可以关闭 VAD 模型。
- `vad_kwargs`：表示 VAD 模型配置，`max_single_segment_time`: 表示 `vad_model` 最大切割音频时长，单位是毫秒 ms。
- `use_itn`：输出结果中是否包含标点与逆文本正则化，正常开启。
- `batch_size_s` 表示采用动态 batch，batch 中总音频时长，单位为秒 s。
- `merge_vad`：是否将 vad 模型切割的短音频碎片合成，合并后长度为 `merge_length_s`，单位为秒 s。
- `ban_emo_unk`：禁用 emo_unk 标签，禁用后所有的句子都会被赋与情感标签。默认 `False`

###asr示例代码（不包含vad）
```python
from funasr import AutoModel
import sys
# paraformer-zh is a multi-functional asr model
# use vad, punc, spk or not as you need

# python3 asr.py /cloudfs-data/db/models/SenseVoiceSmall/ /cloudfs-data/db/models/speech_fsmn_vad_zh-cn-16k-common-pytorch/  /cloudfs-data/db/models/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/
asr_model=sys.argv[1]
vad_model=sys.argv[2] if sys.argv[2] != 'None' else None
punc_model=sys.argv[3] if sys.argv[3] != 'None' else None
wav_file=sys.argv[4]

model = AutoModel(model=asr_model,
                  vad_model=vad_model,
                  vad_kwargs={"max_single_segment_time": 60000},
                  disable_update=True,
                  device='cuda:0'
                  # spk_model="cam++"
                  )

res = model.generate(input=wav_file, batch_size_s=300, batch_size_threshold_s=60, use_itn=True)
print(res)
```

### vad和asr串接代码
https://github.com/modelscope/FunASR/blob/main/funasr/auto/auto_model.py#L381

asr可以组batch或者动态组batch，串接代码是组动态batch，参照：
https://github.com/modelscope/FunASR/blob/main/funasr/auto/auto_model.py#L455