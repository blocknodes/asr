### 接口
#### spk相关
add_zero_shot_spk(self, prompt_text, prompt_speech_16k, zero_shot_spk_id)：
接口说明：增加零样本说话人信息，
输入：prompt_text为prompt_speech_16k对应的文本，prompt_speech_16k为对应的音频文件，zero_shot_spk_id为零样本说话人ID，用于区分不同的说话
输出：True/False表示操作是否成功或者失败    

def save_spkinfo(self):
接口说明：保存说话人信息model目录中。
输入：无

def inference_zero_shot(self, tts_text, prompt_text, prompt_speech_16k, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True):
接口描述：tts_text为待合成的文本，prompt_text为/prompt_speech_16k分别为text/speech的prompt，zero_shot_spk_id为零样本说话人ID，stream为是否流式输出，speed为语速倍数，text_frontend为是否使用前端文本处理，默认为True。
输出：包含{'tts_speech': tensor([[ 1.8248e-05, -3.1754e-05, -2.1626e-05,  ..., -1.2045e-07,-1.0343e-05,  1.0819e-05]])}字典形式的generator


def inference_vc(self, source_speech_16k, prompt_speech_16k, stream=False, speed=1.0):
接口描述：将声音从source_speech_16k clone到prompt_speech_16k
返回：包含{'tts_speech': tensor([[ 1.8248e-05, -3.1754e-05, -2.1626e-05,  ..., -1.2045e-07,-1.0343e-05,  1.0819e-05]])}字典形式的generator的clone后的声音


#### CosyVoice2 Usage
```python
import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False)

# NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
# zero_shot usage
prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
source_speech_16k = load_wav('/workspace/CosyVoice//asset/cross_lingual_prompt.wav', 16000)
for i, j in enumerate(cosyvoice.inference_vc(source_speech_16k, prompt_speech_16k, stream=False)):
    torchaudio.save('vc_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
    torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# save zero_shot spk for future usage
assert cosyvoice.add_zero_shot_spk('希望你以后能够做的比我还好呦。', prompt_speech_16k, 'my_zero_shot_spk') is True
for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '', '', zero_shot_spk_id='my_zero_shot_spk', stream=False)):
    torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
cosyvoice.save_spkinfo()

# fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L248
for i, j in enumerate(cosyvoice.inference_cross_lingual('在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。', prompt_speech_16k, stream=False)):
    torchaudio.save('fine_grained_control_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# instruct usage
for i, j in enumerate(cosyvoice.inference_instruct2('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '用四川话说这句话', prompt_speech_16k, stream=False)):
    torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# bistream usage, you can use generator as input, this is useful when using text llm model as input
# NOTE you should still have some basic sentence split logic because llm can not handle arbitrary sentence length
def text_generator():
    yield '收到好友从远方寄来的生日礼物，'
    yield '那份意外的惊喜与深深的祝福'
    yield '让我心中充满了甜蜜的快乐，'
    yield '笑容如花儿般绽放。'
for i, j in enumerate(cosyvoice.inference_zero_shot(text_generator(), '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
    torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
```