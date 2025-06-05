from funasr import AutoModel
import sys
import soundfile
from pydub import AudioSegment
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
# paraformer-zh is a multi-functional asr model
# use vad, punc, spk or not as you need

# python3 asr.py /cloudfs-data/db/models/SenseVoiceSmall/ /cloudfs-data/db/models/speech_fsmn_vad_zh-cn-16k-common-pytorch/  /cloudfs-data/db/models/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/
asr_model=sys.argv[1]
vad_model=sys.argv[2] if sys.argv[2] != 'None' else None
punc_model=sys.argv[3] if sys.argv[3] != 'None' else None
wav_file=sys.argv[4]


audio_sample_list = load_audio_text_image_video(
                wav_file,
                fs=16000,
                audio_fs=16000,
                data_type="sound",
                tokenizer=None,
            )


model = AutoModel(model=asr_model,
                  vad_model=vad_model,
                  vad_kwargs={"max_single_segment_time": 60000},
                  disable_update=True,
                  device='cuda:0'
                  # spk_model="cam++"
                  )
print(audio_sample_list)
res = model.generate(input=audio_sample_list, batch_size_s=300, batch_size_threshold_s=60, use_itn=True)
print(res)