from funasr import AutoModel
import sys
# paraformer-zh is a multi-functional asr model
# use vad, punc, spk or not as you need

# python3 asr.py /cloudfs-data/db/models/SenseVoiceSmall/ /cloudfs-data/db/models/speech_fsmn_vad_zh-cn-16k-common-pytorch/ /cloudfs-data/db/models/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/ ../audio.mp3

asr_model=sys.argv[1]
vad_model=sys.argv[2]
punc_model=sys.argv[3]
wav_file=sys.argv[4]

model = AutoModel(model=asr_model,
                  vad_model=vad_model,
                  vad_kwargs={"max_single_segment_time": 60000},
                  punc_model=punc_model,
                  disable_update=True,
                  # spk_model="cam++"
                  )

res = model.generate(input=wav_file, batch_size_s=300, batch_size_threshold_s=60, hotword='你好')
print(res)