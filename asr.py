from funasr import AutoModel
import sys
# paraformer-zh is a multi-functional asr model
# use vad, punc, spk or not as you need

# python3 asr.py /cloudfs-data/db/models/SenseVoiceSmall/ /cloudfs-data/db/models/speech_fsmn_vad_zh-cn-16k-common-pytorch/ ` ../audio.mp3

asr_model=sys.argv[1]
vad_model=sys.argv[2] if sys.argv[2] != 'None' else None
punc_model=sys.argv[3] if sys.argv[3] != 'None' else None
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