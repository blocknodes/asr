from funasr import AutoModel
import sys
# paraformer-zh is a multi-functional asr model
# use vad, punc, spk or not as you need

# python3 asr.py /cloudfs-data/db/models/SenseVoiceSmall/ /cloudfs-data/db/models/speech_fsmn_vad_zh-cn-16k-common-pytorch/  /cloudfs-data/db/models/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/
asr_model=sys.argv[1]
wav_file=sys.argv[2]

model = AutoModel(model=asr_model,
                  disable_update=True,
                  )

res = model.generate(input=wav_file, batch_size_s=300, batch_size_threshold_s=60)
print(res)