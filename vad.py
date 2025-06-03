from funasr import AutoModel
import sys

#python3 vad.py  /cloudfs-data/db/models/speech_fsmn_vad_zh-cn-16k-common-pytorch/ ../audio.mp3 cuda:0
#python3 vad.py  /cloudfs-data/db/models/speech_fsmn_vad_zh-cn-16k-common-pytorch/ ../audio.mp3 cpu
#cuda:rtf_avg: 0.008
#cpu:rtf_avg: 0.004

#input: 音频文件
#output: 有声音的片段，list表示，单位ms

model = AutoModel(model=sys.argv[1], model_revision="v2.0.4",device=sys.argv[3])

wav_file = sys.argv[2]
res = model.generate(input=wav_file)
print(res)