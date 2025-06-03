from funasr import AutoModel
import sys

#python3 vad_stream.py  /cloudfs-data/db/models/speech_fsmn_vad_zh-cn-16k-common-pytorch/ ../audio.mp3 cuda:0
#同offline， 唯一区别，部分chunk不完整会以-1作为继续标记
model=sys.argv[1]

chunk_size = 200 # ms
model = AutoModel(model=model, model_revision="v2.0.4",device=sys.argv[3])

import soundfile

wav_file = sys.argv[2]
speech, sample_rate = soundfile.read(wav_file)
chunk_stride = int(chunk_size * sample_rate / 1000)

cache = {}
total_chunk_num = int(len((speech)-1)/chunk_stride+1)
for i in range(total_chunk_num):
    speech_chunk = speech[i*chunk_stride:(i+1)*chunk_stride]
    is_final = i == total_chunk_num - 1
    res = model.generate(input=speech_chunk, cache=cache, is_final=is_final, chunk_size=chunk_size)
    if len(res[0]["value"]):
        print(res)