from funasr import AutoModel
import sys

#python3 vad.py  /cloudfs-data/db/models/speech_fsmn_vad_zh-cn-16k-common-pytorch/ ../audio.mp3 cuda:0
#python3 vad.py  /cloudfs-data/db/models/speech_fsmn_vad_zh-cn-16k-common-pytorch/ ../audio.mp3 cpu
#cuda:rtf_avg: 0.008
#cpu:rtf_avg: 0.004


# model: vad模型路径
# device : cuda or cpu
# disable_update: 是否自动检测模型更新并下载
model = AutoModel(model=sys.argv[1],device=sys.argv[3],disable_update=True)

wav_file = sys.argv[2]

#输入：支持单条音频文件识别，也支持文件列表，列表为kaldi风格wav.scp：wav_id wav_path
#输出：VAD模型输出格式为：[{'key':key1, 'value':[[beg1, end1], [beg2, end2], .., [begN, endN]]}，
#{'key':key2, 'value':[[beg1, end1], [beg2, end2], .., [begN, endN]]}，] 如果输入是wav.scp，则key与wav_id对应
#其中begN/endN表示第N个有效音频片段的起始点/结束点， 单位为毫秒
res = model.generate(input=wav_file)
print(res)