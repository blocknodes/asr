from funasr import AutoModel
import sys

model = AutoModel(model=sys.argv[1], model_revision="v2.0.4",device="cuda:0")
#model = AutoModel(model=sys.argv[1], model_revision="v2.0.4",device="cpu")
wav_file = sys.argv[2]
res = model.generate(input=wav_file)
print(res)