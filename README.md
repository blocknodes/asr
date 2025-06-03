# asr
## bench mark wer

```
cd FunASR/examples/aishell/paraformer
funasr ++model=/cloudfs-data/db/models/SenseVoiceSmall/  +disable_update=True +input=data/data/test/wav.scp | tee /cloudfs/db/myasr/sensevoice_result
python3 /cloudfs/db/myasr/json_to_line.py /cloudfs/db/myasr/sensevoice_result /cloudfs/db/myasr/sensevoice_result_new
python3 utils/postprocess_text_zh.py /cloudfs/db/myasr/sensevoice_result_new /cloudfs/db/myasr/sensevoice_result_post
python3  utils/compute_wer.py /cloudfs/db/myasr/sensevoice_result_post  test_post sense_wer
tail -n 3 sense_wer
```

%WER 3.09 [ 3236 / 104820, 78 ins, 133 del, 3025 sub ]
%SER 27.94 [ 2005 / 7176 ]
Scored 7176 sentences, 0 not present in hyp.


