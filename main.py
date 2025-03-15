import time
import torch
import subprocess
import numpy as np
import soundfile as sf  # 需要安装：pip install soundfile
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
from datasets import load_dataset
import jiwer
import pandas as pd
import tempfile
import os
import uuid

# 配置参数
DATASET_NAME = "hf-internal-testing/librispeech_asr_dummy"
DATASET_SPLIT = "validation"
SAMPLE_LIMIT = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEEPSPEECH_MODEL = "./deepspeech-0.9.3-models.pbmm"
DEEPSPEECH_SCORER = "./deepspeech-0.9.3-models.scorer"

# 加载数据集
dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT, trust_remote_code=True)
dataset = dataset.select(range(SAMPLE_LIMIT))
references = [item['text'].lower() for item in dataset]

TMP_DIR = "tmp"
os.makedirs(TMP_DIR, exist_ok=True)  # 确保临时目录存在

# -------------------------------------------------------------------
# 1. 测试Whisper-small（使用直接音频数组）
# -------------------------------------------------------------------
def test_whisper():
    model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-small")
    processor = AutoProcessor.from_pretrained("openai/whisper-small")
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=DEVICE
    )

    predictions, times = [], []
    for item in dataset:
        audio = item['audio']['array']  # 直接获取音频数组
        start = time.time()
        result = pipe(audio.copy(), generate_kwargs={"language": "english"})
        times.append(time.time() - start)
        predictions.append(result["text"].lower())

    return predictions, np.mean(times)

# -------------------------------------------------------------------
# 2. 测试Wav2Vec2-base（使用直接音频数组）
# -------------------------------------------------------------------
def test_wav2vec2():
    pipe = pipeline(
        "automatic-speech-recognition",
        model="facebook/wav2vec2-base-960h",
        device=DEVICE
    )

    predictions, times = [], []
    for item in dataset:
        audio = item['audio']['array']  # 直接获取音频数组
        start = time.time()
        result = pipe(audio.copy())
        times.append(time.time() - start)
        predictions.append(result["text"].lower())

    return predictions, np.mean(times)

# -------------------------------------------------------------------
# 3. 测试DeepSpeech-0.9（使用直接音频数组）
# -------------------------------------------------------------------
def test_deepspeech():
    predictions, times = [], []

    for item in dataset:
        # 生成唯一文件名
        filename = os.path.join(TMP_DIR, f"temp_{uuid.uuid4().hex}.wav")
        audio = item['audio']['array']

        try:
            # 保存音频文件到临时目录
            sf.write(filename, audio, item['audio']['sampling_rate'], subtype='PCM_16')

            cmd = f"""conda activate money_39;deepspeech --model {DEEPSPEECH_MODEL} --scorer {DEEPSPEECH_SCORER} --audio {filename}"""

            # 执行命令并计时
            start = time.time()
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                shell=True,
            )
            text = result.stdout.strip().lower()
        except subprocess.CalledProcessError as e:
            print(f"DeepSpeech error: {e.stderr}")
            text = ""
        finally:
            # 确保删除临时文件
            if os.path.exists(filename):
                os.remove(filename)

        times.append(time.time() - start)
        predictions.append(text)

    return predictions, np.mean(times)

# 计算词错误率
def calculate_wer(predictions, references):
    return jiwer.wer(references, predictions)

# 执行测试
results = []
for name, test_fn in [
    ("Whisper-small", test_whisper),
                      ("Wav2Vec2-base", test_wav2vec2),
                      ("DeepSpeech-0.9", test_deepspeech)]:
    preds, avg_time = test_fn()
    wer = calculate_wer(preds, references)
    results.append({
        "Model": name,
        "Avg Inference Time (s)": avg_time,
        "Word Error Rate": wer
    })

# 显示结果
df = pd.DataFrame(results)
print("\nBenchmark Results:")
print(df.to_markdown(index=False))
