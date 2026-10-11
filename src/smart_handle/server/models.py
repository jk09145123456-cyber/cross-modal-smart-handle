"""Whisper 음성 인식과 YAMNet 소리 분류 모델 래퍼."""

from __future__ import annotations

# 모델 로딩 상태를 기록하기 위해 사용한다.
import logging



class WhisperTranscriber:
    """faster-whisper 모델로 한국어 안내 음성을 텍스트로 변환한다."""

    def __init__(self, model_name: str, device: str, compute_type: str) -> None:
        # faster_whisper는 음성 인식 모델이 필요할 때만 로드한다.
        # PC/Colab 환경에 따라 cpu 또는 cuda를 선택할 수 있게 인자로 받는다.
        from faster_whisper import WhisperModel

        logging.info(
            "Loading faster-whisper model: %s (device=%s, compute_type=%s)",
            model_name,
            device,
            compute_type,
        )
        self.model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
        )

    def transcribe(self, audio) -> str:
        """오디오 배열을 한국어 문장으로 변환한다."""

        # language="ko"로 지정해서 한국어 음성으로 인식한다.
        # beam_size=1은 속도를 우선한 설정이다.
        # vad_filter=True는 말소리가 아닌 구간을 어느 정도 걸러 준다.
        # condition_on_previous_text=False는 이전 문장에 끌려가 오인식되는 것을 줄이기 위한 설정이다.
        segments, _ = self.model.transcribe(
            audio,
            language="ko",
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()


class YamnetClassifier:
    """YAMNet 모델로 사이렌, 경적, 비명 같은 주변 소리를 분류한다."""

    def __init__(self) -> None:
        # csv는 모델 라벨 파일을 읽고, tensorflow_hub는 YAMNet 모델을 불러온다.
        # YAMNet은 TensorFlow Hub에 공개된 사전 학습 소리 분류 모델이다.
        import csv
        import tensorflow_hub as hub

        logging.info("Loading YAMNet from TensorFlow Hub")
        self.model = hub.load("https://tfhub.dev/google/yamnet/1")

        # 모델이 반환하는 점수는 숫자 인덱스 기준이므로,
        # class_map_path()에서 라벨 파일을 읽어 인덱스를 사람이 읽는 이름으로 바꾼다.
        class_map_path = self.model.class_map_path().numpy().decode("utf-8")
        with open(class_map_path, newline="", encoding="utf-8") as class_map:
            self.class_names = [row["display_name"] for row in csv.DictReader(class_map)]

    def classify(self, audio, top_k: int = 521) -> list[tuple[str, float]]:
        """오디오를 YAMNet에 넣고 점수가 높은 라벨 순서로 반환한다."""

        import numpy as np
        import tensorflow as tf

        # YAMNet은 TensorFlow tensor 형식의 float32 waveform을 입력으로 받는다.
        waveform = tf.convert_to_tensor(audio, dtype=tf.float32)

        # scores는 시간 프레임별 라벨 점수이다.
        # 한 오디오 조각 안에서 한 번이라도 강하게 감지된 소리를 잡기 위해 max(axis=0)을 사용한다.
        scores, _, _ = self.model(waveform)
        peak_scores = np.asarray(scores).max(axis=0)

        # 점수가 높은 순서로 정렬해서 상위 라벨만 반환한다.
        indices = np.argsort(peak_scores)[::-1][:top_k]
        return [(self.class_names[index], float(peak_scores[index])) for index in indices]
