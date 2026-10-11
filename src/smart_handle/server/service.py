"""음성 인식과 소리 분류를 묶어 최종 이벤트를 판단하는 분석 서비스."""

from __future__ import annotations

# 서버 실행 기록과 오류를 출력하기 위해 사용한다.
import logging

# AI 모델을 동시에 호출할 때 충돌하지 않도록 잠금 객체를 사용한다.
import threading

# 처리 시간 측정과 위험 이벤트 유지 시간 계산에 사용한다.
import time

# 소리 분류와 다른 계산을 병렬로 처리하기 위해 사용한다.
from concurrent.futures import ThreadPoolExecutor

# 실제 AI 모델은 models.py, 이벤트 판단 함수는 rules.py에 모아 두었다.
from smart_handle.server.models import WhisperTranscriber, YamnetClassifier
from smart_handle.server.rules import (
    EVENT_PRIORITY,
    detect_sound_event,
    detect_specific_sound_event,
    detect_text_event,
    merge_events,
)

# Raspberry Pi 코드와 같은 이벤트 이름, 같은 샘플레이트를 사용한다.
from smart_handle.protocol import SAMPLE_RATE, Event


class AiService:
    """음성 인식과 소리 분류를 묶어 최종 이벤트를 판단하는 서비스."""

    def __init__(
        self,
        whisper_model: str,
        whisper_device: str,
        whisper_compute_type: str,
        sound_threshold: float,
        speech_threshold: float,
        speech_window_seconds: float,
        scream_threshold: float,
        sound_window_seconds: float,
        hazard_hold_seconds: float,
        voice_rms_threshold: float,
    ) -> None:
        # Whisper는 안내 음성처럼 사람이 말하는 문장을 텍스트로 바꾸는 모델이다.
        self.transcriber = WhisperTranscriber(
            whisper_model,
            whisper_device,
            whisper_compute_type,
        )

        # YAMNet은 사이렌, 경적, 비명 같은 주변 소리를 분류하는 모델이다.
        self.classifier = YamnetClassifier()

        # 각 threshold는 모델 점수가 어느 정도 이상일 때 이벤트로 인정할지 결정한다.
        self.sound_threshold = sound_threshold
        self.speech_threshold = speech_threshold
        self.scream_threshold = scream_threshold

        # 음성 인식은 단어/문장을 보려면 여러 초의 문맥이 필요하므로 speech_buffer를 길게 잡는다.
        self.speech_window_samples = int(SAMPLE_RATE * speech_window_seconds)

        # 주변 위험 소리는 짧은 순간에도 잡아야 하므로 sound_buffer를 비교적 짧게 잡는다.
        self.sound_window_samples = int(SAMPLE_RATE * sound_window_seconds)

        # 사이렌/경적/비명은 한 순간 끊겨도 사용자에게 계속 알려야 하므로 일정 시간 유지한다.
        self.hazard_hold_seconds = hazard_hold_seconds

        # YAMNet speech 라벨이 낮게 나와도 실제 음량이 충분하면 Whisper를 실행하기 위한 RMS 기준이다.
        self.voice_rms_threshold = voice_rms_threshold

        # 최근 위험 소리를 유지하기 위한 상태값이다.
        self.held_sound_event = Event.NONE
        self.held_sound_label = ""
        self.held_sound_score = 0.0
        self.held_sound_until = 0.0

        # 최근 오디오를 누적하는 버퍼이다. 처음에는 아직 데이터가 없으므로 None으로 둔다.
        self.speech_buffer = None
        self.sound_buffer = None

        # AI 모델 객체는 여러 요청이 동시에 접근하면 문제가 생길 수 있어 lock으로 보호한다.
        self.whisper_lock = threading.Lock()
        self.yamnet_lock = threading.Lock()

        # 버퍼도 여러 요청/스레드에서 동시에 접근할 수 있으므로 각각 lock을 둔다.
        self.speech_buffer_lock = threading.Lock()
        self.sound_buffer_lock = threading.Lock()
        self.hazard_lock = threading.Lock()

        # 소리 분류를 다른 계산과 병렬로 돌리기 위한 실행기이다.
        self.executor = ThreadPoolExecutor(max_workers=2)

    def append_speech_audio(self, audio):
        """최근 음성 인식용 오디오를 버퍼에 누적한다."""

        import numpy as np

        with self.speech_buffer_lock:
            # 첫 오디오 조각이면 새 배열로 시작하고, 이후에는 뒤에 이어 붙인다.
            if self.speech_buffer is None:
                self.speech_buffer = np.asarray(audio, dtype=np.float32)
            else:
                self.speech_buffer = np.concatenate((self.speech_buffer, audio))

            # 너무 오래된 오디오는 버리고 최근 speech_window_samples만 유지한다.
            self.speech_buffer = self.speech_buffer[-self.speech_window_samples :]

            # lock 밖에서 분석해도 안전하도록 복사본을 반환한다.
            return self.speech_buffer.copy()

    def clear_speech_audio(self) -> None:
        """안내 문장을 감지한 뒤 같은 문장이 반복 감지되지 않도록 버퍼를 비운다."""

        with self.speech_buffer_lock:
            self.speech_buffer = None

    def append_sound_audio(self, audio):
        """최근 주변 소리 분류용 오디오를 버퍼에 누적한다."""

        import numpy as np

        with self.sound_buffer_lock:
            # 주변 소리 분류용 버퍼도 최근 오디오만 유지한다.
            if self.sound_buffer is None:
                self.sound_buffer = np.asarray(audio, dtype=np.float32)
            else:
                self.sound_buffer = np.concatenate((self.sound_buffer, audio))
            self.sound_buffer = self.sound_buffer[-self.sound_window_samples :]
            return self.sound_buffer.copy()

    def hold_hazard(
        self, event: Event, label: str, score: float
    ) -> tuple[Event, str, float]:
        """사이렌/경적/비명 같은 위험 소리는 짧게 끊겨도 몇 초간 유지한다."""

        now = time.monotonic()
        with self.hazard_lock:
            if event is not Event.NONE:
                # 새 위험 이벤트가 들어오면 기존 이벤트보다 같거나 더 중요할 때 상태를 갱신한다.
                if (
                    self.held_sound_event is Event.NONE
                    or event == self.held_sound_event
                    or EVENT_PRIORITY[event] > EVENT_PRIORITY[self.held_sound_event]
                ):
                    self.held_sound_event = event
                    self.held_sound_label = label
                    self.held_sound_score = score

                # 마지막 감지 시점부터 hazard_hold_seconds 동안 이벤트를 유지한다.
                self.held_sound_until = now + self.hazard_hold_seconds
            if now < self.held_sound_until:
                # 유지 시간이 남아 있으면 지금 들어온 오디오에서 감지가 약해도 이전 이벤트를 반환한다.
                return (
                    self.held_sound_event,
                    self.held_sound_label,
                    self.held_sound_score,
                )

            # 유지 시간이 끝났으면 상태를 비운다.
            self.held_sound_event = Event.NONE
            self.held_sound_label = ""
            self.held_sound_score = 0.0
            return Event.NONE, "", 0.0

    def transcribe(self, audio, should_transcribe: bool) -> tuple[str, Event]:
        """필요할 때만 Whisper를 실행해서 음성을 텍스트 이벤트로 바꾼다."""

        # Whisper는 상대적으로 느리므로 사람 음성이 있을 가능성이 있을 때만 실행한다.
        if not should_transcribe:
            return "", Event.NONE
        with self.whisper_lock:
            text = self.transcriber.transcribe(audio)

        # 텍스트로 바뀐 문장 안에서 좌회전/우회전/과속 키워드를 찾는다.
        return text, detect_text_event(text)

    def classify_sound(self, audio) -> tuple[Event, str, float, list[tuple[str, float]]]:
        """YAMNet으로 주변 소리를 분류하고 위험 소리 이벤트를 찾는다."""

        with self.yamnet_lock:
            predictions = self.classifier.classify(audio)
        sound_event, sound_label, sound_score = detect_sound_event(
            predictions, self.sound_threshold
        )

        # 비명은 일반 위험 소리보다 점수가 낮게 나오는 경우가 있어 별도 threshold로 한 번 더 검사한다.
        scream_event, scream_label, scream_score = detect_specific_sound_event(
            predictions, Event.SCREAM, self.scream_threshold
        )
        if scream_event is Event.SCREAM:
            sound_event, sound_label, sound_score = scream_event, scream_label, scream_score
        return sound_event, sound_label, sound_score, predictions

    def analyze(self, audio) -> dict:
        """Raspberry Pi에서 받은 오디오 한 조각을 분석해 최종 이벤트를 만든다."""

        started_at = time.monotonic()

        # 음성 인식은 조금 긴 문맥이 필요하고, 위험 소리 분류는 짧은 창으로도 충분하다.
        speech_audio = self.append_speech_audio(audio)
        sound_audio = self.append_sound_audio(audio)

        # 소리 분류는 시간이 걸리므로 다른 계산과 동시에 실행한다.
        sound_future = self.executor.submit(self.classify_sound, sound_audio)
        sound_event, sound_label, sound_score, predictions = sound_future.result()
        sound_event, sound_label, sound_score = self.hold_hazard(
            sound_event, sound_label, sound_score
        )

        # YAMNet이 사람 목소리를 감지했거나 RMS가 충분히 크면 Whisper를 실행한다.
        voice_labels = ("speech", "conversation", "narration", "monologue")
        speech_score = max(
            (
                score
                for label, score in predictions
                if any(name in label.lower() for name in voice_labels)
            ),
            default=0.0,
        )
        import numpy as np

        # RMS는 오디오의 평균 음량을 나타내는 값이다.
        # YAMNet speech 라벨이 낮아도 실제 음량이 있으면 Whisper를 실행할 수 있게 보조 기준으로 사용한다.
        voice_rms = float(np.sqrt(np.mean(np.square(speech_audio))))
        # 외부 위험 소리는 느린 음성 인식을 기다리지 않고 바로 처리한다.
        text, stt_event = self.transcribe(
            speech_audio,
            sound_event is Event.NONE
            and (
                speech_score >= self.speech_threshold
                or voice_rms >= self.voice_rms_threshold
            ),
        )
        if stt_event is not Event.NONE:
            # 문장을 한 번 이벤트로 처리했으면 같은 안내가 반복 감지되지 않도록 음성 버퍼를 비운다.
            self.clear_speech_audio()

        # 안내 음성과 주변 위험 소리 중 더 중요한 이벤트를 최종 결과로 선택한다.
        final_event = merge_events(stt_event, sound_event)
        top_predictions = [
            {"label": label, "score": round(score, 4)}
            for label, score in predictions[:5]
        ]

        # 로그에는 처리 시간, 감지된 텍스트, 소리 라벨, 최종 이벤트를 남겨 디버깅할 수 있게 한다.
        logging.info(
            "Analysis completed in %.2fs: speech_window=%.2fs sound_window=%.2fs text=%r sound=%s(%.3f) final=%s top=%s",
            time.monotonic() - started_at,
            len(speech_audio) / SAMPLE_RATE,
            len(sound_audio) / SAMPLE_RATE,
            text,
            sound_label or "none",
            sound_score,
            final_event.value,
            top_predictions,
        )
        return {
            # Arduino로 보낼 최종 이벤트 이름이다.
            "event": final_event.value,

            # Whisper가 인식한 원문 텍스트와, 텍스트 기준 이벤트이다.
            "text": text,
            "stt_event": stt_event.value,

            # YAMNet 기준 주변 소리 이벤트와 해당 라벨/점수이다.
            "sound_event": sound_event.value,
            "sound_label": sound_label,
            "sound_score": sound_score,

            # Whisper 실행 여부를 판단할 때 참고한 사람 음성 점수와 RMS 값이다.
            "speech_score": speech_score,
            "voice_rms": voice_rms,

            # 확인용 상위 YAMNet 예측 결과이다.
            "top_predictions": top_predictions,
        }
