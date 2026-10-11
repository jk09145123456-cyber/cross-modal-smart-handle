"""마이크 녹음과 디버그용 WAV 저장."""

from __future__ import annotations

# 마이크 설정을 로그로 확인하기 위해 사용한다.
import logging

# 녹음 스레드에서 메인 스레드로 오디오 데이터를 전달하기 위해 사용한다.
import queue

# 녹음을 백그라운드에서 계속 실행하기 위해 사용한다.
import threading

# 디버그 WAV 파일 저장 경로를 다루기 위해 사용한다.
from pathlib import Path

# 서버와 같은 샘플레이트를 사용한다.
from smart_handle.protocol import SAMPLE_RATE


class AudioRecorder:
    """마이크에서 오디오를 계속 녹음해 메인 루프에 전달한다."""

    def __init__(
        self,
        seconds: float,
        device: int | str | None = None,
        input_sample_rate: int | None = None,
    ) -> None:
        # sounddevice는 Raspberry Pi의 마이크 입력을 받기 위해 필요하다.
        import sounddevice as sd

        # sounddevice 모듈을 객체에 저장해 다른 메서드에서도 같은 모듈을 사용한다.
        self.sd = sd

        # 한 번에 녹음할 길이이다. 기본값은 1초이다.
        self.seconds = seconds

        # 사용할 마이크 장치 번호 또는 이름이다. None이면 기본 입력 장치를 사용한다.
        self.device = device

        # 선택된 마이크의 기본 샘플레이트를 확인한다.
        device_info = sd.query_devices(device, "input")
        self.input_sample_rate = input_sample_rate or int(device_info["default_samplerate"])

        # 녹음 스레드가 만든 오디오를 메인 루프가 가져가도록 queue를 사용한다.
        # maxsize=2로 작게 둔 이유는 처리 지연이 생겼을 때 오래된 오디오가 쌓이지 않게 하기 위해서이다.
        self.audio_queue: queue.Queue = queue.Queue(maxsize=2)

        # close()가 호출되면 녹음 스레드를 멈추기 위한 이벤트이다.
        self.stopped = threading.Event()

        # 실제 녹음 전에 장치가 요청한 설정을 지원하는지 확인한다.
        sd.check_input_settings(
            device=device,
            channels=1,
            dtype="float32",
            samplerate=self.input_sample_rate,
        )
        logging.info(
            "Microphone: %s, input sample rate: %d Hz, AI sample rate: %d Hz",
            device_info["name"],
            self.input_sample_rate,
            SAMPLE_RATE,
        )

        # 녹음은 백그라운드 스레드에서 계속 돌고, 메인 루프는 서버 통신과 Arduino 전송을 처리한다.
        self.thread = threading.Thread(target=self._record_loop, daemon=True)
        self.thread.start()

    def _record_chunk(self):
        """지정한 시간만큼 녹음하고 AI 서버가 쓰는 샘플레이트로 변환한다."""

        # numpy는 오디오 배열 처리, scipy는 샘플레이트 변환에 필요하다.
        import numpy as np
        from scipy.signal import resample_poly

        # 마이크 입력 샘플레이트 기준으로 녹음해야 할 프레임 수를 계산한다.
        frames = int(self.input_sample_rate * self.seconds)

        # channels=1로 모노 오디오를 녹음한다.
        # dtype=float32로 받아야 서버 전송 형식과 맞추기 쉽다.
        audio = self.sd.rec(
            frames,
            samplerate=self.input_sample_rate,
            channels=1,
            dtype="float32",
            device=self.device,
        )
        self.sd.wait()

        # sounddevice는 (프레임 수, 채널 수) 형태로 반환하므로 1차원 배열로 바꾼다.
        audio = np.squeeze(audio)

        # 마이크 기본 샘플레이트가 16 kHz가 아니면 AI 서버 기준인 16 kHz로 변환한다.
        if self.input_sample_rate != SAMPLE_RATE:
            audio = resample_poly(audio, SAMPLE_RATE, self.input_sample_rate)

        # 변환 과정에서 길이가 조금 달라질 수 있어 정확히 기대 길이만큼 자른다.
        expected_frames = int(SAMPLE_RATE * self.seconds)
        return np.asarray(audio[:expected_frames], dtype=np.float32)

    def _record_loop(self) -> None:
        """백그라운드 스레드에서 계속 녹음한다."""

        while not self.stopped.is_set():
            audio = self._record_chunk()

            # 처리 속도보다 녹음이 빨라지면 가장 오래된 오디오를 버려 지연을 줄인다.
            if self.audio_queue.full():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    pass
            self.audio_queue.put(audio)

    def record(self):
        """가장 최근 녹음된 오디오 한 조각을 가져온다."""

        return self.audio_queue.get()

    def close(self) -> None:
        """녹음 스레드를 종료한다."""

        self.stopped.set()
        self.thread.join(timeout=self.seconds + 1.0)


def write_debug_wav(audio, path: Path) -> None:
    """현재 녹음된 오디오를 확인용 WAV 파일로 저장한다."""

    from scipy.io.wavfile import write
    import numpy as np

    # float32 -1.0~1.0 범위 오디오를 일반 WAV에서 많이 쓰는 int16 PCM으로 변환한다.
    pcm = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
    write(path, SAMPLE_RATE, pcm)
