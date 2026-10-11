"""Raspberry Pi 4에서 실행되는 스마트 핸들 오디오 클라이언트."""

from __future__ import annotations

# 실행 옵션을 명령어 인자로 받기 위해 사용한다.
import argparse

# 서버 응답의 JSON 형식 오류를 처리하기 위해 사용한다.
import json

# 실행 상태와 오류를 확인하기 위해 사용한다.
import logging

# API 키를 환경 변수에서 읽기 위해 사용한다.
import os

# 반복 지연에 사용한다.
import time

# PC AI 서버로 HTTP 요청을 보낼 때 발생하는 오류를 처리하기 위해 사용한다.
import urllib.error

# 디버그 WAV 파일 저장 경로를 다루기 위해 사용한다.
from pathlib import Path

# 서버와 같은 이벤트 이름, 같은 샘플레이트를 사용한다.
from smart_handle.protocol import SAMPLE_RATE, Event

# 녹음, 서버 통신, Arduino 전송은 각각 별도 모듈에 나눠 두었다.
from smart_handle.client.api import NetworkAiClient
from smart_handle.client.arduino import ArduinoSerial, EventSender
from smart_handle.client.audio import AudioRecorder, write_debug_wav


def run_self_test() -> None:
    """장치 없이도 기본 데이터 형식이 맞는지 간단히 확인한다."""

    import numpy as np

    # 2초짜리 무음 오디오를 만들고 float32 바이트 길이가 예상과 맞는지 확인한다.
    audio = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
    assert len(audio.tobytes()) == SAMPLE_RATE * 2 * 4

    # 서버에서 받은 문자열을 Event enum으로 바꿀 수 있는지 확인한다.
    assert Event("SIREN") is Event.SIREN
    print("Raspberry Pi client self-test passed.")


def parse_device(value: str) -> int | str:
    """마이크 장치 번호는 정수로, 장치 이름은 문자열로 처리한다."""

    return int(value) if value.isdigit() else value


def parse_args() -> argparse.Namespace:
    """Raspberry Pi 클라이언트 실행에 필요한 설정값을 받는다."""

    parser = argparse.ArgumentParser(description=__doc__)

    # 녹음 관련 설정이다.
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--device", type=parse_device)
    parser.add_argument(
        "--input-sample-rate",
        type=int,
        help="Microphone recording rate; defaults to the selected device rate",
    )

    # PC AI 서버 접속 설정이다.
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--server-timeout", type=float, default=60.0)
    parser.add_argument("--api-key", default=os.environ.get("SMART_HANDLE_API_KEY"))

    # Arduino Serial 연결 설정이다.
    parser.add_argument("--serial-port", default="/dev/ttyACM0")
    parser.add_argument("--baudrate", type=int, default=115200)

    # 반복 실행과 디버깅 관련 설정이다.
    parser.add_argument("--cooldown", type=float, default=2.0)
    parser.add_argument("--loop-delay", type=float, default=0.1)
    parser.add_argument("--no-serial", action="store_true")
    parser.add_argument("--debug-wav", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    """녹음, AI 서버 분석, 아두이노 전송을 반복 실행한다."""

    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.self_test:
        run_self_test()
        return

    # 녹음기, AI 서버 클라이언트, Arduino 연결을 각각 초기화한다.
    recorder = AudioRecorder(args.seconds, args.device, args.input_sample_rate)
    ai_client = NetworkAiClient(args.server_url, args.server_timeout, args.api_key)
    serial_connection = None if args.no_serial else ArduinoSerial(args.serial_port, args.baudrate)
    sender = EventSender(serial_connection, args.cooldown)

    logging.info("Raspberry Pi client started. AI server: %s", args.server_url)
    try:
        while True:
            # 1. 마이크에서 오디오를 가져온다.
            audio = recorder.record()

            # --debug-wav를 지정하면 현재 전송 중인 오디오를 파일로 저장해 마이크 입력을 확인할 수 있다.
            if args.debug_wav:
                write_debug_wav(audio, args.debug_wav)

            try:
                # 2. PC AI 서버에 오디오를 보내 최종 이벤트를 받는다.
                final_event, result = ai_client.analyze(audio)
                logging.info(
                    "text=%r sound=%s(%.3f) final=%s top=%s",
                    result.get("text", ""),
                    result.get("sound_label") or "none",
                    result.get("sound_score", 0.0),
                    final_event.value,
                    result.get("top_predictions", []),
                )

                # 3. 최종 이벤트를 아두이노로 보내 LED와 진동 모터를 제어한다.
                sender.send_if_needed(final_event)
            except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as error:
                # 네트워크 오류나 서버 응답 형식 오류가 있어도 프로그램을 바로 종료하지 않고 다음 녹음을 계속 시도한다.
                logging.error("AI server request failed: %s", error)
            time.sleep(args.loop_delay)
    except KeyboardInterrupt:
        logging.info("Stopping")
    finally:
        recorder.close()
        if serial_connection:
            serial_connection.close()
