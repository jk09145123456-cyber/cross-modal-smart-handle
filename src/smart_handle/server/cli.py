"""스마트 핸들 시스템의 PC용 HTTP AI 분석 서버."""

from __future__ import annotations

# 실행 옵션을 명령어 인자로 받기 위해 사용한다.
import argparse

# 서버 실행 기록과 오류를 출력하기 위해 사용한다.
import logging

# API 키를 환경 변수에서 읽기 위해 사용한다.
import os

# 별도 웹 프레임워크 없이 간단한 HTTP 서버를 만들기 위해 사용한다.
from http.server import ThreadingHTTPServer

# HTTP 요청 처리는 api.py, AI 분석은 service.py에 나눠 두었다.
from smart_handle.server.api import make_handler
from smart_handle.server.service import AiService


def parse_args() -> argparse.Namespace:
    """서버 실행에 필요한 설정값을 명령어 인자로 받는다."""

    parser = argparse.ArgumentParser(description=__doc__)

    # 네트워크 접속 설정이다. 0.0.0.0은 같은 네트워크의 Raspberry Pi 접속을 허용한다.
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)

    # Whisper 모델 설정이다. Colab GPU에서는 medium/cuda/float16 조합을 사용할 수 있다.
    parser.add_argument("--whisper-model", default="medium")
    parser.add_argument("--whisper-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--whisper-compute-type", default="int8")

    # 감지 민감도와 분석 창 길이 설정이다.
    parser.add_argument("--sound-threshold", type=float, default=0.10)
    parser.add_argument("--speech-threshold", type=float, default=0.10)
    parser.add_argument("--scream-threshold", type=float, default=0.08)
    parser.add_argument("--speech-window-seconds", type=float, default=2.0)
    parser.add_argument("--sound-window-seconds", type=float, default=1.0)
    parser.add_argument("--hazard-hold-seconds", type=float, default=5.0)
    parser.add_argument("--voice-rms-threshold", type=float, default=0.005)

    # 공개 네트워크나 ngrok을 사용할 때 임의 접근을 줄이기 위한 선택 설정이다.
    parser.add_argument("--api-key", default=os.environ.get("SMART_HANDLE_API_KEY"))
    return parser.parse_args()


def main() -> None:
    """AI 서비스를 만들고 HTTP 서버를 계속 실행한다."""

    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    service = AiService(
        args.whisper_model,
        args.whisper_device,
        args.whisper_compute_type,
        args.sound_threshold,
        args.speech_threshold,
        args.speech_window_seconds,
        args.scream_threshold,
        args.sound_window_seconds,
        args.hazard_hold_seconds,
        args.voice_rms_threshold,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service, args.api_key))
    logging.info("AI server listening on http://%s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Stopping")
    finally:
        server.server_close()
