"""Raspberry Pi 요청을 처리하는 HTTP 핸들러."""

from __future__ import annotations

# Raspberry Pi와 JSON 형식으로 분석 결과를 주고받기 위해 사용한다.
import json

# 서버 실행 기록과 오류를 출력하기 위해 사용한다.
import logging

# HTTP 요청 핸들러의 기본 클래스이다.
from http.server import BaseHTTPRequestHandler

# Raspberry Pi 코드와 같은 샘플레이트를 사용한다.
from smart_handle.protocol import SAMPLE_RATE

# 실제 오디오 분석은 service.py가 담당한다.
from smart_handle.server.service import AiService


def make_handler(service: AiService, api_key: str | None):
    """HTTP 요청을 처리하는 Handler 클래스를 생성한다."""

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status: int, body: dict) -> None:
            """응답 딕셔너리를 JSON으로 변환해서 클라이언트에 보낸다."""

            payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            """서버 상태 확인용 /health 요청을 처리한다."""

            self.send_json(200, {"status": "ok"}) if self.path == "/health" else self.send_json(
                404, {"error": "not found"}
            )

        def do_POST(self) -> None:
            """Raspberry Pi가 보낸 오디오를 /analyze에서 분석한다."""

            # 이 서버는 오디오 분석 API만 제공하므로 /analyze 외 POST 요청은 거부한다.
            if self.path != "/analyze":
                self.send_json(404, {"error": "not found"})
                return

            # API 키를 설정한 경우 Raspberry Pi 요청에도 같은 키가 들어 있어야 한다.
            if api_key and self.headers.get("X-API-Key") != api_key:
                self.send_json(401, {"error": "invalid API key"})
                return

            # 서버와 클라이언트가 같은 샘플레이트를 사용해야 모델 입력 길이가 정확해진다.
            if self.headers.get("X-Sample-Rate") != str(SAMPLE_RATE):
                self.send_json(400, {"error": f"sample rate must be {SAMPLE_RATE}"})
                return
            try:
                # float32 오디오 바이트만 받도록 길이와 배수를 검사한다.
                # float32는 샘플 하나가 4바이트이므로 전체 길이가 4의 배수여야 한다.
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > SAMPLE_RATE * 4 * 30 or length % 4 != 0:
                    raise ValueError("audio payload must be between 0 and 30 seconds")
                import numpy as np

                # Little-endian float32("<f4")로 들어온 바이트를 numpy 배열로 해석한다.
                result = service.analyze(np.frombuffer(self.rfile.read(length), dtype="<f4"))
                self.send_json(200, result)
            except Exception:
                logging.exception("Analysis failed")
                self.send_json(500, {"error": "analysis failed"})

        def log_message(self, message: str, *args) -> None:
            """기본 HTTP 로그도 logging 형식으로 맞춰 출력한다."""

            logging.info("%s - %s", self.client_address[0], message % args)

    return Handler
