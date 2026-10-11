"""PC AI 서버와 HTTP로 통신하는 클라이언트."""

from __future__ import annotations

# PC AI 서버와 JSON 형식으로 결과를 주고받기 위해 사용한다.
import json

# 별도 HTTP 라이브러리 없이 서버에 POST 요청을 보내기 위해 사용한다.
import urllib.request

# 서버와 같은 이벤트 이름, 같은 샘플레이트를 사용한다.
from smart_handle.protocol import SAMPLE_RATE, Event


class NetworkAiClient:
    """녹음된 오디오를 PC AI 서버로 보내고 분석 결과를 받는다."""

    def __init__(self, server_url: str, timeout: float, api_key: str | None) -> None:
        self.analyze_url = f"{server_url.rstrip('/')}/analyze"
        self.timeout = timeout
        self.api_key = api_key

    def analyze(self, audio) -> tuple[Event, dict]:
        """float32 오디오 배열을 HTTP POST로 전송하고 이벤트 결과를 반환한다."""

        import numpy as np

        # 서버는 little-endian float32 바이트 배열을 받도록 작성되어 있다.
        payload = np.asarray(audio, dtype="<f4").tobytes()

        # Content-Type은 일반 바이너리 오디오임을 표시한다.
        # X-Sample-Rate는 서버가 입력 샘플레이트를 검증하는 데 사용한다.
        headers = {
            "Content-Type": "application/octet-stream",
            "X-Sample-Rate": str(SAMPLE_RATE),
        }

        # API 키를 사용할 때는 HTTP 헤더로 함께 보낸다.
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        # 표준 라이브러리 urllib로 POST 요청을 만든다.
        request = urllib.request.Request(
            self.analyze_url, data=payload, headers=headers, method="POST"
        )

        # 서버 응답은 JSON 문자열이므로 dict로 변환한다.
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))

        # result["event"]는 문자열이므로 Event enum으로 바꿔 이후 코드에서 안전하게 비교한다.
        return Event(result["event"]), result
