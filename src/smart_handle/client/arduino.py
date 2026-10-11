"""Arduino USB Serial 연결과 이벤트 전송 관리."""

from __future__ import annotations

# 전송 결과를 로그로 확인하기 위해 사용한다.
import logging

# 시리얼 연결 안정화와 전송 시간 기록에 사용한다.
import time

# 이벤트 전송 상태를 간단한 클래스로 만들기 위해 사용한다.
from dataclasses import dataclass

# 서버와 같은 이벤트 이름을 사용한다.
from smart_handle.protocol import Event


class ArduinoSerial:
    """USB 시리얼로 아두이노에 이벤트 명령을 보낸다."""

    def __init__(self, port: str, baudrate: int) -> None:
        # pyserial은 Raspberry Pi와 Arduino의 USB 시리얼 통신에 필요하다.
        import serial

        # timeout/write_timeout을 지정해서 연결 문제가 생겼을 때 무한 대기하지 않도록 한다.
        self.serial = serial.Serial(port, baudrate, timeout=1, write_timeout=1)

        # Arduino는 USB Serial 연결 직후 자동 리셋될 수 있어 잠시 기다린다.
        time.sleep(2.0)

        # 시작 직후 남아 있을 수 있는 입력 버퍼를 비운다.
        self.serial.reset_input_buffer()

    def send(self, event: Event) -> None:
        """아두이노가 읽을 수 있도록 이벤트 이름을 한 줄 문자열로 보낸다."""

        # Arduino 코드는 readStringUntil('\n')으로 한 줄씩 읽으므로 줄바꿈을 붙인다.
        self.serial.write(f"{event.value}\n".encode("ascii"))
        self.serial.flush()

    def close(self) -> None:
        """시리얼 포트를 닫는다."""

        self.serial.close()


@dataclass
class EventSender:
    """같은 이벤트가 계속 반복 전송되지 않도록 관리한다."""

    serial: ArduinoSerial | None
    cooldown_seconds: float
    last_event: Event = Event.NONE
    last_sent_at: float = 0.0

    def send_if_needed(self, event: Event) -> bool:
        """새 이벤트가 있을 때만 아두이노로 전송한다."""

        # NONE은 알림이 없는 상태이므로 Arduino에 보내지 않는다.
        # 대신 last_event를 초기화해 다음 실제 이벤트가 들어오면 다시 보낼 수 있게 한다.
        if event is Event.NONE:
            self.last_event = Event.NONE
            return False

        # 같은 이벤트가 계속 감지되면 Arduino에 반복 전송하지 않는다.
        # Arduino 쪽에서도 패턴을 유지하고 있으므로 같은 명령을 계속 보낼 필요가 없다.
        if event == self.last_event:
            return False

        # --no-serial 옵션이면 실제 Arduino 없이 로그로만 확인한다.
        if self.serial:
            self.serial.send(event)
            logging.info("Sent Arduino command: %s", event.value)
        else:
            logging.info("Serial disabled; command would be: %s", event.value)
        self.last_event = event
        self.last_sent_at = time.monotonic()
        return True
