"""Arduino 이벤트 전송 중복 방지 테스트. 실제 Serial 연결 없이 실행된다."""

from smart_handle.client.arduino import EventSender
from smart_handle.protocol import Event


def test_same_event_is_sent_once():
    sender = EventSender(serial=None, cooldown_seconds=2.0)
    assert sender.send_if_needed(Event.SIREN) is True
    assert sender.send_if_needed(Event.SIREN) is False


def test_none_resets_last_event():
    sender = EventSender(serial=None, cooldown_seconds=2.0)
    sender.send_if_needed(Event.LEFT_TURN)
    assert sender.send_if_needed(Event.NONE) is False
    assert sender.send_if_needed(Event.LEFT_TURN) is True
