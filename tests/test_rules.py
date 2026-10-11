"""이벤트 판단 규칙 테스트. AI 모델 없이 실행된다."""

from smart_handle.protocol import Event
from smart_handle.server.rules import (
    detect_sound_event,
    detect_specific_sound_event,
    detect_text_event,
    merge_events,
)


def test_detect_text_event_ignores_spacing():
    assert detect_text_event("잠시 후 좌 회전 입니다") is Event.LEFT_TURN
    assert detect_text_event("300미터 앞 우회전") is Event.RIGHT_TURN


def test_speed_warning_wins_over_direction():
    assert detect_text_event("과속 주의, 왼쪽 차선") is Event.SPEED_WARNING


def test_detect_text_event_none():
    assert detect_text_event("안녕하세요") is Event.NONE


def test_detect_sound_event_picks_highest_priority():
    predictions = [("Vehicle horn, car horn, honking", 0.9), ("Siren", 0.3)]
    assert detect_sound_event(predictions, 0.1)[0] is Event.SIREN


def test_detect_sound_event_respects_threshold():
    assert detect_sound_event([("Siren", 0.05)], 0.1) == (Event.NONE, "", 0.0)


def test_detect_specific_sound_event():
    predictions = [("Screaming", 0.09), ("Speech", 0.8)]
    assert detect_specific_sound_event(predictions, Event.SCREAM, 0.08) == (
        Event.SCREAM,
        "Screaming",
        0.09,
    )


def test_merge_events_prefers_hazard():
    assert merge_events(Event.LEFT_TURN, Event.SIREN) is Event.SIREN
    assert merge_events(Event.SPEED_WARNING, Event.NONE) is Event.SPEED_WARNING
