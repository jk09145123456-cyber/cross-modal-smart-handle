"""Raspberry Pi 4에서 실행되는 스마트 핸들 오디오 클라이언트.

Raspberry Pi 코드의 역할:
1. 마이크에서 일정 길이의 오디오를 계속 녹음한다. (audio.py)
2. 녹음한 오디오를 PC AI 서버로 보내고 이벤트 결과를 받는다. (api.py)
3. 이벤트 이름을 Arduino로 USB Serial 전송한다. (arduino.py)
4. 위 과정을 명령어 인자에 맞춰 반복 실행한다. (cli.py)

Raspberry Pi에서 직접 AI 분석을 하지 않는 이유:
- Whisper와 YAMNet은 무거운 모델이라 Raspberry Pi에서 실시간 처리하기 어렵다.
- Raspberry Pi는 입력/출력 장치 제어에 집중하고, AI 분석은 PC 또는 Colab 서버가 담당한다.
"""
