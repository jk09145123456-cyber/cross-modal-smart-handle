"""스마트 핸들 시스템의 PC용 HTTP AI 분석 서버.

PC 서버의 역할:
1. Raspberry Pi에서 HTTP POST로 보낸 float32 오디오 데이터를 받는다. (api.py)
2. Whisper로 한국어 안내 음성을 인식한다. (models.py)
3. YAMNet으로 사이렌, 경적, 비명 같은 주변 위험 소리를 분류한다. (models.py)
4. 두 결과를 우선순위 기준으로 합쳐 최종 이벤트를 만든다. (rules.py, service.py)
5. 위 구성 요소를 명령어 인자에 맞춰 실행한다. (cli.py)

Raspberry Pi가 직접 AI 모델을 돌리지 않는 이유:
- Whisper와 TensorFlow/YAMNet은 연산량이 커서 Raspberry Pi에서 실행하면 느릴 수 있다.
- PC 또는 Colab GPU에서 AI 분석을 맡기면 Raspberry Pi는 녹음과 Arduino 통신에 집중할 수 있다.
"""
