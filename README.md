# Smart Handle

An embedded **smart steering handle** that listens for navigation voice prompts and nearby hazard sounds, then **alerts the driver through vibration and LEDs**.
A Raspberry Pi captures microphone audio, a PC AI server analyzes it with Whisper and YAMNet, and an Arduino drives the handle's vibration motors and NeoPixel LEDs.

## Contents

- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Hardware](#hardware)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Events and Alert Patterns](#events-and-alert-patterns)
- [API](#api)
- [Tech Stack](#tech-stack)

## Features

- **Voice prompt recognition**: Transcribes Korean navigation prompts with Whisper (faster-whisper) and detects left turn, right turn, and speeding warnings.
- **Hazard sound detection**: Classifies sirens, car horns, and screams with YAMNet.
- **Priority-based decisions**: When several events are detected at once, the more dangerous one wins.
- **Haptic and visual feedback**: Plays a distinct pattern per event on left/right vibration motors and a 16-LED NeoPixel ring.
- **Non-blocking control**: The Arduino firmware uses `millis()` instead of `delay()`, so a higher-priority alert can interrupt immediately.

## Architecture

```text
┌──────────────┐   HTTP POST /analyze    ┌──────────────────────┐
│ Raspberry Pi │ ──────────────────────▶ │     PC AI Server     │
│  Microphone  │   float32 PCM, 16 kHz   │  Whisper  +  YAMNet  │
│              │ ◀────────────────────── │  Priority merge      │
└──────┬───────┘      JSON (event)       └──────────────────────┘
       │ USB Serial (115200 baud)
       ▼
┌──────────────┐
│   Arduino    │  2x vibration motors, NeoPixel LED ring
└──────────────┘
```

1. The Raspberry Pi records audio from the microphone in 1-second chunks and resamples it to 16 kHz.
2. It sends the audio to the PC AI server's `/analyze` endpoint.
3. The server classifies ambient sound with YAMNet and, when human speech is likely, transcribes it with Whisper.
4. The higher-priority of the text event and the sound event is returned as the final event.
5. The Raspberry Pi sends the final event name to the Arduino over serial, and the Arduino plays the alert pattern.

## Project Structure

```text
.
├── firmware/
│   └── smart_handle/
│       └── smart_handle.ino     # Arduino firmware for vibration motors and NeoPixel
├── src/
│   └── smart_handle/
│       ├── protocol.py          # Event definitions and sample rate (shared by client and server)
│       ├── client/              # Raspberry Pi client
│       │   ├── cli.py           # Entry point and main loop
│       │   ├── audio.py         # Microphone recording and resampling
│       │   ├── api.py           # HTTP client for the AI server
│       │   └── arduino.py       # Arduino serial output and duplicate suppression
│       └── server/              # PC AI server
│           ├── cli.py           # Entry point and server options
│           ├── api.py           # HTTP request handling (/health, /analyze)
│           ├── service.py       # Audio buffers, hazard hold, analysis flow
│           ├── models.py        # Whisper and YAMNet model wrappers
│           └── rules.py         # Keyword- and priority-based event decisions
├── tests/                       # pytest unit tests (no models or devices needed)
├── pyproject.toml               # Package metadata, dependencies, commands
└── README.md
```

## Hardware

| Component | Description |
| --- | --- |
| Raspberry Pi 4 | USB microphone input, server communication, Arduino connection |
| USB microphone | Captures voice prompts and ambient sound |
| Arduino (e.g. Uno) | Controls vibration motors and LEDs |
| 2x vibration motor | Left on D4, right on D2 |
| NeoPixel ring (16 LEDs) | Data pin D6 |
| PC (CUDA GPU recommended) | Runs the AI analysis server |

## Getting Started

### Requirements

- Python 3.10 or later
- PC server: NVIDIA GPU with CUDA recommended (CPU also works)
- Arduino IDE and the [Adafruit NeoPixel](https://github.com/adafruit/Adafruit_NeoPixel) library

### Installation

```bash
git clone https://github.com/jk09145123456-cyber/embedded.git
cd embedded
```

**PC AI server**

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[server]"
```

**Raspberry Pi**

```bash
sudo apt install libportaudio2
python -m venv .venv
source .venv/bin/activate
pip install -e ".[client]"
```

**Arduino**

1. Open `firmware/smart_handle/smart_handle.ino` in the Arduino IDE.
2. Install `Adafruit NeoPixel` from the Library Manager.
3. Select your board and port, then upload.

## Usage

After installation, run the `smart-handle-server` and `smart-handle-client` commands (equivalent to `python -m smart_handle.server` and `python -m smart_handle.client`).

### 1. Start the PC AI server

```bash
smart-handle-server
```

Without CUDA, run on the CPU:

```bash
smart-handle-server --whisper-device cpu --whisper-model small
```

Health check:

```bash
curl http://localhost:8000/health
```

### 2. Start the Raspberry Pi client

```bash
smart-handle-client --server-url http://<PC_IP>:8000 --serial-port /dev/ttyACM0
```

To check things without the hardware:

```bash
smart-handle-client --self-test           # Check data formats
smart-handle-client --no-serial           # Log events without an Arduino
smart-handle-client --debug-wav debug.wav # Save the recorded audio
```

### Tests

```bash
pip install -e ".[dev]"
pytest
```

### Authentication (optional)

Set the same API key on the server and the client to authenticate requests with the `X-API-Key` header.

```bash
export SMART_HANDLE_API_KEY=your-secret-key
```

### Common options

| Target | Option | Default | Description |
| --- | --- | --- | --- |
| Server | `--host` / `--port` | `0.0.0.0` / `8000` | Bind address |
| Server | `--whisper-model` | `medium` | Whisper model size |
| Server | `--whisper-device` | `cuda` | `cpu` or `cuda` |
| Server | `--sound-threshold` | `0.10` | Hazard sound score threshold |
| Server | `--hazard-hold-seconds` | `5.0` | How long a hazard event is held |
| Client | `--seconds` | `1.0` | Length of each recording |
| Client | `--device` | Default input device | Microphone index or name |
| Client | `--server-url` | `http://127.0.0.1:8000` | AI server address |
| Client | `--serial-port` | `/dev/ttyACM0` | Arduino port |
| Client | `--cooldown` | `2.0` | Interval before resending the same event |

Run with `--help` to see every option.

## Events and Alert Patterns

A higher priority means a more dangerous event. A higher-priority alert interrupts a lower-priority alert that is still playing.

| Event | Detected by | Priority | LED | Vibration |
| --- | --- | --- | --- | --- |
| `SCREAM` | YAMNet (scream, shout, yell) | 6 | Red | Both sides, fast repeat |
| `SIREN` | YAMNet (siren, emergency vehicle) | 5 | Blue | Both sides, long pulse |
| `HORN` | YAMNet (car horn, honk) | 4 | Yellow | Both sides, short repeat |
| `SPEED_WARNING` | Whisper (과속, 속도위반, 제한속도 — "speeding", "speed limit") | 3 | Orange | Both sides, slow repeat |
| `LEFT_TURN` | Whisper (좌회전, 왼쪽, 좌측 — "left turn", "left") | 2 | Off | Left motor for 5 s |
| `RIGHT_TURN` | Whisper (우회전, 오른쪽, 우측 — "right turn", "right") | 1 | Off | Right motor for 5 s |
| `NONE` | Nothing detected | 0 | Off | Off |

## API

### `GET /health`

```json
{ "status": "ok" }
```

### `POST /analyze`

| Header | Value |
| --- | --- |
| `Content-Type` | `application/octet-stream` |
| `X-Sample-Rate` | `16000` (required) |
| `X-API-Key` | API key (if configured) |

The request body is 16 kHz mono little-endian float32 PCM, up to 30 seconds.

Example response:

```json
{
  "event": "SIREN",
  "text": "",
  "stt_event": "NONE",
  "sound_event": "SIREN",
  "sound_label": "Siren",
  "sound_score": 0.42,
  "speech_score": 0.01,
  "voice_rms": 0.003,
  "top_predictions": [
    { "label": "Siren", "score": 0.42 },
    { "label": "Vehicle", "score": 0.12 }
  ]
}
```

## Tech Stack

- **Raspberry Pi**: Python, sounddevice, SciPy, pySerial
- **AI server**: Python, faster-whisper, TensorFlow Hub (YAMNet), `http.server`
- **Arduino**: C++, Adafruit NeoPixel
