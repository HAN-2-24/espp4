# Helmet Cloud Voice Bridge

This bridge listens for helmet status-query MQTT messages, asks an LLM for a short Chinese status answer, converts the answer to 16 kHz mono signed 16-bit PCM, and publishes PCM chunks back to the helmet.

## Topics

- Query: `helmet/voice/status/query`
- PCM reply: `helmet/voice/status/pcm/<request_id>/<seq>/<flags>`
- `flags & 1` marks the final PCM chunk.

## Environment

Required:

- `OPENAI_API_KEY`

Optional:

- `HELMET_MQTT_URI` default `mqtts://o1078f8f.ala.cn-shenzhen.emqxsl.cn:8883`
- `HELMET_MQTT_USERNAME` default `ESP32P4`
- `HELMET_MQTT_PASSWORD` default `4444`
- `HELMET_MQTT_CLIENT_ID` default `helmet-cloud-bridge`
- `HELMET_MQTT_TOPIC_STATUS_QUERY` default `helmet/voice/status/query`
- `HELMET_MQTT_TOPIC_STATUS_PCM` default `helmet/voice/status/pcm`
- `OPENAI_LLM_MODEL` default `gpt-4o-mini`
- `OPENAI_TTS_MODEL` default `gpt-4o-mini-tts`
- `OPENAI_TTS_VOICE` default `alloy`

## Run

```powershell
python -m venv .venv-cloud
.\.venv-cloud\Scripts\pip install -r cloud\requirements.txt
$env:OPENAI_API_KEY="sk-..."
.\.venv-cloud\Scripts\python cloud\helmet_cloud_bridge.py
```
