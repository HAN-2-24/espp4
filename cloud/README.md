# Helmet Cloud Voice Bridge

This bridge listens for helmet status-query MQTT messages, asks Qwen for a short Chinese status answer, converts the answer with Edge TTS to 16 kHz mono signed 16-bit PCM, and publishes PCM chunks back to the helmet.

## Topics

- Query: `helmet/voice/status/query`
- Audio reply: `helmet/voice/status/pcm/<request_id>/<seq>/<flags>`
- `flags & 1` marks the final PCM chunk.
- `flags & 2` means the payload is IMA ADPCM. The helmet decodes it to 16 kHz mono PCM before playback.

The reply payload is raw binary audio bytes, not JSON, base64, or text. The default downlink format is IMA ADPCM to keep ESP-Hosted SDIO traffic well below raw PCM size.

## Conda Environment

```powershell
conda create -n helmet-cloud python=3.11 -y
conda activate helmet-cloud
python -m pip install -r cloud\requirements.txt
conda install -c conda-forge ffmpeg -y
```

## Required API Key

Qwen/DashScope cloud API uses Alibaba Cloud Model Studio in the Chinese mainland region by default. Free quota depends on your account and current service policy; it is not guaranteed to be permanently unlimited free usage.

Recommended local file:

1. Copy `cloud\bridge_secrets.example.json` to `cloud\bridge_secrets.json`.
2. Put your real DashScope API key in `cloud\bridge_secrets.json`.

```json
{
  "DASHSCOPE_API_KEY": "your real DashScope or Qwen Cloud API key",
  "QWEN_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
  "QWEN_MODEL": "qwen-flash"
}
```

`cloud\bridge_secrets.json` is ignored by git. Environment variables still work and override the local file:

```powershell
$env:DASHSCOPE_API_KEY="your DashScope or Qwen Cloud API key"
```

If `DASHSCOPE_API_KEY` is missing from both places or Qwen fails, the bridge keeps running and uses a local rule-based summary as the answer text.

## Optional Configuration

MQTT defaults match the helmet firmware:

```powershell
$env:HELMET_MQTT_URI="mqtts://o1078f8f.ala.cn-shenzhen.emqxsl.cn:8883"
$env:HELMET_MQTT_USERNAME="ESP32P4"
$env:HELMET_MQTT_PASSWORD="4444"
$env:HELMET_MQTT_CLIENT_ID="helmet-cloud-bridge"
$env:HELMET_MQTT_TOPIC_STATUS_QUERY="helmet/voice/status/query"
$env:HELMET_MQTT_TOPIC_STATUS_PCM="helmet/voice/status/pcm"
$env:HELMET_PCM_CHUNK_BYTES="512"
$env:HELMET_PCM_PUBLISH_DELAY_MS="0"
$env:HELMET_PCM_PUBLISH_PREFILL_CHUNKS="0"
```

Qwen and Edge TTS defaults:

```powershell
$env:QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:QWEN_MODEL="qwen-flash"
$env:QWEN_TIMEOUT_SECONDS="30"
$env:EDGE_TTS_VOICE="zh-CN-XiaoxiaoNeural"
$env:EDGE_TTS_RATE="+25%"
$env:EDGE_TTS_VOLUME="+0%"
$env:EDGE_TTS_TIMEOUT_SECONDS="25"
$env:FFMPEG_BIN="ffmpeg"
$env:FFMPEG_TIMEOUT_SECONDS="15"
```

`HELMET_PCM_CHUNK_BYTES` is clamped to 128-512 bytes and forced to an even value so the device does not receive split 16-bit samples or oversized fragmented MQTT chunks.

`HELMET_PCM_PUBLISH_DELAY_MS` throttles audio downlink packets. The device now stores the reply to a PCM file before playback, so the default is `0` to reduce command-to-playback delay. Use `5` or `10` only if ESP-Hosted downlink becomes unstable.

`HELMET_PCM_PREFILL_CHUNKS` is fixed at `0` on the current SDIO path. Even small bursts can be coalesced by MQTT/TLS into multi-kilobyte Hosted reads.

`qwen-flash` is the default because helmet status answers are short and the model is cost-efficient after the free quota is used. You can set `QWEN_MODEL` to `qwen3.6-flash`, `qwen-plus`, or another compatible Model Studio chat model if you want stronger reasoning.

## Run

```powershell
conda activate helmet-cloud
python cloud\helmet_cloud_bridge.py
```

Expected startup logs:

```text
[bridge] config mqtt_uri=... query=helmet/voice/status/query pcm_prefix=helmet/voice/status/pcm ...
[bridge] config qwen_base=... qwen_model=... dashscope_key=set edge_voice=zh-CN-XiaoxiaoNeural ...
[bridge] MQTT connected ... subscribing helmet/voice/status/query
```

After saying "查询状态" on the helmet:

```text
[bridge] request=<id> received state_keys=...
[bridge] request=<id> answer=...
[bridge] request=<id> tts_start voice=zh-CN-XiaoxiaoNeural rate=+15% ...
[bridge] request=<id> tts_mp3_bytes=...
[bridge] request=<id> ffmpeg_start target_rate=16000 ...
[bridge] request=<id> ffmpeg_pcm_bytes=...
[bridge] request=<id> publish_start pcm_bytes=... chunks=... chunk_size=512 delay=0ms prefill=0
[bridge] request=<id> publish_queued first_topic=helmet/voice/status/pcm/<id>/0/0 last_topic=helmet/voice/status/pcm/<id>/<N>/1
[bridge] request=<id> pcm_bytes=<bytes> adpcm_bytes=<bytes> chunks=<count>
```

On failure, the bridge logs the error and publishes an empty final PCM packet so the device does not stay pending forever.
