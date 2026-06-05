#!/usr/bin/env python3
"""MQTT bridge for helmet status-query voice replies.

Device request payload:
  {
    "type": "helmet_status_query",
    "request_id": 1,
    "audio": {"format": "pcm_s16le", "sample_rate": 16000, "channels": 1},
    "state": {...}
  }

Device response topics:
  <pcm_prefix>/<request_id>/<seq>/<flags>

Payload is raw signed 16-bit little-endian mono PCM. flags bit 0 marks final.
"""

from __future__ import annotations

import json
import math
import os
import ssl
import struct
import sys
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import paho.mqtt.client as mqtt
from openai import OpenAI


DEFAULT_MQTT_URI = "mqtts://o1078f8f.ala.cn-shenzhen.emqxsl.cn:8883"
PCM_FINAL_FLAG = 0x01


@dataclass
class BridgeConfig:
    mqtt_uri: str
    mqtt_username: str | None
    mqtt_password: str | None
    mqtt_client_id: str
    query_topic: str
    pcm_topic_prefix: str
    mqtt_qos: int
    pcm_chunk_bytes: int
    llm_model: str
    tts_model: str
    tts_voice: str
    tts_source_rate: int


def env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def load_config() -> BridgeConfig:
    return BridgeConfig(
        mqtt_uri=env("HELMET_MQTT_URI", DEFAULT_MQTT_URI) or DEFAULT_MQTT_URI,
        mqtt_username=env("HELMET_MQTT_USERNAME", "ESP32P4"),
        mqtt_password=env("HELMET_MQTT_PASSWORD", "4444"),
        mqtt_client_id=env("HELMET_MQTT_CLIENT_ID", "helmet-cloud-bridge") or "helmet-cloud-bridge",
        query_topic=env("HELMET_MQTT_TOPIC_STATUS_QUERY", "helmet/voice/status/query")
        or "helmet/voice/status/query",
        pcm_topic_prefix=env("HELMET_MQTT_TOPIC_STATUS_PCM", "helmet/voice/status/pcm")
        or "helmet/voice/status/pcm",
        mqtt_qos=int(env("HELMET_MQTT_QOS", "0") or "0"),
        pcm_chunk_bytes=int(env("HELMET_PCM_CHUNK_BYTES", "512") or "512"),
        llm_model=env("OPENAI_LLM_MODEL", "gpt-4o-mini") or "gpt-4o-mini",
        tts_model=env("OPENAI_TTS_MODEL", "gpt-4o-mini-tts") or "gpt-4o-mini-tts",
        tts_voice=env("OPENAI_TTS_VOICE", "alloy") or "alloy",
        tts_source_rate=int(env("OPENAI_TTS_PCM_SAMPLE_RATE", "24000") or "24000"),
    )


def mqtt_client(config: BridgeConfig) -> tuple[mqtt.Client, str, int]:
    parsed = urlparse(config.mqtt_uri)
    scheme = parsed.scheme or "mqtts"
    host = parsed.hostname
    if not host:
        raise ValueError(f"invalid MQTT URI: {config.mqtt_uri}")

    port = parsed.port or (8883 if scheme == "mqtts" else 1883)

    try:
        client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=config.mqtt_client_id,
            protocol=mqtt.MQTTv311,
        )
    except (AttributeError, TypeError):
        client = mqtt.Client(client_id=config.mqtt_client_id, protocol=mqtt.MQTTv311)

    if config.mqtt_username:
        client.username_pw_set(config.mqtt_username, config.mqtt_password)

    if scheme == "mqtts":
        client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
        client.tls_insecure_set(False)

    return client, host, port


def risk_sentence(state: dict[str, Any]) -> str:
    risk = str(state.get("risk", "UNKNOWN"))
    alarm = bool(state.get("alarm_active", False))
    eye = state.get("eye") or {}
    pose = state.get("pose") or {}
    location = state.get("location") or {}

    parts: list[str] = []
    if alarm:
        parts.append("当前报警已触发")
    if risk == "DANGER":
        parts.append("风险等级危险")
    elif risk == "FATIGUE":
        parts.append("检测到疲劳风险")
    elif risk == "ATTENTION":
        parts.append("需要注意当前状态")
    else:
        parts.append("当前风险正常")

    if eye:
        parts.append(f"PERCLOS {float(eye.get('perclos', 0.0)):.2f}")
        if eye.get("yawn_detected"):
            parts.append("检测到打哈欠")

    if pose:
        parts.append(
            f"姿态 roll {float(pose.get('roll', 0.0)):.1f} 度 pitch {float(pose.get('pitch', 0.0)):.1f} 度"
        )
        if pose.get("fall_detected"):
            parts.append("检测到跌倒")
        if pose.get("impact_detected"):
            parts.append("检测到冲击")

    if location and location.get("valid"):
        parts.append(f"速度 {float(location.get('speed', 0.0)):.1f}")

    return "，".join(parts) + "。"


def build_prompt(state: dict[str, Any]) -> str:
    compact = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
    return (
        "请根据下面安全头盔实时数据，用中文给佩戴者播报状态。"
        "要求：一句到三句话，先说风险和是否需要处理；不要读 JSON；不要使用项目符号；"
        "如果有危险、跌倒、冲击、疲劳或报警，直接给行动建议。"
        f"\n\n当前数据：{compact}"
    )


def generate_status_text(openai_client: OpenAI, config: BridgeConfig, state: dict[str, Any]) -> str:
    prompt = build_prompt(state)
    system = (
        "你是智能安全头盔的语音助手。回答要短、明确、适合直接语音播报，"
        "单位和数值可以保留一位小数。"
    )

    try:
        response = openai_client.responses.create(
            model=config.llm_model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_output_tokens=180,
        )
        text = getattr(response, "output_text", "")
    except Exception as exc:
        print(f"[bridge] LLM failed, using local summary: {exc}", file=sys.stderr)
        text = ""

    text = " ".join(str(text).strip().split())
    return text or risk_sentence(state)


def synthesize_pcm(openai_client: OpenAI, config: BridgeConfig, text: str) -> bytes:
    try:
        speech = openai_client.audio.speech.create(
            model=config.tts_model,
            voice=config.tts_voice,
            input=text,
            response_format="pcm",
        )
    except TypeError:
        speech = openai_client.audio.speech.create(
            model=config.tts_model,
            voice=config.tts_voice,
            input=text,
        )

    if hasattr(speech, "read"):
        return speech.read()
    if hasattr(speech, "content"):
        return speech.content
    return bytes(speech)


def resample_pcm_s16le(pcm: bytes, source_rate: int, target_rate: int) -> bytes:
    if not pcm or source_rate == target_rate:
        return pcm

    try:
        import audioop  # type: ignore

        converted, _ = audioop.ratecv(pcm, 2, 1, source_rate, target_rate, None)
        return converted
    except Exception:
        pass

    sample_count = len(pcm) // 2
    if sample_count <= 1:
        return pcm

    samples = struct.unpack("<" + "h" * sample_count, pcm[: sample_count * 2])
    out_count = max(1, int(math.ceil(sample_count * target_rate / source_rate)))
    out = bytearray(out_count * 2)

    for i in range(out_count):
        src_pos = i * source_rate / target_rate
        left = int(src_pos)
        right = min(left + 1, sample_count - 1)
        frac = src_pos - left
        value = int(round(samples[left] * (1.0 - frac) + samples[right] * frac))
        struct.pack_into("<h", out, i * 2, max(-32768, min(32767, value)))

    return bytes(out)


def publish_pcm(
    client: mqtt.Client,
    config: BridgeConfig,
    request_id: int,
    pcm: bytes,
) -> None:
    chunk_size = max(128, config.pcm_chunk_bytes)

    if not pcm:
        topic = f"{config.pcm_topic_prefix}/{request_id}/0/{PCM_FINAL_FLAG}"
        client.publish(topic, b"", qos=config.mqtt_qos, retain=False).wait_for_publish()
        return

    seq = 0
    total = len(pcm)
    for offset in range(0, total, chunk_size):
        chunk = pcm[offset : offset + chunk_size]
        final = PCM_FINAL_FLAG if offset + chunk_size >= total else 0
        topic = f"{config.pcm_topic_prefix}/{request_id}/{seq}/{final}"
        info = client.publish(topic, chunk, qos=config.mqtt_qos, retain=False)
        info.wait_for_publish()
        seq += 1


def handle_status_query(
    client: mqtt.Client,
    openai_client: OpenAI,
    config: BridgeConfig,
    payload: bytes,
) -> None:
    try:
        message = json.loads(payload.decode("utf-8"))
    except Exception as exc:
        print(f"[bridge] bad JSON payload: {exc}", file=sys.stderr)
        return

    if message.get("type") != "helmet_status_query":
        print(f"[bridge] ignore message type={message.get('type')!r}")
        return

    request_id = int(message.get("request_id") or int(time.time()))
    audio = message.get("audio") or {}
    state = message.get("state") or {}

    target_rate = int(audio.get("sample_rate") or 16000)
    channels = int(audio.get("channels") or 1)
    fmt = str(audio.get("format") or "pcm_s16le")

    if fmt != "pcm_s16le" or channels != 1:
        print(f"[bridge] unsupported audio request format={fmt} channels={channels}", file=sys.stderr)
        return

    text = generate_status_text(openai_client, config, state)
    print(f"[bridge] request={request_id} answer={text}")

    pcm = synthesize_pcm(openai_client, config, text)
    pcm = resample_pcm_s16le(pcm, config.tts_source_rate, target_rate)
    publish_pcm(client, config, request_id, pcm)
    print(f"[bridge] request={request_id} pcm_bytes={len(pcm)}")


def main() -> int:
    if not env("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is required", file=sys.stderr)
        return 2

    config = load_config()
    openai_client = OpenAI()
    client, host, port = mqtt_client(config)

    def on_connect(client_: mqtt.Client, userdata: Any, flags: Any, reason_code: Any, properties: Any = None) -> None:
        print(f"[bridge] MQTT connected reason={reason_code}; subscribing {config.query_topic}")
        client_.subscribe(config.query_topic, qos=config.mqtt_qos)

    def on_message(client_: mqtt.Client, userdata: Any, message: mqtt.MQTTMessage) -> None:
        handle_status_query(client_, openai_client, config, message.payload)

    client.on_connect = on_connect
    client.on_message = on_message

    print(f"[bridge] connecting MQTT {config.mqtt_uri} client_id={config.mqtt_client_id}")
    client.connect(host, port, keepalive=60)
    client.loop_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
