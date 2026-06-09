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

flags bit 0 marks final. flags bit 1 means payload is IMA ADPCM decoded by the helmet to 16 kHz mono PCM.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import ssl
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import edge_tts
import paho.mqtt.client as mqtt
import requests


DEFAULT_MQTT_URI = "mqtts://o1078f8f.ala.cn-shenzhen.emqxsl.cn:8883"
DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_SECRETS_FILE = Path(__file__).with_name("bridge_secrets.json")
PCM_FINAL_FLAG = 0x01
PCM_ADPCM_FLAG = 0x02
PCM_MAX_CHUNK_BYTES = 512
PCM_MIN_CHUNK_BYTES = 128


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
    pcm_publish_delay_ms: int
    pcm_publish_prefill_chunks: int
    qwen_base_url: str
    qwen_model: str
    qwen_api_key: str | None
    qwen_timeout: float
    edge_tts_voice: str
    edge_tts_rate: str
    edge_tts_volume: str
    edge_tts_timeout: float
    ffmpeg_bin: str
    ffmpeg_timeout: float


def env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def load_local_secrets() -> dict[str, str]:
    path_value = env("HELMET_BRIDGE_SECRETS_FILE", str(DEFAULT_SECRETS_FILE))
    path = Path(path_value or DEFAULT_SECRETS_FILE)

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[bridge] local secrets ignored: {path} {exc}", file=sys.stderr)
        return {}

    if not isinstance(raw, dict):
        print(f"[bridge] local secrets ignored: {path} must contain a JSON object", file=sys.stderr)
        return {}

    return {str(key): str(value) for key, value in raw.items() if value not in (None, "")}


def env_int(name: str, default: int) -> int:
    value = env(name)
    if value is None:
        return default

    try:
        return int(value)
    except ValueError:
        print(f"[bridge] invalid integer {name}={value!r}, using {default}", file=sys.stderr)
        return default


def env_float(name: str, default: float) -> float:
    value = env(name)
    if value is None:
        return default

    try:
        return float(value)
    except ValueError:
        print(f"[bridge] invalid float {name}={value!r}, using {default}", file=sys.stderr)
        return default


def normalize_pcm_chunk_bytes(value: int) -> int:
    chunk_size = max(PCM_MIN_CHUNK_BYTES, min(PCM_MAX_CHUNK_BYTES, value))
    if chunk_size & 1:
        chunk_size -= 1
    return chunk_size


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def load_config() -> BridgeConfig:
    secrets = load_local_secrets()

    def config_value(name: str, default: str | None = None) -> str | None:
        return env(name, secrets.get(name, default))

    def config_int(name: str, default: int) -> int:
        value = config_value(name)
        if value is None:
            return default

        try:
            return int(value)
        except ValueError:
            print(f"[bridge] invalid integer {name}={value!r}, using {default}", file=sys.stderr)
            return default

    def config_float(name: str, default: float) -> float:
        value = config_value(name)
        if value is None:
            return default

        try:
            return float(value)
        except ValueError:
            print(f"[bridge] invalid float {name}={value!r}, using {default}", file=sys.stderr)
            return default

    return BridgeConfig(
        mqtt_uri=config_value("HELMET_MQTT_URI", DEFAULT_MQTT_URI) or DEFAULT_MQTT_URI,
        mqtt_username=config_value("HELMET_MQTT_USERNAME", "ESP32P4"),
        mqtt_password=config_value("HELMET_MQTT_PASSWORD", "4444"),
        mqtt_client_id=config_value("HELMET_MQTT_CLIENT_ID", "helmet-cloud-bridge") or "helmet-cloud-bridge",
        query_topic=config_value("HELMET_MQTT_TOPIC_STATUS_QUERY", "helmet/voice/status/query")
        or "helmet/voice/status/query",
        pcm_topic_prefix=config_value("HELMET_MQTT_TOPIC_STATUS_PCM", "helmet/voice/status/pcm")
        or "helmet/voice/status/pcm",
        mqtt_qos=0,
        pcm_chunk_bytes=normalize_pcm_chunk_bytes(config_int("HELMET_PCM_CHUNK_BYTES", 512)),
        pcm_publish_delay_ms=max(0, config_int("HELMET_PCM_PUBLISH_DELAY_MS", 0)),
        pcm_publish_prefill_chunks=max(0, config_int("HELMET_PCM_PUBLISH_PREFILL_CHUNKS", 0)),
        qwen_base_url=config_value("QWEN_BASE_URL", DEFAULT_QWEN_BASE_URL) or DEFAULT_QWEN_BASE_URL,
        qwen_model=config_value("QWEN_MODEL", "qwen-flash") or "qwen-flash",
        qwen_api_key=config_value("DASHSCOPE_API_KEY"),
        qwen_timeout=config_float("QWEN_TIMEOUT_SECONDS", 30.0),
        edge_tts_voice=config_value("EDGE_TTS_VOICE", "zh-CN-XiaoxiaoNeural") or "zh-CN-XiaoxiaoNeural",
        edge_tts_rate=config_value("EDGE_TTS_RATE", "+25%") or "+25%",
        edge_tts_volume=config_value("EDGE_TTS_VOLUME", "+0%") or "+0%",
        edge_tts_timeout=config_float("EDGE_TTS_TIMEOUT_SECONDS", 25.0),
        ffmpeg_bin=config_value("FFMPEG_BIN", "ffmpeg") or "ffmpeg",
        ffmpeg_timeout=config_float("FFMPEG_TIMEOUT_SECONDS", 15.0),
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
        "不要输出思考过程。正常状态要非常短；如果有危险、跌倒、冲击、疲劳或报警，"
        "必须说明原因并直接给行动建议。"
        f"\n\n当前数据：{compact}"
    )


def clean_qwen_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return " ".join(text.strip().split())


def status_is_normal(state: dict[str, Any]) -> bool:
    pose = as_dict(state.get("pose"))
    eye = as_dict(state.get("eye"))
    return (
        str(state.get("risk") or "NORMAL").upper() == "NORMAL"
        and int(state.get("warning") or 0) == 0
        and not bool(state.get("alarm_active"))
        and not bool(state.get("alarm_suppressed"))
        and not bool(pose.get("fall_detected"))
        and not bool(pose.get("impact_detected"))
        and not bool(eye.get("yawn_detected"))
    )


def compact_status_answer(state: dict[str, Any], text: str) -> str:
    pose = as_dict(state.get("pose"))
    risk = str(state.get("risk") or "UNKNOWN").upper()
    normal = status_is_normal(state)
    danger = (
        risk == "DANGER"
        or bool(state.get("alarm_active"))
        or bool(pose.get("fall_detected"))
        or bool(pose.get("impact_detected"))
    )
    limit = 50 if normal else (90 if danger else 60)

    sentences = re.findall(r"[^。！？!?]+[。！？!?]?", text)
    compact = "".join(sentences[:2]).strip()
    if len(compact) > limit:
        compact = compact[:limit].rstrip("，,；;、 ") + "。"
    if compact:
        return compact
    return "当前状态正常，请继续保持专注。" if normal else text


def generate_status_text(config: BridgeConfig, state: dict[str, Any]) -> tuple[str, str]:
    if not config.qwen_api_key:
        print("[bridge] DASHSCOPE_API_KEY is not set, using local summary", file=sys.stderr)
        return compact_status_answer(state, risk_sentence(state)), "local_no_key"

    prompt = build_prompt(state)
    system = (
        "你是智能安全头盔的语音助手。回答要短、明确、适合直接语音播报。"
        "正常状态说一句完整状态，不要只说四个字；可以结合姿态、疲劳、定位等数据简短说明。"
        "异常状态必须包含风险、原因和行动建议。"
        "单位和数值可以保留一位小数。"
    )
    url = f"{config.qwen_base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": config.qwen_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 180,
        "temperature": 0.4,
    }
    headers = {
        "Authorization": f"Bearer {config.qwen_api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=config.qwen_timeout)
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        source = "qwen"
    except Exception as exc:
        print(f"[bridge] Qwen API failed, using local summary: {exc}", file=sys.stderr)
        text = ""
        source = "local_qwen_failed"

    cleaned = clean_qwen_text(str(text)) or risk_sentence(state)
    return compact_status_answer(state, cleaned), source


async def edge_tts_save(text: str, config: BridgeConfig, output_path: str) -> None:
    communicate = edge_tts.Communicate(
        text=text,
        voice=config.edge_tts_voice,
        rate=config.edge_tts_rate,
        volume=config.edge_tts_volume,
    )
    await communicate.save(output_path)


def synthesize_pcm(config: BridgeConfig, request_id: int, text: str, target_rate: int) -> bytes:
    ffmpeg_bin = shutil.which(config.ffmpeg_bin) or config.ffmpeg_bin
    tmp_path = ""

    try:
        with tempfile.NamedTemporaryFile(prefix="helmet_tts_", suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        print(
            f"[bridge] request={request_id} tts_start "
            f"voice={config.edge_tts_voice} rate={config.edge_tts_rate} "
            f"chars={len(text)} timeout={config.edge_tts_timeout:g}s"
        )
        try:
            asyncio.run(asyncio.wait_for(edge_tts_save(text, config, tmp_path), timeout=config.edge_tts_timeout))
        except TimeoutError as exc:
            raise RuntimeError(f"Edge TTS timed out after {config.edge_tts_timeout:g}s") from exc

        mp3_bytes = Path(tmp_path).stat().st_size
        print(f"[bridge] request={request_id} tts_mp3_bytes={mp3_bytes}")

        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            tmp_path,
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            str(target_rate),
            "pipe:1",
        ]
        print(
            f"[bridge] request={request_id} ffmpeg_start "
            f"target_rate={target_rate} timeout={config.ffmpeg_timeout:g}s"
        )
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=config.ffmpeg_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"ffmpeg timed out after {config.ffmpeg_timeout:g}s") from exc
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg failed rc={proc.returncode}: {stderr}")

        print(f"[bridge] request={request_id} ffmpeg_pcm_bytes={len(proc.stdout)}")
        return proc.stdout
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def encode_ima_adpcm(pcm: bytes) -> bytes:
    index_table = (
        -1, -1, -1, -1, 2, 4, 6, 8,
        -1, -1, -1, -1, 2, 4, 6, 8,
    )
    step_table = (
        7, 8, 9, 10, 11, 12, 13, 14, 16, 17,
        19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
        50, 55, 60, 66, 73, 80, 88, 97, 107, 118,
        130, 143, 157, 173, 190, 209, 230, 253, 279, 307,
        337, 371, 408, 449, 494, 544, 598, 658, 724, 796,
        876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066,
        2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358,
        5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899,
        15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767,
    )

    sample_count = len(pcm) // 2
    if sample_count < 2:
        return b""
    if sample_count & 1:
        sample_count -= 1

    predictor = 0
    index = 0
    pending_nibble: int | None = None
    encoded = bytearray(sample_count // 2)

    def encode_sample(sample: int) -> int:
        nonlocal predictor, index

        step = step_table[index]
        diff = sample - predictor
        nibble = 0
        if diff < 0:
            nibble = 8
            diff = -diff

        temp_step = step
        if diff >= temp_step:
            nibble |= 4
            diff -= temp_step
        temp_step >>= 1
        if diff >= temp_step:
            nibble |= 2
            diff -= temp_step
        temp_step >>= 1
        if diff >= temp_step:
            nibble |= 1

        delta = step >> 3
        if nibble & 1:
            delta += step >> 2
        if nibble & 2:
            delta += step >> 1
        if nibble & 4:
            delta += step

        if nibble & 8:
            predictor -= delta
        else:
            predictor += delta
        predictor = max(-32768, min(32767, predictor))

        index += index_table[nibble & 0x0F]
        index = max(0, min(88, index))

        return nibble & 0x0F

    out_pos = 0
    for i in range(sample_count):
        sample = int.from_bytes(pcm[i * 2 : i * 2 + 2], byteorder="little", signed=True)
        nibble = encode_sample(sample)
        if pending_nibble is None:
            pending_nibble = nibble
        else:
            encoded[out_pos] = pending_nibble | (nibble << 4)
            out_pos += 1
            pending_nibble = None

    return bytes(encoded[:out_pos])


def publish_pcm(
    client: mqtt.Client,
    config: BridgeConfig,
    request_id: int,
    pcm: bytes,
    extra_flags: int = 0,
) -> int:
    chunk_size = config.pcm_chunk_bytes

    if not pcm:
        topic = f"{config.pcm_topic_prefix}/{request_id}/0/{PCM_FINAL_FLAG | extra_flags}"
        info = client.publish(topic, b"", qos=config.mqtt_qos, retain=False)
        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            raise RuntimeError(f"MQTT publish failed rc={info.rc} topic={topic}")
        info.wait_for_publish()
        print(f"[bridge] request={request_id} publish_empty_final topic={topic}")
        return 1

    seq = 0
    total = len(pcm)
    chunk_count = (total + chunk_size - 1) // chunk_size
    first_topic = ""
    last_topic = ""

    print(
        f"[bridge] request={request_id} publish_start "
        f"pcm_bytes={total} chunks={chunk_count} chunk_size={chunk_size} "
        f"delay={config.pcm_publish_delay_ms}ms prefill={config.pcm_publish_prefill_chunks}"
    )

    delay_s = config.pcm_publish_delay_ms / 1000.0
    for offset in range(0, total, chunk_size):
        chunk = pcm[offset : offset + chunk_size]
        if len(chunk) & 1:
            chunk = chunk[:-1]
        final = PCM_FINAL_FLAG if offset + chunk_size >= total else 0
        topic = f"{config.pcm_topic_prefix}/{request_id}/{seq}/{final | extra_flags}"
        if not first_topic:
            first_topic = topic
        last_topic = topic
        info = client.publish(topic, chunk, qos=config.mqtt_qos, retain=False)
        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            raise RuntimeError(f"MQTT publish failed rc={info.rc} topic={topic}")
        info.wait_for_publish()
        seq += 1
        if (
            delay_s > 0
            and offset + chunk_size < total
            and seq >= config.pcm_publish_prefill_chunks
        ):
            time.sleep(delay_s)

    print(f"[bridge] request={request_id} publish_queued first_topic={first_topic} last_topic={last_topic}")
    return seq


def handle_status_query(
    client: mqtt.Client,
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

    try:
        request_id = int(message.get("request_id") or int(time.time()))
    except (TypeError, ValueError):
        request_id = int(time.time())

    audio = as_dict(message.get("audio"))
    state = as_dict(message.get("state"))
    print(f"[bridge] request={request_id} received state_keys={','.join(sorted(state.keys()))}")

    try:
        target_rate = int(audio.get("sample_rate") or 16000)
        channels = int(audio.get("channels") or 1)
    except (TypeError, ValueError):
        target_rate = 16000
        channels = 1

    fmt = str(audio.get("format") or "pcm_s16le")

    if fmt != "pcm_s16le" or channels != 1:
        print(f"[bridge] unsupported audio request format={fmt} channels={channels}", file=sys.stderr)
        publish_pcm(client, config, request_id, b"")
        return

    try:
        text, answer_source = generate_status_text(config, state)
        print(f"[bridge] request={request_id} answer_source={answer_source} answer={text}")

        pcm = synthesize_pcm(config, request_id, text, target_rate)
        adpcm = encode_ima_adpcm(pcm)
        chunks = publish_pcm(client, config, request_id, adpcm, extra_flags=PCM_ADPCM_FLAG)
        print(
            f"[bridge] request={request_id} "
            f"pcm_bytes={len(pcm)} adpcm_bytes={len(adpcm)} chunks={chunks}"
        )
    except Exception as exc:
        print(f"[bridge] request={request_id} failed, sending empty final: {exc}", file=sys.stderr)
        try:
            publish_pcm(client, config, request_id, b"")
        except Exception as publish_exc:
            print(f"[bridge] request={request_id} empty final publish failed: {publish_exc}", file=sys.stderr)


def log_config(config: BridgeConfig) -> None:
    key_state = "set" if config.qwen_api_key else "missing"
    print(
        "[bridge] config "
        f"mqtt_uri={config.mqtt_uri} "
        f"query={config.query_topic} "
        f"pcm_prefix={config.pcm_topic_prefix} "
        f"qos={config.mqtt_qos} "
        f"chunk={config.pcm_chunk_bytes} "
        f"publish_delay={config.pcm_publish_delay_ms}ms "
        f"prefill={config.pcm_publish_prefill_chunks}"
    )
    print(
        "[bridge] config "
        f"qwen_base={config.qwen_base_url} "
        f"qwen_model={config.qwen_model} "
        f"dashscope_key={key_state} "
        f"edge_voice={config.edge_tts_voice} "
        f"edge_rate={config.edge_tts_rate} "
        f"edge_timeout={config.edge_tts_timeout:g}s "
        f"ffmpeg={config.ffmpeg_bin} "
        f"ffmpeg_timeout={config.ffmpeg_timeout:g}s"
    )


def main() -> int:
    config = load_config()
    log_config(config)

    if not config.qwen_api_key:
        print("[bridge] warning: DASHSCOPE_API_KEY is missing; Qwen replies will use local summary", file=sys.stderr)

    client, host, port = mqtt_client(config)

    def on_connect(client_: mqtt.Client, userdata: Any, flags: Any, reason_code: Any, properties: Any = None) -> None:
        print(f"[bridge] MQTT connected reason={reason_code}; subscribing {config.query_topic}")
        client_.subscribe(config.query_topic, qos=config.mqtt_qos)

    def on_message(client_: mqtt.Client, userdata: Any, message: mqtt.MQTTMessage) -> None:
        payload = bytes(message.payload)
        worker = threading.Thread(
            target=handle_status_query,
            args=(client_, config, payload),
            name="helmet-status-query",
            daemon=True,
        )
        worker.start()

    client.on_connect = on_connect
    client.on_message = on_message

    print(f"[bridge] connecting MQTT {config.mqtt_uri} client_id={config.mqtt_client_id}")
    client.connect(host, port, keepalive=60)
    client.loop_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
