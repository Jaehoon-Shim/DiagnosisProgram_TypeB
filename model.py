"""로컬 gpt-oss 추론: 원본 전류를 보내고 모델의 최종 답변을 그대로 돌려준다."""
from __future__ import annotations

import http.client
import json
import math
import socket
import threading
import time

CLASS_NAMES = ("정상", "A상 상단 개방", "A상 하단 개방", "B상 상단 개방",
               "B상 하단 개방", "C상 상단 개방", "C상 하단 개방")
SCHEMA = {
    "type": "object",
    "properties": {
        "class_id": {"type": ["integer", "null"], "minimum": 0, "maximum": 6},
        "reason": {"type": "string", "minLength": 1, "maxLength": 180},
        "uncertainty": {"type": "string", "minLength": 1, "maxLength": 160},
    },
    "required": ["class_id", "reason", "uncertainty"],
    "additionalProperties": False,
}
PROMPT = """Use your own pretrained electrical-engineering knowledge to diagnose the supplied three-phase currents.
Class IDs: 0 normal-like, 1 A upper switch open, 2 A lower open, 3 B upper open, 4 B lower open,
5 C upper open, 6 C lower open. The stated topology and current direction are user assumptions.
Choose a class, or null if you cannot identify the condition or switch position.
Keep your analysis concise; provide the required final JSON when enough evidence is available. Do not invent evidence to satisfy a budget.
Additional measurement/circuit context may clarify the user's assumptions; it does not replace the required output format.
Return only JSON with class_id, reason, uncertainty. Write your own final reason in Korean,
1-2 sentences <=140 characters, and uncertainty in Korean <=140 characters. No hidden chain-of-thought.
"""


class ModelError(RuntimeError):
    """연결·시간 제한·모델 출력 오류. 진단 결과로 대체하지 않는다."""


class DiagnosisCancelled(ModelError):
    """사용자가 현재 행의 추론을 취소했다."""


# 1. 입력 준비: 3×200 숫자 검사 → 원본 값 + 사용자 가정을 JSON 요청으로 구성
def _make_payload(currents, model, current_direction, *, reasoning="low", num_predict=1024,
                  num_ctx=8192, temperature=0.0, user_prompt=""):
    try:
        phases = [list(phase) for phase in currents]
        if len(phases) != 3 or any(len(phase) != 200 for phase in phases):
            raise ValueError
        values = []
        for phase in phases:
            if any(isinstance(value, bool) for value in phase):
                raise ValueError
            numbers = [float(value) for value in phase]
            if not all(math.isfinite(value) for value in numbers):
                raise ValueError
            values.append(numbers)
    except (TypeError, ValueError, OverflowError) as error:
        raise ModelError("전류는 유한한 숫자 3상 × 200개여야 합니다.") from error
    if not isinstance(model, str) or not model.strip():
        raise ModelError("설치된 Ollama 모델 이름을 입력하세요.")
    if not isinstance(current_direction, str) or not current_direction.strip():
        raise ModelError("전류의 양의 방향 또는 unknown을 지정하세요.")
    if reasoning not in ("low", "medium", "high"):
        raise ModelError("추론 수준은 low, medium, high 중 하나여야 합니다.")
    for name, value, low, high in (("num_predict", num_predict, 256, 16384), ("num_ctx", num_ctx, 8192, 65536)):
        if type(value) is not int or not low <= value <= high:
            raise ModelError(f"{name}는 {low}~{high} 범위의 정수여야 합니다.")
    try:
        sampling = float(temperature)
    except (TypeError, ValueError) as error:
        raise ModelError("temperature는 0~1 범위의 숫자여야 합니다.") from error
    if isinstance(temperature, bool) or not math.isfinite(sampling) or not 0 <= sampling <= 1:
        raise ModelError("temperature는 0~1 범위의 유한한 숫자여야 합니다.")
    if not isinstance(user_prompt, str) or len(user_prompt) > 2000:
        raise ModelError("추가 측정·회로 설명은 2000자 이하의 문자열이어야 합니다.")
    data = {
        "phase_currents": dict(zip(("ia", "ib", "ic"), values)),
        "measurement": {"samples_per_phase": 200, "current_unit": "not provided",
                        "sample_rate_hz": None},
        "assumptions": {"topology": "user-assumed two-level three-phase VSI",
                        "positive_current_direction": current_direction},
    }
    if user_prompt:
        data["additional_measurement_context"] = user_prompt
    return {"model": model.strip(), "messages": [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": json.dumps(data, ensure_ascii=False, allow_nan=False)}],
        "format": SCHEMA, "stream": True, "think": reasoning, "keep_alive": "15m",
        "options": {"temperature": sampling, "num_ctx": num_ctx, "num_predict": num_predict}}


# 2. 로컬 통신: localhost로 한 번 요청하고 NDJSON의 최종 답변 조각만 누적
def _stream(payload, cancel_event, progress, timeout):
    started = time.monotonic()
    deadline = started + timeout
    connection = http.client.HTTPConnection("127.0.0.1", 11434, timeout=min(10, timeout))
    finished = threading.Event()
    state = {"read_socket": None, "cancel_socket": None, "abort": None,
             "stage": "모델 로딩·입력 처리 중"}

    def notify():
        if progress is not None:
            try:
                progress(f"{state['stage']} · {time.monotonic() - started:.0f}초")
            except Exception:
                pass  # 화면 알림의 실패가 추론 결과를 바꾸지는 않는다.

    def interruption():
        if state["abort"] == "cancel" or cancel_event is not None and cancel_event.is_set():
            raise DiagnosisCancelled("현재 행의 진단을 취소했습니다.")
        if state["abort"] == "timeout" or time.monotonic() >= deadline:
            raise ModelError(f"행당 전체 제한 시간 {timeout:g}초를 초과했습니다. 자동 재요청하지 않았습니다.")

    # 3. 병행 감시: 로딩 중에도 전체 기한/취소를 감시하고 실제 소켓을 닫는다.
    def watchdog():
        last_notice = started
        while not finished.wait(0.05):
            reason = ("cancel" if cancel_event is not None and cancel_event.is_set()
                      else "timeout" if time.monotonic() >= deadline else None)
            if reason:
                state["abort"] = reason
                for active in (state["read_socket"], state["cancel_socket"], connection.sock):
                    if active is not None:
                        try:
                            active.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass
                original = state["read_socket"]
                if original is not None:
                    try:
                        descriptor = original.detach()
                        if descriptor != -1:
                            socket.close(descriptor)
                    except OSError:
                        pass
                if state["cancel_socket"] is not None:
                    state["cancel_socket"].close()
                return
            if time.monotonic() - last_notice >= 1:
                notify()
                last_notice = time.monotonic()

    interruption()
    notify()
    monitor = threading.Thread(target=watchdog, daemon=True)
    monitor.start()
    response, answer, total_bytes, done = None, [], 0, False
    try:
        connection.connect()
        connection.auto_open = 0  # 취소 후 HTTPConnection이 몰래 재접속하지 않게 한다.
        state["read_socket"] = connection.sock
        # Windows/HTTP1.0의 응답 파일과 별도로 취소용 핸들을 유지한다.
        state["cancel_socket"] = socket.socket.dup(connection.sock)
        interruption()
        connection.sock.settimeout(max(0.001, deadline - time.monotonic()))
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        connection.request("POST", "/api/chat", body=body,
                           headers={"Content-Type": "application/json", "Accept": "application/x-ndjson"})
        response = connection.getresponse()
        if response.status != 200:
            detail = response.read(2048).decode("utf-8", errors="replace")
            raise ModelError(f"Ollama 오류 HTTP {response.status}: {detail[:300]}")
        while True:
            interruption()
            line = response.readline(1024 * 1024 + 1)
            interruption()
            if not line:
                break
            total_bytes += len(line)
            if len(line) > 1024 * 1024 or total_bytes > 8 * 1024 * 1024:
                raise ModelError("모델 응답이 허용 크기를 넘었습니다.")
            if not line.strip():
                continue
            try:
                chunk = json.loads(line.decode("utf-8"))
            except (ValueError, UnicodeError) as error:
                raise ModelError("NDJSON 응답을 읽을 수 없습니다.") from error
            if not isinstance(chunk, dict) or chunk.get("error"):
                raise ModelError(f"모델 서버 오류: {str(chunk)[:300]}")
            message = chunk.get("message", {})
            content = message.get("content", "") if isinstance(message, dict) else None
            if not isinstance(content, str):
                raise ModelError("모델 메시지 형식이 올바르지 않습니다.")
            next_stage = state["stage"]
            if content:
                answer.append(content)
                next_stage = "최종 답변 수신 중"
            elif message.get("thinking"):
                next_stage = "모델 추론 중"  # 내부 추론 내용은 보관·표시하지 않는다.
            if next_stage != state["stage"]:
                state["stage"] = next_stage
                notify()
            if chunk.get("done") is True:
                if chunk.get("done_reason") in ("length", "max_tokens"):
                    raise ModelError("출력 길이 제한으로 답변이 잘렸습니다. "
                                     "추론·생성 한도를 늘리거나 추론 수준을 낮춰주세요. 이 행은 미진단 상태입니다.")
                done = True
                break
        interruption()
        if not done:
            raise ModelError("최종 완료(done=true) 없이 연결이 끝났습니다.")
        return "".join(answer)
    except ModelError:
        interruption()
        raise
    except (OSError, http.client.HTTPException, ValueError) as error:
        interruption()
        raise ModelError(f"로컬 Ollama 통신 실패: {error}") from error
    finally:
        finished.set()
        if response is not None:
            response.close()
        connection.close()
        if state["cancel_socket"] is not None:
            state["cancel_socket"].close()
        monitor.join(timeout=0.2)


# 4. 결과 검사: JSON 형식만 검사하고 모델의 분류·이유·불확실성을 그대로 반환
def _parse_answer(content):
    try:
        result = json.loads(content)
    except (TypeError, ValueError) as error:
        raise ModelError("모델의 최종 답변이 유효한 JSON이 아닙니다.") from error
    if not isinstance(result, dict) or set(result) != {"class_id", "reason", "uncertainty"}:
        raise ModelError("최종 JSON에 class_id, reason, uncertainty 세 항목이 필요합니다.")
    class_id = result["class_id"]
    if class_id is not None and (type(class_id) is not int or not 0 <= class_id <= 6):
        raise ModelError("class_id는 0~6 정수 또는 null이어야 합니다.")
    for key, limit in (("reason", 180), ("uncertainty", 160)):
        if not isinstance(result[key], str) or not result[key].strip() or len(result[key]) > limit:
            raise ModelError(f"{key}가 비어 있거나 허용 길이를 넘었습니다.")
    return {"class_id": class_id, "label": CLASS_NAMES[class_id] if class_id is not None else "판단 보류",
            "reason": result["reason"], "uncertainty": result["uncertainty"]}


def diagnose(currents, *, model="gpt-oss:20b", current_direction="inverter_to_load",
             cancel_event=None, progress=None, timeout=900.0, reasoning="low",
             num_predict=1024, num_ctx=8192, temperature=0.0, user_prompt=""):
    """한 행을 한 번 추론한다. 특징 규칙·결과 보정·학습·모델 설치는 수행하지 않는다."""
    if cancel_event is not None and cancel_event.is_set():
        raise DiagnosisCancelled("현재 행의 진단을 취소했습니다.")
    try:
        limit = float(timeout)
    except (TypeError, ValueError) as error:
        raise ModelError("전체 제한 시간은 유한한 양수여야 합니다.") from error
    if isinstance(timeout, bool) or not math.isfinite(limit) or limit <= 0:
        raise ModelError("전체 제한 시간은 유한한 양수여야 합니다.")
    payload = _make_payload(currents, model, current_direction, reasoning=reasoning, num_predict=num_predict,
                            num_ctx=num_ctx, temperature=temperature, user_prompt=user_prompt)
    return _parse_answer(_stream(payload, cancel_event, progress, limit))
