"""Translate the project's bounded requests across supported OpenAI wire APIs."""
from __future__ import annotations

from copy import deepcopy
import base64
import re
from urllib.parse import urlparse


def _input_content(content):
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    if not isinstance(content, list):
        raise ValueError("unsupported_message_content")
    result=[]
    for part in content:
        if not isinstance(part, dict):
            raise ValueError("unsupported_message_content")
        if part.get("type")=="text" and isinstance(part.get("text"),str):
            result.append({"type":"input_text","text":part["text"]})
        elif part.get("type")=="image_url":
            image=part.get("image_url")
            url=image.get("url") if isinstance(image,dict) else image
            if not isinstance(url,str):raise ValueError("unsupported_image_content")
            result.append({"type":"input_image","image_url":url})
        else:
            raise ValueError("unsupported_message_content")
    return result


def prepare_request(settings, chat_payload):
    """Return ``(endpoint, payload)`` without adding credentials."""
    if settings.wire_api=="chat_completions":
        return settings.base_url.rstrip("/")+"/chat/completions",deepcopy(chat_payload)
    if settings.wire_api=="anthropic_messages":
        messages=chat_payload.get("messages")
        if not isinstance(messages,list):raise ValueError("missing_messages")
        system=[];rows=[]
        for message in messages:
            if not isinstance(message,dict) or message.get("role") not in {"system","user","assistant"}:
                raise ValueError("unsupported_message")
            content=message.get("content")
            if message["role"]=="system":
                if not isinstance(content,str):raise ValueError("unsupported_system_message")
                system.append(content);continue
            if isinstance(content,str):
                converted=content
            elif isinstance(content,list):
                converted=[]
                for part in content:
                    if not isinstance(part,dict):raise ValueError("unsupported_message_content")
                    if part.get("type")=="text" and isinstance(part.get("text"),str):
                        converted.append({"type":"text","text":part["text"]})
                    elif part.get("type")=="image_url":
                        image=part.get("image_url");uri=image.get("url") if isinstance(image,dict) else image
                        match=re.fullmatch(r"data:(image/(?:png|jpeg|webp));base64,([A-Za-z0-9+/=]+)",uri or "")
                        if not match:raise ValueError("unsupported_image_content")
                        try:base64.b64decode(match.group(2),validate=True)
                        except ValueError:raise ValueError("unsupported_image_content") from None
                        converted.append({"type":"image","source":{"type":"base64","media_type":match.group(1),"data":match.group(2)}})
                    else:raise ValueError("unsupported_message_content")
            else:raise ValueError("unsupported_message_content")
            rows.append({"role":message["role"],"content":converted})
        payload={"model":chat_payload["model"],"messages":rows,
                 "max_tokens":chat_payload.get("max_tokens",1000)}
        if system:payload["system"]="\n\n".join(system)
        if "temperature" in chat_payload:payload["temperature"]=chat_payload["temperature"]
        return settings.base_url.rstrip("/")+"/v1/messages",payload
    if settings.wire_api!="responses":
        raise ValueError("unsupported_wire_api")
    messages=chat_payload.get("messages")
    if not isinstance(messages,list):raise ValueError("missing_messages")
    instructions=[];input_rows=[]
    for message in messages:
        if not isinstance(message,dict) or message.get("role") not in {"system","user","assistant"}:
            raise ValueError("unsupported_message")
        if message["role"]=="system":
            if not isinstance(message.get("content"),str):raise ValueError("unsupported_system_message")
            instructions.append(message["content"])
        else:
            input_rows.append({"role":message["role"],"content":_input_content(message.get("content"))})
    payload={"model":chat_payload["model"],"input":input_rows,
             "max_output_tokens":chat_payload.get("max_tokens",1000),
             "store":not settings.disable_response_storage}
    if instructions:payload["instructions"]="\n\n".join(instructions)
    return settings.base_url.rstrip("/")+"/responses",payload


def extract_text(settings, response):
    """Return model text, audit source and a normalized finish reason."""
    body=response.json()
    if settings.wire_api=="chat_completions":
        choice=body["choices"][0]
        content=choice["message"]["content"]
        if not isinstance(content,str):raise ValueError("invalid_text_content")
        return content,"message.content",choice.get("finish_reason"),body.get("usage") or {}
    if settings.wire_api=="anthropic_messages":
        texts=[part["text"] for part in body.get("content") or []
               if isinstance(part,dict) and part.get("type")=="text" and isinstance(part.get("text"),str)]
        stop_reason=body.get("stop_reason")
        normalized={"end_turn":"stop","stop_sequence":"stop","max_tokens":"length","tool_use":"tool_calls"}.get(stop_reason,stop_reason)
        # Reasoning-capable Anthropic-compatible servers can exhaust the output
        # budget in ``thinking`` blocks before emitting a text block. Preserve
        # the normalized stop reason so callers can report ``truncated_output``
        # instead of the misleading ``invalid_envelope``.
        if not texts:
            if normalized not in (None,"stop"):
                return "","anthropic.content.text",normalized,body.get("usage") or {}
            raise ValueError("missing_output_text")
        return "".join(texts),"anthropic.content.text",normalized,body.get("usage") or {}
    if body.get("status")!="completed":
        return "","responses.output",body.get("status") or "incomplete",body.get("usage") or {}
    texts=[]
    for item in body.get("output") or []:
        if not isinstance(item,dict) or item.get("type")!="message" or item.get("role")!="assistant":continue
        for part in item.get("content") or []:
            if isinstance(part,dict) and part.get("type")=="output_text" and isinstance(part.get("text"),str):
                texts.append(part["text"])
    if not texts:raise ValueError("missing_output_text")
    return "".join(texts),"responses.output_text","stop",body.get("usage") or {}


def request_headers(settings) -> dict:
    if settings.auth_scheme == "x-api-key":
        return {"x-api-key": settings.api_key, "anthropic-version": "2023-06-01",
                "content-type": "application/json"}
    headers={"Authorization": "Bearer " + settings.api_key}
    if settings.wire_api == "anthropic_messages":headers["anthropic-version"]="2023-06-01"
    return headers


def endpoint_allowed(settings) -> bool:
    parsed=urlparse(settings.base_url)
    schemes={"https","http"} if settings.allow_insecure_http else {"https"}
    return bool(parsed.scheme in schemes and parsed.hostname and not parsed.username and not parsed.password
                and not parsed.query and not parsed.fragment)
