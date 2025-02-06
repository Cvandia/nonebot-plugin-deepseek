import httpx
import json
from nonebot.log import logger

from ..config import config
from ..compat import model_dump

# from ..function_call import registry
from ..exception import RequestException
from ..schemas import Balance, ChatCompletions, Message, Choice, Usage


class API:
    _headers = {
        "Accept": "application/json",
    }

    @classmethod
    async def chat(cls, message: list[dict[str, str]], model: str = "deepseek-chat") -> ChatCompletions:
        """普通对话"""
        model_config = config.get_model_config(model)
        is_stream = config.is_stream

        api_key = model_config.api_key or config.api_key
        prompt = model_dump(model_config, exclude_none=True).get("prompt", config.prompt)

        _json = {
            "messages": [{"content": prompt, "role": "system"}] + message if prompt else message,
            "model": model,
            "stream": is_stream,
            **model_config.to_dict(),
        }
        logger.debug(f"使用模型 {model}，配置：{_json}")

        # if model == "deepseek-chat":
        #     json.update({"tools": registry.to_json()})
        async with httpx.AsyncClient() as client:
            if is_stream:
                async with client.stream(
                    "POST",
                    f"{model_config.base_url}/chat/completions",
                    headers={
                        **cls._headers,
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=_json,
                    timeout=50,
                ) as response:
                    full_content = ""
                    response_data = ChatCompletions(
                        id="",
                        object="chat.completion",
                        model=model,
                        created=0,
                        choices=[
                            Choice(
                                finish_reason=None,
                                index=0,
                                message=Message(role="assistant", content=""),
                                logprobs=None,
                            )
                        ],
                        usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                    )
                    async for line in response.aiter_lines():
                        if line.strip() and line.startswith("data: "):
                            line = line.lstrip("data: ")
                            if line == "[DONE]":
                                break
                            try:
                                chunk = json.loads(line)
                                # 更新response_data
                                response_data.id = response_data.id or chunk.get("id", "")
                                response_data.created = response_data.created or chunk.get("created", 0)

                                if chunk.get("choices"):
                                    delta = chunk["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        full_content += delta["content"]
                                    response_data.choices[0].message.content = full_content
                                    response_data.choices[0].finish_reason = chunk["choices"][0].get(
                                        "finish_reason", response_data.choices[0].finish_reason
                                    )

                                if chunk.get("usage"):
                                    response_data.usage = Usage(**chunk["usage"])

                            except json.JSONDecodeError:
                                continue

                logger.debug(f"完整响应: {response_data}")
                return response_data
            else:
                response = await client.post(
                    f"{model_config.base_url}/chat/completions",
                    headers={
                        **cls._headers,
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=_json,
                    timeout=50,
                )
                if erro := response.json().get("error"):
                    raise RequestException(erro)
                return ChatCompletions(**response.json())

    @classmethod
    async def query_balance(cls, model_name: str) -> Balance:
        model_config = config.get_model_config(model_name)
        api_key = model_config.api_key or config.api_key

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{model_config.base_url}/user/balance",
                headers={**cls._headers, "Authorization": f"Bearer {api_key}"},
            )
        if response.status_code == 404:
            raise RequestException("本地模型不支持查询余额，请更换默认模型")
        return Balance(**response.json())
