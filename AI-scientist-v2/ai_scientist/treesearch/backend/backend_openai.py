import json
import logging
import os
import time

from .utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create
from funcy import notnone, once, select_values
import openai
from openai import AzureOpenAI
from rich import print

logger = logging.getLogger("ai-scientist")

# Azure OpenAI configuration
_azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
_azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
_azure_api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")


def _use_azure():
    return bool(_azure_endpoint and _azure_api_key)


def _get_azure_model_map():
    map_str = os.environ.get("AZURE_OPENAI_MODEL_MAP", "{}")
    try:
        return json.loads(map_str)
    except json.JSONDecodeError:
        return {}


def _resolve_model(model):
    """Resolve model name for Azure deployment."""
    if _use_azure():
        model_map = _get_azure_model_map()
        return model_map.get(model, model)
    return model


OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)

def get_ai_client(model: str, max_retries=2) -> openai.OpenAI:
    if model.startswith("ollama/"):
        client = openai.OpenAI(
            base_url="http://localhost:11434/v1",
            max_retries=max_retries
        )
    elif _use_azure():
        client = AzureOpenAI(
            api_version=_azure_api_version,
            azure_endpoint=_azure_endpoint,
            api_key=_azure_api_key,
            max_retries=max_retries
        )
    else:
        client = openai.OpenAI(max_retries=max_retries)
    return client


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    client = get_ai_client(model_kwargs.get("model"), max_retries=0)
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore

    messages = opt_messages_to_list(system_message, user_message)

    if func_spec is not None:
        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        # force the model to use the function
        filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

    if filtered_kwargs.get("model", "").startswith("ollama/"):
       filtered_kwargs["model"] = filtered_kwargs["model"].replace("ollama/", "")
    else:
       filtered_kwargs["model"] = _resolve_model(filtered_kwargs.get("model", ""))

    t0 = time.time()
    completion = backoff_create(
        client.chat.completions.create,
        OPENAI_TIMEOUT_EXCEPTIONS,
        messages=messages,
        **filtered_kwargs,
    )
    req_time = time.time() - t0

    choice = completion.choices[0]

    if func_spec is None:
        output = choice.message.content
    else:
        assert (
            choice.message.tool_calls
        ), f"function_call is empty, it is not a function call: {choice.message}"
        assert (
            choice.message.tool_calls[0].function.name == func_spec.name
        ), "Function name mismatch"
        try:
            print(f"[cyan]Raw func call response: {choice}[/cyan]")
            output = json.loads(choice.message.tool_calls[0].function.arguments)
        except json.JSONDecodeError as e:
            logger.error(
                f"Error decoding the function arguments: {choice.message.tool_calls[0].function.arguments}"
            )
            raise e

    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
    }

    return output, req_time, in_tokens, out_tokens, info
