import aiofiles
import asyncio
import importlib
import json
import logging
import subprocess
import sys
import traceback
from typing import Any
from colorama import Fore, Style, init
import os
from enum import Enum
from contextlib import aclosing
from gpt_researcher.utils.usage import extract_usage_report as _extract_usage_report

_SUPPORTED_PROVIDERS = {
    "openai",
    "anthropic",
    "azure_openai",
    "cohere",
    "google_vertexai",
    "google_genai",
    "fireworks",
    "ollama",
    "together",
    "mistralai",
    "huggingface",
    "groq",
    "bedrock",
    "dashscope",
    "xai",
    "deepseek",
    "litellm",
    "gigachat",
    "openrouter",
    "vllm_openai",
    "aimlapi",
    "netmind",
}

NO_SUPPORT_TEMPERATURE_MODELS = [
    "deepseek/deepseek-reasoner",
    "o1-mini",
    "o1-mini-2024-09-12",
    "o1",
    "o1-2024-12-17",
    "o3-mini",
    "o3-mini-2025-01-31",
    "o1-preview",
    "o3",
    "o3-2025-04-16",
    "o4-mini",
    "o4-mini-2025-04-16",
    # GPT-5 family: OpenAI enforces default temperature only
    "gpt-5",
    "gpt-5-mini",
]

SUPPORT_REASONING_EFFORT_MODELS = [
    "o3-mini",
    "o3-mini-2025-01-31",
    "o3",
    "o3-2025-04-16",
    "o4-mini",
    "o4-mini-2025-04-16",
]

class ReasoningEfforts(Enum):
    High = "high"
    Medium = "medium"
    Low = "low"


class ChatLogger:
    """Helper utility to log all chat requests and their corresponding responses
    plus the stack trace leading to the call.
    """

    def __init__(self, fname: str):
        self.fname = fname
        self._lock = asyncio.Lock()

    async def log_request(self, messages, response):
        async with self._lock:
            async with aiofiles.open(self.fname, mode="a", encoding="utf-8") as handle:
                await handle.write(json.dumps({
                    "messages": messages,
                    "response": response,
                    "stacktrace": traceback.format_exc()
                }) + "\n")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _build_openrouter_extra_body(tracking: Any) -> dict[str, Any]:
    if not isinstance(tracking, dict):
        return {}

    extra_body: dict[str, Any] = {
        "usage": {"include": True},
    }

    trace = _as_dict(tracking.get("trace"))
    if trace:
        extra_body["trace"] = trace

    user = tracking.get("user")
    if isinstance(user, str) and user:
        extra_body["user"] = user

    session_id = tracking.get("session_id")
    if isinstance(session_id, str) and session_id:
        extra_body["session_id"] = session_id

    return extra_body


def _merge_extra_body(existing: Any, incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing) if isinstance(existing, dict) else {}

    for key, value in incoming.items():
        if key == "trace" and isinstance(value, dict):
            merged[key] = {**_as_dict(merged.get(key)), **value}
        elif key == "usage" and isinstance(value, dict):
            merged[key] = {**_as_dict(merged.get(key)), **value}
        else:
            merged[key] = value

    return merged


class GenericLLMProvider:

    def __init__(self, llm, chat_log: str | None = None,  verbose: bool = True, model_id: str | None = None):
        self.llm = llm
        self.chat_logger = ChatLogger(chat_log) if chat_log else None
        self.verbose = verbose
        self.model_id = model_id

    @classmethod
    def from_provider(cls, provider: str, chat_log: str | None = None, verbose: bool=True, **kwargs: Any):
        from gpt_researcher.utils.budget import current_research_budget, ResearchBudgetError
        budget = current_research_budget.get()
        if budget is not None:
            if provider != "openrouter":
                if budget.mode == "enforce":
                    raise ResearchBudgetError("budget_invalid_transition")
            else:
                if "http_client" in kwargs or "http_async_client" in kwargs:
                    if budget.mode == "enforce":
                        raise ResearchBudgetError("budget_invalid_transition")
                    logging.getLogger(__name__).warning("Shadow research custom HTTP client is not budget-instrumented")
                else:
                    kwargs.update(budget.http_clients())
                # LangChain/OpenAI SDK retries would be new native requests with
                # new step numbers. Disable before any paid call in enforcement.
                if budget.mode == "enforce":
                    kwargs["max_retries"] = 0
        model_id = kwargs.get("model") or kwargs.get("model_name") or kwargs.get("model_id")

        if provider == "openai":
            _check_pkg("langchain_openai")
            from langchain_openai import ChatOpenAI

            # Support custom OpenAI-compatible APIs via OPENAI_BASE_URL
            if "openai_api_base" not in kwargs and os.environ.get("OPENAI_BASE_URL"):
                kwargs["openai_api_base"] = os.environ["OPENAI_BASE_URL"]

            llm = ChatOpenAI(**kwargs)
        elif provider == "anthropic":
            _check_pkg("langchain_anthropic")
            from langchain_anthropic import ChatAnthropic

            llm = ChatAnthropic(**kwargs)
        elif provider == "azure_openai":
            _check_pkg("langchain_openai")
            from langchain_openai import AzureChatOpenAI

            if "model" in kwargs:
                model_name = kwargs.get("model", None)
                kwargs = {"azure_deployment": model_name, **kwargs}

            llm = AzureChatOpenAI(**kwargs)
        elif provider == "cohere":
            _check_pkg("langchain_cohere")
            from langchain_cohere import ChatCohere

            llm = ChatCohere(**kwargs)
        elif provider == "google_vertexai":
            _check_pkg("langchain_google_vertexai")
            from langchain_google_vertexai import ChatVertexAI

            llm = ChatVertexAI(**kwargs)
        elif provider == "google_genai":
            _check_pkg("langchain_google_genai")
            from langchain_google_genai import ChatGoogleGenerativeAI

            llm = ChatGoogleGenerativeAI(**kwargs)
        elif provider == "fireworks":
            _check_pkg("langchain_fireworks")
            from langchain_fireworks import ChatFireworks

            llm = ChatFireworks(**kwargs)
        elif provider == "ollama":
            _check_pkg("langchain_community")
            _check_pkg("langchain_ollama")
            from langchain_ollama import ChatOllama

            llm = ChatOllama(base_url=os.environ["OLLAMA_BASE_URL"], **kwargs)
        elif provider == "together":
            _check_pkg("langchain_together")
            from langchain_together import ChatTogether

            llm = ChatTogether(**kwargs)
        elif provider == "mistralai":
            _check_pkg("langchain_mistralai")
            from langchain_mistralai import ChatMistralAI

            llm = ChatMistralAI(**kwargs)
        elif provider == "huggingface":
            _check_pkg("langchain_huggingface")
            from langchain_huggingface import ChatHuggingFace

            if "model" in kwargs or "model_name" in kwargs:
                model_id = kwargs.pop("model", None) or kwargs.pop("model_name", None)
                kwargs = {"model_id": model_id, **kwargs}
            llm = ChatHuggingFace(**kwargs)
        elif provider == "groq":
            _check_pkg("langchain_groq")
            from langchain_groq import ChatGroq

            llm = ChatGroq(**kwargs)
        elif provider == "bedrock":
            _check_pkg("langchain_aws")
            from langchain_aws import ChatBedrock

            if "model" in kwargs or "model_name" in kwargs:
                model_id = kwargs.pop("model", None) or kwargs.pop("model_name", None)
                kwargs = {"model_id": model_id, "model_kwargs": kwargs}
            llm = ChatBedrock(**kwargs)
        elif provider == "dashscope":
            _check_pkg("langchain_openai")
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(openai_api_base='https://dashscope.aliyuncs.com/compatible-mode/v1',
                     openai_api_key=os.environ["DASHSCOPE_API_KEY"],
                     **kwargs
                )
        elif provider == "xai":
            _check_pkg("langchain_xai")
            from langchain_xai import ChatXAI

            llm = ChatXAI(**kwargs)
        elif provider == "deepseek":
            _check_pkg("langchain_openai")
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(openai_api_base='https://api.deepseek.com',
                     openai_api_key=os.environ["DEEPSEEK_API_KEY"],
                     **kwargs
                )
        elif provider == "litellm":
            _check_pkg("langchain_community")
            from langchain_community.chat_models.litellm import ChatLiteLLM

            llm = ChatLiteLLM(**kwargs)
        elif provider == "gigachat":
            _check_pkg("langchain_gigachat")
            from langchain_gigachat.chat_models import GigaChat

            kwargs.pop("model", None) # Use env GIGACHAT_MODEL=GigaChat-Max
            llm = GigaChat(**kwargs)
        elif provider == "openrouter":
            _check_pkg("langchain_openai")
            from langchain_openai import ChatOpenAI
            from langchain_core.rate_limiters import InMemoryRateLimiter

            tracking = kwargs.pop("nevel_tracking", None)
            extra_body = _build_openrouter_extra_body(tracking)
            if extra_body:
                kwargs["extra_body"] = _merge_extra_body(
                    kwargs.pop("extra_body", None),
                    extra_body,
                )
                kwargs.setdefault("stream_usage", True)

            rps = float(os.environ["OPENROUTER_LIMIT_RPS"]) if "OPENROUTER_LIMIT_RPS" in os.environ else 1.0

            rate_limiter = InMemoryRateLimiter(
                requests_per_second=rps,
                check_every_n_seconds=0.1,
                max_bucket_size=10,
            )

            llm = ChatOpenAI(openai_api_base='https://openrouter.ai/api/v1',
                     openai_api_key=os.environ["OPENROUTER_API_KEY"],
                     rate_limiter=rate_limiter,
                     **kwargs
                )
        elif provider == "vllm_openai":
            _check_pkg("langchain_openai")
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                openai_api_key=os.environ["VLLM_OPENAI_API_KEY"],
                openai_api_base=os.environ["VLLM_OPENAI_API_BASE"],
                **kwargs
            )
        elif provider == "aimlapi":
            _check_pkg("langchain_openai")
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(openai_api_base='https://api.aimlapi.com/v1',
                             openai_api_key=os.environ["AIMLAPI_API_KEY"],
                             **kwargs
                             )
        elif provider == 'netmind':
            _check_pkg("langchain_netmind")
            from langchain_netmind import ChatNetmind

            llm = ChatNetmind(**kwargs)
        else:
            supported = ", ".join(_SUPPORTED_PROVIDERS)
            raise ValueError(
                f"Unsupported {provider}.\n\nSupported model providers are: {supported}"
            )
        return cls(llm, chat_log, verbose=verbose, model_id=model_id)


    async def get_chat_response(self, messages, stream, websocket=None, cost_callback=None, **kwargs):
        if not stream:
            # Getting output from the model chain using ainvoke for asynchronous invoking
            output = await self.llm.ainvoke(messages, **kwargs)
            usage_report = _extract_usage_report(output, self.model_id)
            if usage_report and cost_callback:
                cost_callback(usage_report)

            res = output.content

        else:
            res = await self.stream_response(messages, websocket, cost_callback=cost_callback, **kwargs)

        if self.chat_logger:
            await self.chat_logger.log_request(messages, res)

        return res

    async def stream_response(self, messages, websocket=None, cost_callback=None, **kwargs):
        paragraph = ""
        response = ""
        last_usage_report = None

        # Usage chunks describe cumulative snapshots, not additive deltas. Keep
        # the latest evidence and publish it once, including on cancellation or
        # a downstream WebSocket/file failure after the provider charged us.
        try:
            # Close upstream promptly when the consumer disappears; don't leave
            # a paid stream running until async-generator garbage collection.
            async with aclosing(self.llm.astream(messages, **kwargs)) as stream:
                async for chunk in stream:
                    usage_report = _extract_usage_report(chunk, self.model_id)
                    if usage_report:
                        last_usage_report = usage_report

                    content = chunk.content
                    if content is not None:
                        response += content
                        paragraph += content
                        if "\n" in paragraph:
                            await self._send_output(paragraph, websocket)
                            paragraph = ""

            if paragraph:
                await self._send_output(paragraph, websocket)
        except BaseException:
            if last_usage_report and cost_callback:
                try:
                    cost_callback(last_usage_report)
                except Exception:
                    # Preserve the original failure/cancellation. Never include
                    # request payloads or provider credentials in this log.
                    logging.getLogger(__name__).error("Research usage callback failed during stream cleanup")
            raise
        else:
            if last_usage_report and cost_callback:
                cost_callback(last_usage_report)

        return response

    async def _send_output(self, content, websocket=None):
        if websocket is not None:
            await websocket.send_json({"type": "report", "output": content})
        elif self.verbose:
            print(f"{Fore.GREEN}{content}{Style.RESET_ALL}")


def _check_pkg(pkg: str) -> None:
    if not importlib.util.find_spec(pkg):
        pkg_kebab = pkg.replace("_", "-")
        # Import colorama and initialize it
        init(autoreset=True)

        try:
            print(f"{Fore.YELLOW}Installing {pkg_kebab}...{Style.RESET_ALL}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", pkg_kebab])
            print(f"{Fore.GREEN}Successfully installed {pkg_kebab}{Style.RESET_ALL}")

            # Try importing again after install
            importlib.import_module(pkg)

        except subprocess.CalledProcessError:
            raise ImportError(
                Fore.RED + f"Failed to install {pkg_kebab}. Please install manually with "
                f"`pip install -U {pkg_kebab}`"
            )
