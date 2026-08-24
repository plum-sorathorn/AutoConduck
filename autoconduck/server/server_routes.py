"""FastAPI route installation for the lazily loaded server."""

from typing import Any
from fastapi import Request

from autoconduck.server.server_chat import handle_chat_completions
from autoconduck.server.server_messages import (
    handle_messages,
    handle_messages_count_tokens,
)
from autoconduck.server.server_meta import (
    handle_healthz,
    handle_models,
    handle_stats,
)
from autoconduck.server.server_models import (
    CompletionRequest,
    MessagesRequest,
)
from autoconduck.server.server_router import (
    call_litellm,
    is_active_tool_session,
    route_target,
)


def install_routes(
    app: Any,
    _Request: Any,
    JSONResponse: Any,
    StreamingResponse: Any,
    BaseModel: Any,
    Field: Any,
    helpers: Any,
    serve_model_ids: Any,
    PSEUDO_MODELS: set[str],
    cache: dict[str, Any] | None,
) -> Any:
    """Mount API routes on the FastAPI application."""
    (
        openai_messages_from_anthropic,
        openai_tools_from_anthropic,
        openai_tool_choice_from_anthropic,
        litellm_params_for,
        count_tokens,
        AnthropicSSETranslator,
        anthropic_response_text,
        coerce_content_text,
        sanitize_tools,
        messages_litellm_kwargs,
        *extra_helpers,
    ) = helpers
    normalize_messages_for_llm = extra_helpers[0] if extra_helpers else None
    if normalize_messages_for_llm is None:
        from autoconduck.server.messages_api import normalize_messages_for_llm

    if app is None:
        from fastapi import FastAPI

        app = FastAPI(title="AutoConduck")

    decisions: list[dict[str, Any]] = []

    async def _call(model, body, path=None, pseudo=None, messages=None):
        return await call_litellm(
            model,
            body,
            path=path,
            pseudo=pseudo,
            messages=messages,
            normalize_messages_for_llm=normalize_messages_for_llm,
            sanitize_tools=sanitize_tools,
            litellm_params_for=litellm_params_for,
        )

    async def _route_target(
        body_model, messages, request=None, on_progress=None, client_type=None
    ):
        return await route_target(
            body_model,
            messages,
            request=request,
            on_progress=on_progress,
            client_type=client_type,
            decisions=decisions,
            PSEUDO_MODELS=PSEUDO_MODELS,
            litellm_params_for=litellm_params_for,
            normalize_messages_for_llm=normalize_messages_for_llm,
        )

    def healthz():
        return handle_healthz()

    async def models():
        return await handle_models(serve_model_ids)

    async def get_stats():
        return await handle_stats(decisions)

    async def completions(body: CompletionRequest, request: Request):
        return await handle_chat_completions(
            body,
            request,
            PSEUDO_MODELS=PSEUDO_MODELS,
            route_target_fn=_route_target,
            call_litellm_fn=_call,
            sanitize_tools=sanitize_tools,
            normalize_messages_for_llm=normalize_messages_for_llm,
            StreamingResponse=StreamingResponse,
            JSONResponse=JSONResponse,
        )

    async def messages_endpoint(body: MessagesRequest, request: Request):
        return await handle_messages(
            body,
            request,
            route_target_fn=_route_target,
            openai_messages_from_anthropic=openai_messages_from_anthropic,
            openai_tools_from_anthropic=openai_tools_from_anthropic,
            openai_tool_choice_from_anthropic=openai_tool_choice_from_anthropic,
            count_tokens=count_tokens,
            AnthropicSSETranslator=AnthropicSSETranslator,
            anthropic_response_text=anthropic_response_text,
            coerce_content_text=coerce_content_text,
            messages_litellm_kwargs=messages_litellm_kwargs,
            normalize_messages_for_llm=normalize_messages_for_llm,
            StreamingResponse=StreamingResponse,
            JSONResponse=JSONResponse,
        )

    async def messages_count_tokens(request: Request):
        return await handle_messages_count_tokens(
            request,
            openai_messages_from_anthropic=openai_messages_from_anthropic,
            count_tokens=count_tokens,
        )

    app.get("/healthz")(healthz)
    app.get("/v1/models")(models)
    app.get("/stats")(get_stats)
    app.post("/v1/chat/completions")(completions)
    app.post("/v1/messages")(messages_endpoint)
    app.post("/v1/messages/count_tokens")(messages_count_tokens)

    if cache is not None:
        cache.update(
            CompletionRequest=CompletionRequest,
            MessagesRequest=MessagesRequest,
            _route_target=_route_target,
            _call=_call,
            completions=completions,
            messages_endpoint=messages_endpoint,
            healthz=healthz,
            models=models,
            get_stats=get_stats,
        )
    return app
