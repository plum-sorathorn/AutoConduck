"""Hermetic connectivity matrix helpers and fail-soft async probes."""
from __future__ import annotations

import asyncio
import time
from typing import Any


def build_health_matrix(
    providers: list[dict[str, Any]],
    results: dict[str, dict[str, Any]] | None = None,
    local_ports: list[int] | None = None,
) -> dict[str, Any]:
    """Build a displayable health matrix without reading config or doing I/O."""
    results = results or {}
    provider_rows = []
    for provider in providers:
        name = str(provider.get("name") or provider.get("provider") or "provider")
        credential = provider.get("api_key") or provider.get("token")
        row = results.get(name, {})
        provider_rows.append({
            "name": name,
            "credential_present": bool(credential),
            "reachable": row.get("reachable") if credential else False,
            "latency_ms": row.get("latency_ms"),
            "action": None if credential else "configure token credential",
        })
    return {
        "providers": provider_rows,
        "local_ports": [
            {"port": int(port), **results.get(f"localhost:{port}", {
                "reachable": False,
                "latency_ms": None,
            })}
            for port in (local_ports or [])
        ],
    }


async def probe_connectivity(
    providers: list[dict[str, Any]], local_ports: list[int] | None = None,
    timeout: float = 2.0,
) -> dict[str, Any]:
    """Probe configured upstreams and local ports concurrently; never raises."""
    async def provider_probe(provider: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        name = str(provider.get("name") or provider.get("provider") or "provider")
        if not (provider.get("api_key") or provider.get("token")):
            return name, {"reachable": False, "latency_ms": None}
        url = str(provider.get("base_url", "")).rstrip("/") + "/models"
        started = time.perf_counter()
        try:
            import httpx
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url, headers={"Authorization": f"Bearer {provider.get('api_key') or provider.get('token')}"})
                response.raise_for_status()
            return name, {"reachable": True, "latency_ms": round((time.perf_counter() - started) * 1000, 1)}
        except Exception:
            return name, {"reachable": False, "latency_ms": round((time.perf_counter() - started) * 1000, 1)}

    async def port_probe(port: int) -> tuple[str, dict[str, Any]]:
        started = time.perf_counter()
        try:
            reader, writer = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", port), timeout)
            writer.close()
            await writer.wait_closed()
            return f"localhost:{port}", {"reachable": True, "latency_ms": round((time.perf_counter() - started) * 1000, 1)}
        except Exception:
            return f"localhost:{port}", {"reachable": False, "latency_ms": round((time.perf_counter() - started) * 1000, 1)}

    pairs = await asyncio.gather(
        *(provider_probe(provider) for provider in providers),
        *(port_probe(int(port)) for port in (local_ports or [])),
    )
    results = dict(pairs)
    return build_health_matrix(providers, results, local_ports)


def render_health_matrix(matrix: dict[str, Any]) -> str:
    """Render a compact Textual-markup connectivity matrix."""
    lines = ["[bold]Connectivity[/bold]"]
    for row in matrix.get("providers", []):
        status = "missing token" if not row["credential_present"] else ("reachable" if row["reachable"] else "unreachable")
        latency = f" · {row['latency_ms']} ms" if row["latency_ms"] is not None else ""
        lines.append(f"  {row['name']}: {status}{latency}")
    for row in matrix.get("local_ports", []):
        status = "reachable" if row["reachable"] else "unreachable"
        lines.append(f"  proxy:{row['port']}: {status}")
    return "\n".join(lines)
