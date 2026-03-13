from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Set

import httpx


def extract_citations(resp: Any) -> List[Dict[str, Any]]:
    citations: List[Dict[str, Any]] = []

    def add(
        uri: Optional[str],
        title: Optional[str],
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
    ) -> None:
        if not (uri or title):
            return
        citations.append(
            {
                "uri": uri,
                "title": title,
                "start_index": start_idx,
                "end_index": end_idx,
            }
        )

    try:
        gm = getattr(resp, "grounding_metadata", None) or getattr(resp, "groundingMetadata", None)
        if gm:
            atts = (
                getattr(gm, "grounding_attributions", None)
                or getattr(gm, "attributions", None)
                or []
            )
            for a in atts:
                web = getattr(a, "web", None) or getattr(a, "source", None) or {}
                uri = getattr(web, "uri", None) or getattr(web, "url", None)
                title = getattr(web, "title", None) or getattr(web, "site", None)
                add(uri, title)
    except Exception:
        pass

    try:
        cm = getattr(resp, "citation_metadata", None) or getattr(resp, "citationMetadata", None)
        if cm:
            sources = getattr(cm, "citation_sources", None) or getattr(cm, "sources", None) or []
            for s in sources:
                uri = getattr(s, "uri", None) or getattr(s, "url", None)
                title = getattr(s, "title", None)
                add(uri, title)
    except Exception:
        pass

    try:
        for cand in getattr(resp, "candidates", []) or []:
            gm = getattr(cand, "grounding_metadata", None) or getattr(
                cand, "groundingMetadata", None
            )
            if gm:
                atts = (
                    getattr(gm, "grounding_attributions", None)
                    or getattr(gm, "attributions", None)
                    or []
                )
                for a in atts:
                    web = getattr(a, "web", None) or getattr(a, "source", None) or {}
                    uri = getattr(web, "uri", None) or getattr(web, "url", None)
                    title = getattr(web, "title", None) or getattr(web, "site", None)
                    add(uri, title)

            cm = getattr(cand, "citation_metadata", None) or getattr(cand, "citationMetadata", None)
            if cm:
                sources = (
                    getattr(cm, "citation_sources", None) or getattr(cm, "sources", None) or []
                )
                for s in sources:
                    uri = getattr(s, "uri", None) or getattr(s, "url", None)
                    title = getattr(s, "title", None)
                    add(uri, title)
    except Exception:
        pass

    try:
        for cand in getattr(resp, "candidates", []) or []:
            for part in getattr(getattr(cand, "content", object()), "parts", []) or []:
                meta = getattr(part, "metadata", None)
                if not meta:
                    continue
                for c in getattr(meta, "citations", None) or []:
                    uri = getattr(c, "uri", None) or getattr(c, "url", None)
                    title = getattr(c, "title", None)
                    start_idx = getattr(c, "start_index", None)
                    end_idx = getattr(c, "end_index", None)
                    add(uri, title, start_idx, end_idx)
    except Exception:
        pass

    seen: set[tuple[Optional[str], Optional[str]]] = set()
    deduped: List[Dict[str, Any]] = []
    for c in citations:
        key = (c.get("uri"), c.get("title"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    return deduped


def resolve_vertex_redirect(url: str, client: httpx.Client, cache: Dict[str, str]) -> str:
    try:
        if url in cache:
            return cache[url]
        if "vertexaisearch.cloud.google.com" in (url or "") and "/grounding-api-redirect/" in (
            url or ""
        ):
            try:
                r = client.head(url)
                final_url = str(r.url)
            except Exception:
                r = client.get(url)
                final_url = str(r.url)
            cache[url] = final_url
            return final_url
        return url
    except Exception:
        return url


def resolve_citation_urls(
    citations: List[Dict[str, Any]], client: httpx.Client
) -> List[Dict[str, Any]]:
    cache: Dict[str, str] = {}
    out: List[Dict[str, Any]] = []
    for c in citations:
        u = c.get("uri")
        if isinstance(u, str) and u:
            c = {**c, "uri": resolve_vertex_redirect(u, client, cache)}
        out.append(c)
    return out


def collect_grounding_urls(resp: Any) -> Set[str]:
    urls: Set[str] = set()
    try:
        gm = getattr(resp, "grounding_metadata", None) or getattr(resp, "groundingMetadata", None)
        if not gm:
            cands = getattr(resp, "candidates", []) or []
            if cands:
                gm = getattr(cands[0], "grounding_metadata", None) or getattr(
                    cands[0], "groundingMetadata", None
                )
        if not gm:
            return urls
        chunks = getattr(gm, "grounding_chunks", None) or getattr(gm, "chunks", None) or []
        for ch in chunks:
            try:
                web = getattr(ch, "web", None) or {}
                url = getattr(web, "uri", None) or getattr(web, "url", None)
                if url:
                    urls.add(str(url))
            except Exception:
                continue
        return urls
    except Exception:
        return urls


async def async_resolve_urls(
    urls: Set[str], client: httpx.AsyncClient, max_concurrency: int = 4
) -> Dict[str, str]:
    cache: Dict[str, str] = {}

    sem = asyncio.Semaphore(max_concurrency)

    async def resolve_one(u: str) -> None:
        try:
            async with sem:
                if "vertexaisearch.cloud.google.com" in (
                    u or ""
                ) and "/grounding-api-redirect/" in (u or ""):
                    try:
                        r = await client.head(u)
                        final_url = str(r.url)
                    except Exception:
                        r = await client.get(u)
                        final_url = str(r.url)
                    cache[u] = final_url
                else:
                    cache[u] = u
        except Exception:
            cache[u] = u

    tasks = [resolve_one(u) for u in urls]
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    return cache


def inject_inline_citations(text: str, resp: Any, resolve_url) -> str:
    try:
        if not text:
            return text

        gm = getattr(resp, "grounding_metadata", None) or getattr(resp, "groundingMetadata", None)
        if not gm:
            cands = getattr(resp, "candidates", []) or []
            if cands:
                gm = getattr(cands[0], "grounding_metadata", None) or getattr(
                    cands[0], "groundingMetadata", None
                )
        if not gm:
            return text

        chunks = getattr(gm, "grounding_chunks", None) or getattr(gm, "chunks", None) or []
        supports = getattr(gm, "grounding_supports", None) or getattr(gm, "supports", None) or []
        if not chunks or not supports:
            return text

        idx_to_url: Dict[int, str] = {}
        for idx, ch in enumerate(chunks):
            try:
                web = getattr(ch, "web", None) or {}
                url = getattr(web, "uri", None) or getattr(web, "url", None)
                if url:
                    idx_to_url[idx] = str(url)
            except Exception:
                continue

        insertions: List[Dict[str, Any]] = []
        for s in supports:
            try:
                seg = getattr(s, "segment", None) or {}
                end_idx = getattr(seg, "end_index", None)
                idxs = (
                    getattr(s, "grounding_chunk_indices", None) or getattr(s, "indices", None) or []
                )
                if not isinstance(end_idx, int):
                    continue
                urls: List[str] = []
                seen_local: set[str] = set()
                for i in idxs or []:
                    key = int(i) if isinstance(i, (int,)) else i
                    u = idx_to_url.get(key)
                    if u and u not in seen_local:
                        seen_local.add(u)
                        urls.append(resolve_url(u))
                if urls:
                    pos = max(0, min(len(text), end_idx))
                    insertions.append({"pos": pos, "urls": urls})
            except Exception:
                continue

        if not insertions:
            return text

        merged: Dict[int, List[str]] = {}
        for ins in insertions:
            p = int(ins["pos"])  # safe
            cur = merged.get(p, [])
            for u in ins["urls"]:
                if u not in cur:
                    cur.append(u)
            merged[p] = cur

        for p in sorted(merged.keys(), reverse=True):
            urls = merged[p]
            marker = " [cite: " + ", ".join(urls) + "]"
            if 0 <= p <= len(text):
                text = text[:p] + marker + text[p:]
        return text
    except Exception:
        return text
