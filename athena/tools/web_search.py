"""Web Search provider for Athena.

Provides a provider abstraction for web search. The first implementation
uses DuckDuckGo via the Lite endpoint (no API key required).

Future providers can be added by:
1. Implementing the same function signature.
2. Registering with @register_provider("name").

The provider is swappable by configuration only - set WEB_SEARCH_PROVIDER
in settings.py to change the implementation.
"""

import re
from typing import Optional

import requests

from athena.logging.logger import logger


# Provider registry

_PROVIDERS = {}


def register_provider(name: str):
    """Decorator to register a web search provider implementation."""
    def decorator(func):
        _PROVIDERS[name] = func
        return func
    return decorator


def get_provider(name: str):
    """Retrieve a registered provider by name."""
    return _PROVIDERS.get(name)


# DuckDuckGo provider (Lite endpoint)

_DDG_LITE_URL = "https://lite.duckduckgo.com/lite/"
_DDG_SESSION = requests.Session()


def _parse_ddg_lite(html: str, max_results: int) -> list[dict[str, str]]:
    """Parse DuckDuckGo Lite HTML search results.

    The Lite endpoint returns a simple HTML table structure:
        <a rel="nofollow" href="URL" class='result-link'>Title</a>
        <td class='result-snippet'>Snippet</td>

    Extracts title, URL, and snippet from paired link/snippet rows.
    """
    # Extract all result links: (url, title_html)
    link_pattern = (
        r'<a rel="nofollow"[^>]*href="(https?://[^"]+)"[^>]*'
        r"class='result-link'[^>]*>(.*?)</a>"
    )
    link_matches = re.findall(link_pattern, html, re.DOTALL)

    # Extract all result snippets
    snippet_pattern = r"<td class='result-snippet'>(.*?)</td>"
    snippet_matches = re.findall(snippet_pattern, html, re.DOTALL)

    results = []
    for i, (url, title_html) in enumerate(link_matches):
        if len(results) >= max_results:
            break

        # Strip HTML tags from title
        title = re.sub(r'<[^>]+>', '', title_html).strip()

        # Get corresponding snippet
        snippet_value = ""
        if i < len(snippet_matches):
            snippet_value = re.sub(r'<[^>]+>', '', snippet_matches[i]).strip()

        if title or url:
            results.append({
                "title": title,
                "href": url,
                "body": snippet_value,
            })

    return results


@register_provider("duckduckgo")
def _duckduckgo_search(
    query: str,
    max_results: int = 5,
    timeout: int = 10,
    user_agent: str = "Athena",
) -> Optional[list[dict[str, str]]]:
    """Search DuckDuckGo and return results using the Lite endpoint.

    Each result dict contains:
        - title: The page title.
        - href: The page URL.
        - body: A snippet of text from the page.

    Uses a persistent requests.Session to handle cookies across requests.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return.
        timeout: Request timeout in seconds.
        user_agent: User-Agent string to use for the request.

    Returns:
        A list of result dicts, or an empty list if no results,
        or None if the search failed.
    """
    try:
        headers = {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

        # Ensure we have cookies first (GET sets the session cookie)
        _DDG_SESSION.get(
            _DDG_LITE_URL,
            headers=headers,
            timeout=timeout,
        )

        # Perform the search
        response = _DDG_SESSION.post(
            _DDG_LITE_URL,
            data={"q": query},
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()

        results = _parse_ddg_lite(response.text, max_results)

        if not results:
            logger.info("Web search returned zero results for query: %s", query)
            return []

        return results

    except requests.Timeout:
        logger.error("Web search (DuckDuckGo) timed out for query: %s", query)
        return None
    except requests.ConnectionError:
        logger.error(
            "Web search (DuckDuckGo) connection error for query: %s", query
        )
        return None
    except Exception:
        logger.exception(
            "Web search (DuckDuckGo) failed for query: %s", query
        )
        return None