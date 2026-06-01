'''
Search tool with local search server support.

When USE_LOCAL_SEARCH=true (default), sends HTTP requests to a local
search server instead of the Serper API. This keeps each Ray worker stateless.

When USE_LOCAL_SEARCH=false, falls back to the original Serper API.
'''

import os
from typing import List, Optional, Union

from qwen_agent.tools.base import BaseTool, register_tool

USE_LOCAL_SEARCH = os.environ.get("USE_LOCAL_SEARCH", "true").lower() == "true"
LOCAL_SEARCH_SERVER_URL = os.environ.get("LOCAL_SEARCH_SERVER_URL", "http://localhost:8890")
LOCAL_SEARCH_TOPK = int(os.environ.get("LOCAL_SEARCH_TOPK", "10"))


@register_tool("search", allow_overwrite=True)
class Search(BaseTool):
    name = "search"
    description = "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Array of query strings. Include multiple complementary search queries in a single call.",
            },
        },
        "required": ["query"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)

    # ── Local search server (stateless HTTP) ──────────────────────────────

    def _search_local_batch(self, queries: List[str]) -> List[str]:
        """Query local search server once and format Serper-compatible results."""
        import requests

        try:
            session = requests.Session()
            session.trust_env = False
            resp = session.post(
                f"{LOCAL_SEARCH_SERVER_URL}/search",
                json={"queries": queries, "topk": LOCAL_SEARCH_TOPK},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            raise RuntimeError(f"Local search server request failed: {e}") from e

        results_list = data.get("results", [])
        if len(results_list) != len(queries):
            raise RuntimeError(
                f"Local search server returned {len(results_list)} result groups for {len(queries)} queries"
            )
        return [
            self._format_local_results(query, result.get("passages", []))
            for query, result in zip(queries, results_list)
        ]

    def _format_local_results(self, query: str, passages: List[dict]) -> str:
        if not passages:
            return f"No results found for '{query}'."

        snippets = []
        for i, p in enumerate(passages, 1):
            url = f"local://{p['id']}"
            snippet = p["text"][:200]
            line = f"{i}. [{p['title']}]({url})\n{snippet}"
            snippets.append(line)

        header = f"A Google search for '{query}' found {len(passages)} results:\n\n## Web Results\n"
        return header + "\n\n".join(snippets)

    def _search_local(self, query: str) -> str:
        return self._search_local_batch([query])[0]

    # ── Original Serper API ───────────────────────────────────────────────

    def _search_serper(self, query: str) -> str:
        import http.client
        import json
        from dotenv import load_dotenv
        load_dotenv()
        SERPER_KEY = os.environ.get('SERPER_KEY_ID')

        conn = http.client.HTTPSConnection("google.serper.dev")

        def contains_chinese_basic(text: str) -> bool:
            return any('\u4E00' <= char <= '\u9FFF' for char in text)

        if contains_chinese_basic(query):
            payload = json.dumps({"q": query, "location": "China", "gl": "cn", "hl": "zh-cn"})
        else:
            payload = json.dumps({"q": query, "location": "United States", "gl": "us", "hl": "en"})
        headers = {'X-API-KEY': SERPER_KEY, 'Content-Type': 'application/json'}

        for i in range(5):
            try:
                conn.request("POST", "/search", payload, headers)
                res = conn.getresponse()
                break
            except Exception as e:
                print(e)
                if i == 4:
                    return f"Google search Timeout, return None, Please try again later."
                continue

        data = res.read()
        results = json.loads(data.decode("utf-8"))

        try:
            if "organic" not in results:
                raise Exception(f"No results found for query: '{query}'.")

            web_snippets = []
            idx = 0
            for page in results["organic"]:
                idx += 1
                date_published = "\nDate published: " + page["date"] if "date" in page else ""
                source = "\nSource: " + page["source"] if "source" in page else ""
                snippet = "\n" + page["snippet"] if "snippet" in page else ""
                redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{snippet}"
                redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                web_snippets.append(redacted_version)

            return f"A Google search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(web_snippets)
        except:
            return f"No results found for '{query}'. Try with a more general query."

    # ── Unified interface ─────────────────────────────────────────────────

    def search_with_serp(self, query: str):
        if USE_LOCAL_SEARCH:
            return self._search_local(query)
        else:
            return self._search_serper(query)

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            query = params["query"]
        except:
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field"

        if isinstance(query, str):
            response = self.search_with_serp(query)
        else:
            if not isinstance(query, list) or not all(isinstance(q, str) for q in query):
                return "[Search] Invalid request format: 'query' must be a string or a list of strings"
            if USE_LOCAL_SEARCH:
                responses = self._search_local_batch(query)
            else:
                responses = [self.search_with_serp(q) for q in query]
            response = "\n=======\n".join(responses)

        return response
