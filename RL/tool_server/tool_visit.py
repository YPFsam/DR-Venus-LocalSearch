'''
Visit tool with local BM25 server support.

When USE_LOCAL_SEARCH=true (default), retrieves documents from the local
BM25 search server via HTTP instead of using Jina + LLM summarization.

When USE_LOCAL_SEARCH=false, falls back to the original Jina + LLM pipeline.
'''

import os
import re
import time
from typing import Union

from qwen_agent.tools.base import BaseTool, register_tool

USE_LOCAL_SEARCH = os.environ.get("USE_LOCAL_SEARCH", "true").lower() == "true"
LOCAL_SEARCH_SERVER_URL = os.environ.get("LOCAL_SEARCH_SERVER_URL", "http://localhost:8890")


@register_tool('visit', allow_overwrite=True)
class Visit(BaseTool):
    name = 'visit'
    description = 'Visit webpage(s) and return the summary of the content.'
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": ["string", "array"],
                "items": {"type": "string"},
                "minItems": 1,
                "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."
            },
            "goal": {
                "type": "string",
                "description": "The goal of the visit for webpage(s)."
            }
        },
        "required": ["url", "goal"]
    }

    # ── Local BM25 server (stateless HTTP) ────────────────────────────────

    def _visit_local(self, url: str, goal: str) -> str:
        """Retrieve a document from the local BM25 server.

        Only local:// URLs (returned by local search) are supported.
        Regular http(s):// URLs are rejected with a clear error message.
        """
        import requests as req
        from urllib.parse import quote

        # Extract passage ID from local:// URLs only
        passage_id_match = re.match(r'local://(.+)', url)
        if not passage_id_match:
            return (
                f"The useful information in {url} for user goal {goal} as follows: \n\n"
                f"Evidence in page: \n"
                f"Online access is disabled in local search mode. "
                f"Only local:// URLs from search results are supported.\n\n"
                f"Summary: \n"
                f"Document not available. Use search tool to find local documents.\n\n"
            )

        passage_id = passage_id_match.group(1)

        try:
            session = req.Session()
            session.trust_env = False
            resp = session.get(
                f"{LOCAL_SEARCH_SERVER_URL}/document/{quote(passage_id, safe='')}",
                timeout=10,
            )
            if resp.status_code == 404:
                return (
                    f"The useful information in local://{passage_id} for user goal {goal} as follows: \n\n"
                    f"Evidence in page: \n"
                    f"Document {passage_id} not found in local corpus.\n\n"
                    f"Summary: \n"
                    f"Document not found.\n\n"
                )
            resp.raise_for_status()
            doc = resp.json()
        except Exception as e:
            raise RuntimeError(f"Local document server request failed for {passage_id}: {e}") from e

        text = doc.get("text", "")
        title = doc.get("title", "")
        evidence = text
        summary = text[:500] if len(text) > 500 else text

        return (
            f"The useful information in local://{passage_id} "
            f"(Title: {title}) for user goal {goal} as follows: \n\n"
            f"Evidence in page: \n{evidence}\n\n"
            f"Summary: \n{summary}\n\n"
        )

    # ── Original Jina + LLM pipeline ──────────────────────────────────────

    def _visit_online(self, url: str, goal: str) -> str:
        import json
        import requests
        from dotenv import load_dotenv
        from openai import OpenAI
        from tool_server.tool_prompt import EXTRACTOR_PROMPT

        load_dotenv()

        JINA_API_KEYS = os.getenv("JINA_API_KEYS", "")
        PROXY = os.getenv("PROXY", "")
        VISIT_SERVER_MAX_RETRIES = int(os.getenv('VISIT_SERVER_MAX_RETRIES', 1))

        def truncate_to_tokens(text: str, max_tokens: int = 95000) -> str:
            try:
                import tiktoken
                encoding = tiktoken.get_encoding("cl100k_base")
                tokens = encoding.encode(text)
                if len(tokens) <= max_tokens:
                    return text
                return encoding.decode(tokens[:max_tokens])
            except ImportError:
                return text[:max_tokens * 4]

        def jina_readpage(url: str) -> str:
            for attempt in range(3):
                headers = {"Authorization": f"Bearer {JINA_API_KEYS}"}
                proxies = {"http": PROXY, "https": PROXY}
                try:
                    response = requests.get(
                        f"https://r.jina.ai/{url}",
                        proxies=proxies, headers=headers, timeout=1000
                    )
                    if response.status_code == 200:
                        return response.text
                    raise ValueError("jina readpage error")
                except Exception:
                    time.sleep(0.5)
                    if attempt == 2:
                        return "[visit] Failed to read page."
            return "[visit] Failed to read page."

        def call_server(msgs, max_retries=2):
            api_key = os.environ.get("API_KEY")
            url_llm = os.environ.get("API_BASE")
            model_name = os.environ.get("SUMMARY_MODEL_NAME", "")
            client = OpenAI(api_key=api_key, base_url=url_llm)
            for attempt in range(max_retries):
                try:
                    chat_response = client.chat.completions.create(
                        model=model_name, messages=msgs, temperature=0.7
                    )
                    content = chat_response.choices[0].message.content
                    if content:
                        try:
                            json.loads(content)
                        except:
                            left = content.find('{')
                            right = content.rfind('}')
                            if left != -1 and right != -1 and left <= right:
                                content = content[left:right+1]
                        return content
                except:
                    if attempt == (max_retries - 1):
                        return ""
                    continue

        content = None
        for _ in range(8):
            content = jina_readpage(url)
            if content and not content.startswith("[visit]") and not content.startswith("[document_parser]"):
                break
        else:
            content = None

        if content:
            content = truncate_to_tokens(content, max_tokens=95000)
            messages = [{"role": "user", "content": EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)}]
            raw = call_server(messages, max_retries=VISIT_SERVER_MAX_RETRIES)

            summary_retries = 3
            while len(raw) < 10 and summary_retries >= 0:
                truncate_length = int(0.7 * len(content)) if summary_retries > 0 else 25000
                content = content[:truncate_length]
                extraction_prompt = EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)
                messages = [{"role": "user", "content": extraction_prompt}]
                raw = call_server(messages, max_retries=VISIT_SERVER_MAX_RETRIES)
                summary_retries -= 1

            parse_retry_times = 0
            if isinstance(raw, str):
                raw = raw.replace("```json", "").replace("```", "").strip()
            while parse_retry_times < 3:
                try:
                    raw = json.loads(raw)
                    break
                except:
                    raw = call_server(messages, max_retries=VISIT_SERVER_MAX_RETRIES)
                    parse_retry_times += 1

            if parse_retry_times >= 3:
                useful_information = f"The useful information in {url} for user goal {goal} as follows: \n\n"
                useful_information += "Evidence in page: \nThe provided webpage content could not be accessed.\n\n"
                useful_information += "Summary: \nThe webpage content could not be processed.\n\n"
            else:
                useful_information = f"The useful information in {url} for user goal {goal} as follows: \n\n"
                useful_information += f"Evidence in page: \n{str(raw['evidence'])}\n\n"
                useful_information += f"Summary: \n{str(raw['summary'])}\n\n"
            return useful_information
        else:
            useful_information = f"The useful information in {url} for user goal {goal} as follows: \n\n"
            useful_information += "Evidence in page: \nThe provided webpage content could not be accessed.\n\n"
            useful_information += "Summary: \nThe webpage content could not be processed.\n\n"
            return useful_information

    # ── Unified interface ─────────────────────────────────────────────────

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            url = params["url"]
            goal = params["goal"]
        except:
            return "[Visit] Invalid request format: Input must be a JSON object containing 'url' and 'goal' fields"

        start_time = time.time()

        if isinstance(url, str):
            if USE_LOCAL_SEARCH:
                response = self._visit_local(url, goal)
            else:
                response = self._visit_online(url, goal)
        else:
            if not isinstance(url, list) or not all(isinstance(u, str) for u in url):
                return "[Visit] Invalid request format: 'url' must be a string or a list of strings"
            responses = []
            for u in url:
                if time.time() - start_time > 900:
                    responses.append(
                        f"The useful information in {u} for user goal {goal} as follows: \n\n"
                        f"Evidence in page: \nTimeout.\n\nSummary: \nTimeout.\n\n"
                    )
                else:
                    try:
                        if USE_LOCAL_SEARCH:
                            responses.append(self._visit_local(u, goal))
                        else:
                            responses.append(self._visit_online(u, goal))
                    except Exception as e:
                        if USE_LOCAL_SEARCH:
                            raise
                        responses.append(f"Error fetching {u}: {str(e)}")
            response = "\n=======\n".join(responses)

        return response.strip()
