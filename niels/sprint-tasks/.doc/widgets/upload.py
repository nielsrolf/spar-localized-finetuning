"""Upload widget files to the planning doc's workspace (too large to pass through a chat tool call).

The doc server rebuilds a widget with its build command, so build.py and its data files must be uploaded too.

Calls the r-docs MCP server's upload_files tool directly, with the connection settings Claude Code already has.
  python upload.py build.py pareto_data.json pareto_rollouts.json cipher_judge_data.json
"""
import base64
import json
import sys
from pathlib import Path

import httpx

DOC = "cmvqoowyd00sz1gicsw7a10xu"
HERE = Path(__file__).parent
servers = [p["mcpServers"]["r-docs"] for p in json.loads((Path.home() / ".claude.json").read_text())["projects"].values()
           if "r-docs" in p.get("mcpServers", {})]
url, headers = servers[0]["url"], {**servers[0]["headers"], "Accept": "application/json, text/event-stream"}


def rpc(client, method, params=None, notify=False):
    body = {"jsonrpc": "2.0", "method": method, "params": params or {}}
    if not notify:
        body["id"] = 1
    r = client.post(url, json=body)
    r.raise_for_status()
    if "mcp-session-id" in r.headers:
        client.headers["mcp-session-id"] = r.headers["mcp-session-id"]
    if notify or not r.text.strip():
        return None
    text = r.text
    if "text/event-stream" in r.headers.get("content-type", ""):
        text = [line[5:] for line in text.splitlines() if line.startswith("data:")][-1]
    return json.loads(text)


with httpx.Client(headers=headers, timeout=300) as client:
    rpc(client, "initialize", {"protocolVersion": "2025-03-26", "capabilities": {},
                               "clientInfo": {"name": "widget-upload", "version": "1"}})
    rpc(client, "notifications/initialized", notify=True)
    files = [{"path": f".doc/widgets/{name}", "content_base64": base64.b64encode((HERE / name).read_bytes()).decode()}
             for name in sys.argv[1:]]
    out = rpc(client, "tools/call", {"name": "upload_files", "arguments": {
        "document": DOC, "files": files, "message": "Update widgets"}})
    print(json.dumps(out.get("result", out))[:400])
