import json
import os
from pathlib import Path
from aiohttp import web

ALERTS_DB_PATH = Path(os.getenv("ALERTS_DB_PATH", "/data/alerts.jsonl"))
SNAPSHOT_DIR = Path(os.getenv("CLOUD_SNAPSHOT_DIR", "/cloud_snapshots"))
WHITELIST_DIR = Path(os.getenv("WHITELIST_DIR", "/cloud/whitelist"))
WHITELIST_DIR.mkdir(parents=True, exist_ok=True)


def _load_alerts(limit: int = 200):
    if not ALERTS_DB_PATH.exists():
        return []
    items = []
    with open(ALERTS_DB_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    # Most recent first
    items = items[-limit:]
    items.reverse()
    return items


async def alerts_handler(request):
    try:
        limit = int(request.query.get("limit", "200"))
    except ValueError:
        limit = 200
    data = _load_alerts(limit=limit)
    resp = web.json_response({"alerts": data})
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


async def snapshot_handler(request):
    name = request.match_info.get("name")
    if not name:
        return web.Response(status=404)
    safe_name = Path(name).name
    path = SNAPSHOT_DIR / safe_name
    if not path.exists():
        return web.Response(status=404)
    resp = web.FileResponse(path)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


def _is_allowed_image(name: str) -> bool:
    return name.lower().endswith((".jpg", ".jpeg", ".png"))


async def whitelist_list_handler(_request):
    items = []
    for path in sorted(WHITELIST_DIR.glob("*")):
        if not path.is_file() or not _is_allowed_image(path.name):
            continue
        items.append({
            "name": path.stem,
            "filename": path.name,
        })
    resp = web.json_response({"items": items})
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


async def whitelist_upload_handler(request):
    reader = await request.multipart()
    field = await reader.next()
    if field is None or field.name != "file":
        return web.json_response({"error": "missing file field"}, status=400)
    filename = Path(field.filename).name if field.filename else ""
    if not filename or not _is_allowed_image(filename):
        return web.json_response({"error": "invalid filename"}, status=400)

    out_path = WHITELIST_DIR / filename
    with open(out_path, "wb") as f:
        while True:
            chunk = await field.read_chunk()
            if not chunk:
                break
            f.write(chunk)

    resp = web.json_response({"ok": True, "filename": filename})
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


async def whitelist_delete_handler(request):
    name = request.match_info.get("name")
    if not name:
        return web.Response(status=404)
    safe_name = Path(name).name
    path = WHITELIST_DIR / safe_name
    if not path.exists():
        return web.Response(status=404)
    try:
        path.unlink()
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)
    resp = web.json_response({"ok": True})
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


async def whitelist_image_handler(request):
    name = request.match_info.get("name")
    if not name:
        return web.Response(status=404)
    safe_name = Path(name).name
    path = WHITELIST_DIR / safe_name
    if not path.exists():
        return web.Response(status=404)
    resp = web.FileResponse(path)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


async def health_handler(_request):
    return web.json_response({"ok": True})


app = web.Application()
app.router.add_get("/api/alerts", alerts_handler)
app.router.add_get("/snapshots/{name}", snapshot_handler)
app.router.add_get("/api/whitelist", whitelist_list_handler)
app.router.add_post("/api/whitelist", whitelist_upload_handler)
app.router.add_delete("/api/whitelist/{name}", whitelist_delete_handler)
app.router.add_get("/whitelist/{name}", whitelist_image_handler)
app.router.add_get("/health", health_handler)

if __name__ == "__main__":
    port = int(os.getenv("HISTORY_API_PORT", "8780"))
    web.run_app(app, port=port)
