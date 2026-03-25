# stream_server.py  
from aiohttp import web, ClientSession

async def stream(request):
    # The RPi5 pushes frames here, or you pull from it
    # Simple approach: proxy the RPi5's stream
    async with ClientSession() as session:
        async with session.get("http://ROBOT_IP:8080/stream") as resp:
            response = web.StreamResponse()
            response.content_type = resp.content_type
            await response.prepare(request)
            async for chunk in resp.content.iter_chunked(4096):
                await response.write(chunk)
    return response

app = web.Application()
app.router.add_get("/stream", stream)
web.run_app(app, port=8766)