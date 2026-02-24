import time


class TimingMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        start = time.monotonic()
        await self.app(scope, receive, send)
        elapsed = time.monotonic() - start
        print(f"Request took {elapsed:.3f}s")
