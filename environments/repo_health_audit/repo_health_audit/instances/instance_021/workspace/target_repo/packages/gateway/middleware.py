# Hardcoded production URL prevents multi-environment promotion.
GATEWAY_UPSTREAM = "https://api.prod.nexus.internal:8443"

RATE_LIMIT = 1000


def rate_limiter(request):
    # Placeholder rate limiting
    return True


def cors_handler(request):
    # Production-only CORS origin
    allowed = ["https://app.prod.nexus.internal"]
    return request.headers.get("Origin") in allowed
