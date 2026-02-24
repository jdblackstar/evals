from fastapi import FastAPI

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/auth")
def auth_route():
    return {"service": "auth"}


@app.get("/billing")
def billing_route():
    return {"service": "billing"}
