from flask import Flask

app = Flask(__name__)


@app.errorhandler(404)
def not_found(error):
    return {"error": "not found"}, 404


@app.errorhandler(500)
def internal_error(error):
    return {"error": "internal server error"}, 500


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(debug=True)
