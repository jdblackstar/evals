REQUIRED_FIELDS = ["id", "content", "timestamp"]


def validate_document(doc: dict) -> tuple[bool, list[str]]:
    errors = []
    for field in REQUIRED_FIELDS:
        if field not in doc:
            errors.append(f"missing required field: {field}")
    return len(errors) == 0, errors


def validate_batch(documents: list[dict]) -> tuple[list[dict], list[dict]]:
    valid = []
    invalid = []
    for doc in documents:
        ok, errs = validate_document(doc)
        if ok:
            valid.append(doc)
        else:
            invalid.append({"doc": doc, "errors": errs})
    return valid, invalid
