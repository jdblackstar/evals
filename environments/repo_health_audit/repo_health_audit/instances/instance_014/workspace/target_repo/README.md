# MySite Blog

A legacy Django project for a simple blog application.

## Running Tests

```bash
python manage.py test
```

## Coverage

```bash
coverage run manage.py test && coverage report
```

A coverage snapshot is available in `artifacts/coverage_output.txt`.

## Deployment

The project includes a Dockerfile for containerised deployment with gunicorn.

## Notes

- Settings are hardcoded in `mysite/settings.py` with no environment-based overrides.
- Static files are served via whitenoise.
