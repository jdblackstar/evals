# Service Mesh Monorepo

A microservices platform with gateway routing, user management, and notifications.

## Structure

- `services/gateway/` - API gateway and authentication
- `services/users/` - User CRUD service
- `services/notifications/` - Async notification delivery
- `libs/common/` - Shared configuration and utilities

## Setup

```bash
pip install -r requirements.txt
```

## Testing

Run all service tests:

```bash
make test-all
```

## Coverage

```bash
make coverage
```

## Deployment

Each service has its own Dockerfile and can be deployed independently.
Services communicate over HTTP/gRPC via the gateway.
