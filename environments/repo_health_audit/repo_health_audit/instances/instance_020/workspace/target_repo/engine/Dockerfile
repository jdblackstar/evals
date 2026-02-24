FROM rust:1.77-slim AS builder
WORKDIR /app
COPY . /app
RUN cargo build --release

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/engine /usr/local/bin/engine
CMD ["engine"]
