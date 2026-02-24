FROM golang:1.22-alpine
WORKDIR /app
COPY . /app
RUN go build -o worker .
CMD ["./worker"]
