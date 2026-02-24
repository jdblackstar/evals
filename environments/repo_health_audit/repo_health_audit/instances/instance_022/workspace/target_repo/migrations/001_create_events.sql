CREATE TABLE IF NOT EXISTS events (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(255) NOT NULL,
    payload JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_events_type ON events (event_type);
