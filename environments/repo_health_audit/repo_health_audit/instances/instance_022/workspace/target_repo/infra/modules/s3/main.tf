resource "aws_s3_bucket" "ingest" {
  bucket = "dataforge-prod-ingest"
}

resource "aws_s3_bucket" "archive" {
  bucket = "dataforge-prod-archive"
}
