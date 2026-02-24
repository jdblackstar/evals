resource "aws_sqs_queue" "pipeline_queue" {
  name                       = "pipeline-tasks"
  visibility_timeout_seconds = 300
  message_retention_seconds  = 86400
}
