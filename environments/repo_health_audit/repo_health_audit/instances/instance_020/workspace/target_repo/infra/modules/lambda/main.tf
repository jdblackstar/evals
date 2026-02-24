resource "aws_lambda_function" "orchestrator" {
  function_name = "pipeline-orchestrator"
  runtime       = "python3.11"
  handler       = "orchestrator.pipeline.handler"
  timeout       = 300
  memory_size   = 512
}
