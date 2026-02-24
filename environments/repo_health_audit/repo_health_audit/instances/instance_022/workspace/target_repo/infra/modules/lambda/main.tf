resource "aws_lambda_function" "etl_trigger" {
  function_name = "dataforge-etl-trigger"
  runtime       = "python3.11"
  handler       = "lambda_handler.handler"
  timeout       = 300
  memory_size   = 512
}
