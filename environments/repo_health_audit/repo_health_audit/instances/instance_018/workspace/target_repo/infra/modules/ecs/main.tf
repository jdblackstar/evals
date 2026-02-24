resource "aws_ecs_cluster" "main" {
  name = "microservices-cluster"
}

resource "aws_ecs_service" "api" {
  name            = "api-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = "api-task"
  desired_count   = 2
}
