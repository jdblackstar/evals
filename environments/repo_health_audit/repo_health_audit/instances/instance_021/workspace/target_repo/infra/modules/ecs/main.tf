resource "aws_ecs_cluster" "main" {
  name = "nexus-cluster"
}

resource "aws_ecs_service" "gateway" {
  name            = "gateway"
  cluster         = aws_ecs_cluster.main.id
  task_definition = "gateway:latest"
  desired_count   = 2
}
