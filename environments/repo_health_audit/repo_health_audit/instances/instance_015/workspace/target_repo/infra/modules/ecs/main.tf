terraform {
  required_version = ">= 1.5"
}

variable "cluster_name" {
  type    = string
  default = "workspace-cluster"
}

variable "service_count" {
  type    = number
  default = 2
}

resource "aws_ecs_cluster" "main" {
  name = var.cluster_name
}

resource "aws_ecs_service" "api" {
  name            = "api-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = "api-task:1"
  desired_count   = var.service_count
}

resource "aws_ecs_service" "worker" {
  name            = "worker-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = "worker-task:1"
  desired_count   = var.service_count
}
