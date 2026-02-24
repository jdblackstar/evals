terraform {
  required_version = ">= 1.5"
}

variable "job_name" {
  type    = string
  default = "data-platform-etl"
}

variable "glue_version" {
  type    = string
  default = "4.0"
}

resource "aws_glue_job" "etl" {
  name     = var.job_name
  role_arn = "arn:aws:iam::role/GlueServiceRole"

  command {
    name            = "glueetl"
    script_location = "s3://data-platform-scripts/etl.py"
    python_version  = "3"
  }

  glue_version      = var.glue_version
  number_of_workers = 5
  worker_type       = "G.1X"
}

resource "aws_glue_catalog_database" "main" {
  name = "data_platform"
}
