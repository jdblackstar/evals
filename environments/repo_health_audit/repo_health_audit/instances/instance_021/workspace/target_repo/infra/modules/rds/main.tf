resource "aws_db_instance" "main" {
  engine         = "postgres"
  instance_class = "db.t3.medium"
  allocated_storage = 20
}
