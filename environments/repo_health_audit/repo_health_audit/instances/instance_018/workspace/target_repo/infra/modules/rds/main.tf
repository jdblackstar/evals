resource "aws_db_instance" "main" {
  engine         = "postgres"
  instance_class = "db.t3.micro"
  allocated_storage = 20
}
