resource "aws_db_instance" "warehouse" {
  engine            = "postgres"
  engine_version    = "15.4"
  instance_class    = "db.r6g.large"
  allocated_storage = 100
  db_name           = "warehouse"
}
