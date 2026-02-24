module "vpc" {
  source = "./modules/vpc"
}

module "db" {
  source = "./modules/db"
}
