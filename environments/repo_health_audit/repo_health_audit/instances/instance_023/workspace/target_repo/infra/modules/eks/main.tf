resource "aws_eks_cluster" "main" {
  name     = "beacon-cluster"
  role_arn = "arn:aws:iam::role/eks-cluster-role"

  vpc_config {
    subnet_ids = ["subnet-abc123"]
  }
}
