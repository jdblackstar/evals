resource "aws_elasticache_cluster" "redis" {
  cluster_id      = "nexus-cache"
  engine          = "redis"
  node_type       = "cache.t3.micro"
  num_cache_nodes = 1
}
