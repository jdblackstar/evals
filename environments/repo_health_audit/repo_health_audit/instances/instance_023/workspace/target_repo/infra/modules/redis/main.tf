resource "aws_elasticache_cluster" "sessions" {
  cluster_id      = "beacon-sessions"
  engine          = "redis"
  node_type       = "cache.r6g.large"
  num_cache_nodes = 2
}
