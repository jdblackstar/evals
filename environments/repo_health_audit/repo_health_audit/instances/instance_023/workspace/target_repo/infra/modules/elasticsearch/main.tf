resource "aws_elasticsearch_domain" "search" {
  domain_name           = "beacon-search"
  elasticsearch_version = "7.17"

  cluster_config {
    instance_type  = "r6g.large.elasticsearch"
    instance_count = 3
  }
}
