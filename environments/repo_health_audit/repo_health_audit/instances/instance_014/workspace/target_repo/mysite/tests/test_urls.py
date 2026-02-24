from django.test import SimpleTestCase
from django.urls import resolve, reverse


def test_blog_list_url_resolves():
    url = reverse("blog:post_list")
    assert resolve(url).func.__name__ == "post_list"


def test_admin_url_resolves():
    url = "/admin/"
    match = resolve(url)
    assert match.app_name == "admin"
