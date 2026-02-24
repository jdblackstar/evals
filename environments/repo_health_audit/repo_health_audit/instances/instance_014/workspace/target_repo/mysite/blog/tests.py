from django.test import TestCase

from .models import Post


class PostModelTest(TestCase):
    def setUp(self):
        self.post = Post(title="Hello", body="World " * 30, published=True)

    def test_str_returns_title(self):
        self.assertEqual(str(self.post), "Hello")

    def test_short_body_truncates_at_140(self):
        self.assertTrue(len(self.post.short_body()) <= 140)

    def test_default_ordering_is_newest_first(self):
        ordering = Post._meta.ordering
        self.assertEqual(ordering, ["-created_at"])
