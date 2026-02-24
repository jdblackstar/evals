from django.http import JsonResponse
from django.shortcuts import get_object_or_404

from .models import Post


def post_list(request):
    posts = Post.objects.filter(published=True).values("id", "title", "created_at")
    return JsonResponse(list(posts), safe=False)


def post_detail(request, pk):
    post = get_object_or_404(Post, pk=pk)
    return JsonResponse(
        {"id": post.pk, "title": post.title, "body": post.body}
    )
