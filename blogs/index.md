---
layout: default
title: Home
---

## Posts

{% for post in site.posts %}
<div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; margin-bottom: 1em;">
  <a href="{{ post.url | relative_url }}" style="font-size: 1.1em;">{{ post.title }}</a>
  <small style="color: #999;">{{ post.date | date: "%Y-%m-%d" }}{% if post.tags.size > 0 %} · {% for tag in post.tags %}<a href="{{ '/tags/' | relative_url }}#{{ tag | slugify }}">{{ tag }}</a>{% unless forloop.last %}, {% endunless %}{% endfor %}{% endif %}</small>
</div>
{% endfor %}
