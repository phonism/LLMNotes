---
layout: default
title: Tags
permalink: /tags/
---

# > Tags_

{% assign tags = site.posts | map: "tags" | uniq | compact | sort %}

{% for tag in tags %}
<h2 id="{{ tag | slugify }}">{{ tag }}</h2>

{% for post in site.posts %}
{% if post.tags contains tag %}
- [{{ post.title }}]({{ post.url | relative_url }}) <small>{{ post.date | date: "%Y-%m-%d" }}</small>
{% endif %}
{% endfor %}

{% endfor %}
