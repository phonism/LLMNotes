---
layout: default
title: Tags
permalink: /en/tags/
lang: en
translation: /tags/
---

# Tags

{% assign filtered_posts = site.posts | where: "lang", "en" %}
{% assign tag_counts = "" %}
{% for post in filtered_posts %}
  {% for tag in post.tags %}
    {% assign tag_counts = tag_counts | append: tag | append: "|" %}
  {% endfor %}
{% endfor %}
{% assign tag_array = tag_counts | split: "|" %}
{% assign unique_tags = tag_array | uniq %}
{% assign sorted_tags = "" %}
{% for tag in unique_tags %}
  {% if tag != "" %}
    {% assign count = 0 %}
    {% for t in tag_array %}
      {% if t == tag %}{% assign count = count | plus: 1 %}{% endif %}
    {% endfor %}
    {% assign sorted_tags = sorted_tags | append: count | append: ":" | append: tag | append: "|" %}
  {% endif %}
{% endfor %}
{% assign sorted_tags = sorted_tags | split: "|" | sort | reverse %}

{% for item in sorted_tags %}
  {% if item != "" %}
    {% assign parts = item | split: ":" %}
    {% assign tag = parts[1] %}
    {% assign count = parts[0] %}
<h2 id="{{ tag | slugify }}">{{ tag }} ({{ count }})</h2>

<ul>
{% for post in filtered_posts %}
{% if post.tags contains tag %}
<li><a href="{{ post.url | relative_url }}">{{ post.title }}</a> <small>{{ post.date | date: "%Y-%m-%d" }}</small></li>
{% endif %}
{% endfor %}
</ul>
  {% endif %}
{% endfor %}
