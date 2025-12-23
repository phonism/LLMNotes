---
layout: default
title: Home
lang: zh
translation: /en/
---

{% assign posts = site.posts | where: "lang", "zh" | sort: "date" | reverse %}

## 文章列表

{% for post in posts %}
<div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; margin-bottom: 0.8em;">
  <a href="{{ post.url | relative_url }}" style="font-size: 1.05em;">{{ post.title }}</a>
  <small style="color: #999;">{{ post.date | date: "%Y-%m-%d" }}</small>
</div>
{% endfor %}

## 标签

{% assign tag_counts = "" %}
{% for post in posts %}
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

<div style="display: flex; flex-wrap: wrap; gap: 8px;">
{% for item in sorted_tags %}
  {% if item != "" %}
    {% assign parts = item | split: ":" %}
    {% assign tag = parts[1] %}
    {% assign count = parts[0] %}
    {% assign tag_slug = tag | slugify %}
<a href="{{ '/tags/' | relative_url }}#{{ tag_slug }}" class="tag-link">{{ tag }} ({{ count }})</a>
  {% endif %}
{% endfor %}
</div>
