---
layout: default
title: Home
lang: zh
translation: /en/
---

{% assign posts = site.posts | where: "lang", "zh" | sort: "date" %}

## 文章列表

{% for post in posts %}
<div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; margin-bottom: 0.8em;">
  <a href="{{ post.url | relative_url }}" style="font-size: 1.05em;">{{ post.title }}</a>
  <small style="color: #999;">{{ post.date | date: "%Y-%m-%d" }}</small>
</div>
{% endfor %}
