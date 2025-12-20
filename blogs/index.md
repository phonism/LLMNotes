---
layout: default
title: Home
lang: zh
translation: /en/
---

{% assign posts = site.posts | where: "lang", "zh" %}

## Transformer 系列

{% assign transformer_posts = posts | where_exp: "post", "post.url contains 'transformer'" | sort: "title" %}
{% for post in transformer_posts %}
<div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; margin-bottom: 0.8em;">
  <a href="{{ post.url | relative_url }}" style="font-size: 1.05em;">{{ post.title }}</a>
  <small style="color: #999;">{{ post.date | date: "%Y-%m-%d" }}</small>
</div>
{% endfor %}

## 强化学习系列

{% assign rl_posts = posts | where_exp: "post", "post.url contains '/rl/'" | sort: "date" | reverse %}
{% for post in rl_posts %}
<div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; margin-bottom: 0.8em;">
  <a href="{{ post.url | relative_url }}" style="font-size: 1.05em;">{{ post.title }}</a>
  <small style="color: #999;">{{ post.date | date: "%Y-%m-%d" }}</small>
</div>
{% endfor %}

## LLM 对齐系列

{% assign alignment_posts = posts | where_exp: "post", "post.url contains 'alignment'" | sort: "title" %}
{% for post in alignment_posts %}
<div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; margin-bottom: 0.8em;">
  <a href="{{ post.url | relative_url }}" style="font-size: 1.05em;">{{ post.title }}</a>
  <small style="color: #999;">{{ post.date | date: "%Y-%m-%d" }}</small>
</div>
{% endfor %}

## 其他文章

{% assign other_posts = posts | where_exp: "post", "post.url contains '/misc/'" %}
{% for post in other_posts %}
<div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; margin-bottom: 0.8em;">
  <a href="{{ post.url | relative_url }}" style="font-size: 1.05em;">{{ post.title }}</a>
  <small style="color: #999;">{{ post.date | date: "%Y-%m-%d" }}</small>
</div>
{% endfor %}
