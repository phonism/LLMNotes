---
layout: default
title: Home
lang: en
translation: /
---

{% assign posts = site.posts | where: "lang", "en" %}

## Transformer Series

{% assign transformer_posts = posts | where_exp: "post", "post.url contains 'transformer'" | sort: "title" %}
{% for post in transformer_posts %}
<div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; margin-bottom: 0.8em;">
  <a href="{{ post.url | relative_url }}" style="font-size: 1.05em;">{{ post.title }}</a>
  <small style="color: #999;">{{ post.date | date: "%Y-%m-%d" }}</small>
</div>
{% endfor %}

## Reinforcement Learning Series

{% assign rl_posts = posts | where_exp: "post", "post.url contains '/rl/'" | sort: "date" | reverse %}
{% for post in rl_posts %}
<div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; margin-bottom: 0.8em;">
  <a href="{{ post.url | relative_url }}" style="font-size: 1.05em;">{{ post.title }}</a>
  <small style="color: #999;">{{ post.date | date: "%Y-%m-%d" }}</small>
</div>
{% endfor %}

## LLM Alignment Series

{% assign alignment_posts = posts | where_exp: "post", "post.url contains 'alignment'" | sort: "title" %}
{% for post in alignment_posts %}
<div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; margin-bottom: 0.8em;">
  <a href="{{ post.url | relative_url }}" style="font-size: 1.05em;">{{ post.title }}</a>
  <small style="color: #999;">{{ post.date | date: "%Y-%m-%d" }}</small>
</div>
{% endfor %}

## Other Posts

{% assign other_posts = posts | where_exp: "post", "post.url contains '/misc/'" %}
{% for post in other_posts %}
<div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; margin-bottom: 0.8em;">
  <a href="{{ post.url | relative_url }}" style="font-size: 1.05em;">{{ post.title }}</a>
  <small style="color: #999;">{{ post.date | date: "%Y-%m-%d" }}</small>
</div>
{% endfor %}
