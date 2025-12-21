---
layout: default
title: Tags
permalink: /tags/
lang: zh
translation: /en/tags/
---

# Tags

{% assign filtered_posts = site.posts | where: "lang", "zh" %}
{% assign rawtags = "" %}
{% for post in filtered_posts %}
  {% assign ttags = post.tags | join:'|' | append:'|' %}
  {% assign rawtags = rawtags | append:ttags %}
{% endfor %}
{% assign rawtags = rawtags | split:'|' | sort %}

{% assign tags = "" %}
{% for tag in rawtags %}
  {% if tag != "" %}
    {% if tags == "" %}
      {% assign tags = tag | split:'|' %}
    {% endif %}
    {% unless tags contains tag %}
      {% assign tags = tags | join:'|' | append:'|' | append:tag | split:'|' %}
    {% endunless %}
  {% endif %}
{% endfor %}

{% for tag in tags %}
<h2 id="{{ tag | slugify }}">{{ tag }}</h2>

<ul>
{% for post in filtered_posts %}
{% if post.tags contains tag %}
<li><a href="{{ post.url | relative_url }}">{{ post.title }}</a> <small>{{ post.date | date: "%Y-%m-%d" }}</small></li>
{% endif %}
{% endfor %}
</ul>

{% endfor %}
