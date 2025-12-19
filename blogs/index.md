---
layout: default
title: Home
---

# > LLM Notes
LLM学习笔记

## Posts

{% for post in site.posts %}
### [{{ post.title }}]({{ post.url | relative_url }})
<small>{{ post.date | date: "%Y-%m-%d" }}</small>

{{ post.excerpt }}

---
{% endfor %}

## About

- 📘 [RL Notes](https://github.com/phonism/LLMNotes/tree/main/notes/RL) - 强化学习完整笔记
- 📗 [Transformer Notes](https://github.com/phonism/LLMNotes/tree/main/notes/Transformers) - LLM 技术笔记
- 📧 [luqi.code@gmail.com](mailto:luqi.code@gmail.com)
