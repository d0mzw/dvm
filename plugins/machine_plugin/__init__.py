"""Theme helpers for the Machine theme.

- TOC: keeps the Markdown ``toc`` extension output on each article/page as
  ``toc`` (only when there are 2+ headings).
- Reading stats: ``words`` and ``minutes`` on every article/page.
- Prev/next: ``prev_article`` (older) and ``next_article`` (newer).
- Content post-processing: wraps tables in a scroll container and adds a
  language label to fenced code blocks.
- ``fence_mermaid_format``: superfences formatter for ```mermaid fences that
  accepts a ``{caption="..."}`` attribute and renders a <figure>.
- YAML front matter: a leading ``---`` block (Obsidian properties) is read as
  metadata, so a post can live in an Obsidian vault and be symlinked here.
"""

import datetime
import importlib
import re

import yaml

from markupsafe import Markup, escape
from pymdownx.superfences import fence_div_format

from pelican import signals
from pelican.readers import MarkdownReader
from pelican.urlwrappers import Category
from pelican.utils import pelican_open

try:
    from markdown import Markdown
except ImportError:  # pragma: no cover
    Markdown = None

WORDS_PER_MINUTE = 200

# Fence language -> label shown on the code block.
LANG_LABELS = {
    "console": "shell",
    "shell-session": "shell",
    "sh": "shell",
    "bash": "shell",
    "py": "python",
}


def fence_mermaid_format(source, language, class_name, options, md, **kwargs):
    """Superfences formatter: ``<div class="mermaid">`` wrapped in a
    ``<figure>`` when the fence has a ``caption="..."`` attribute."""
    attrs = dict(kwargs.get("attrs") or {})
    caption = attrs.pop("caption", None)
    kwargs["attrs"] = attrs
    div = fence_div_format(source, language, class_name, options, md, **kwargs)
    if not caption:
        return div
    return f"<figure>{div}<figcaption>{escape(caption)}</figcaption></figure>"


def _count_toc(tokens):
    return sum(1 + _count_toc(t.get("children", [])) for t in tokens)


_FRONT_MATTER_RE = re.compile(r"\A---[ \t]*\n(.*?\n)---[ \t]*(?:\n|\Z)", re.S)


def _split_front_matter(text):
    """Return (meta, body) for a leading YAML ``---`` block, else (None, text).
    Values become one-item lists of strings, the shape the ``meta`` extension
    produces; YAML lists (Obsidian tags) are joined with commas, which Pelican
    splits again for tags and authors."""
    match = _FRONT_MATTER_RE.match(text)
    if not match:
        return None, text
    data = yaml.safe_load(match.group(1)) or {}
    meta = {}
    for key, value in data.items():
        if value is None:
            continue
        values = value if isinstance(value, list) else [value]
        values = [
            v.isoformat() if isinstance(v, (datetime.date, datetime.datetime)) else str(v)
            for v in values
        ]
        meta[str(key)] = [", ".join(values)]
    return meta, text[match.end():]


class TocMarkdownReader(MarkdownReader):
    """MarkdownReader that keeps the rendered TOC before metadata parsing
    resets the Markdown instance, and accepts YAML front matter."""

    def read(self, source_path):
        self._source_path = source_path
        self._md = Markdown(**self.settings["MARKDOWN"])
        with pelican_open(source_path) as text:
            front_matter, text = _split_front_matter(text)
            if front_matter is not None:
                # A leading blank line stops the `meta` extension reading the body.
                text = "\n" + text
            content = self._md.convert(text)

        toc_html = getattr(self._md, "toc", "")
        toc_count = _count_toc(getattr(self._md, "toc_tokens", []))

        if front_matter is not None:
            metadata = self._parse_metadata(front_matter)
        elif hasattr(self._md, "Meta"):
            metadata = self._parse_metadata(self._md.Meta)
        else:
            metadata = {}

        if toc_count >= 2 and "toc" not in metadata:
            metadata["toc"] = Markup(toc_html)
        return content, metadata


def add_reader(readers):
    for ext in TocMarkdownReader.file_extensions:
        readers.reader_classes[ext] = TocMarkdownReader


_TAG_RE = re.compile(r"<[^>]+>")
_CODE_RE = re.compile(r'<div class="language-([\w+#.-]+) highlight">')


def _label(match):
    lang = match.group(1)
    label = LANG_LABELS.get(lang, lang)
    return f'{match.group(0)}<span class="code-lang">{label}</span>'


def process_content(instance):
    content = instance._content
    if not content:
        instance.words, instance.minutes = 0, 1
        return

    text = _TAG_RE.sub(" ", content)
    instance.words = len(text.split())
    instance.minutes = max(1, round(instance.words / WORDS_PER_MINUTE))

    content = content.replace("<table>", '<div class="table-wrap"><table>')
    content = content.replace("</table>", "</table></div>")
    content = _CODE_RE.sub(_label, content)
    instance._content = content


def set_neighbors(generator):
    articles = generator.articles  # newest first
    for i, article in enumerate(articles):
        article.next_article = articles[i - 1] if i > 0 else None
        article.prev_article = articles[i + 1] if i + 1 < len(articles) else None

    # Pelican 4.12 reads `draft.category` unconditionally in generate_drafts,
    # which crashes when CATEGORY_SAVE_AS = "" leaves articles without one.
    for draft in generator.drafts + generator.drafts_translations:
        if not hasattr(draft, "category"):
            draft.category = Category(generator.settings["DEFAULT_CATEGORY"], generator.settings)


def resolve_fence_formatters(pelican):
    """Superfences needs callables for custom fence ``format``/``validator``;
    allow dotted-path strings in pelicanconf.py (as MkDocs configs do)."""
    configs = pelican.settings.get("MARKDOWN", {}).get("extension_configs", {})
    for fence in configs.get("pymdownx.superfences", {}).get("custom_fences", []):
        for key in ("format", "validator"):
            if isinstance(fence.get(key), str):
                module, _, name = fence[key].rpartition(".")
                fence[key] = getattr(importlib.import_module(module), name)


def register():
    signals.initialized.connect(resolve_fence_formatters)
    signals.readers_init.connect(add_reader)
    signals.content_object_init.connect(process_content)
    signals.article_generator_finalized.connect(set_neighbors)
