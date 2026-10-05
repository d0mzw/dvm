AUTHOR = "Dominic Wang"
SITENAME = "Dom 🆚 Machine"
SITEURL = ""

PATH = "content"
THEME = "themes/machine"
PLUGIN_PATHS = ["plugins"]
PLUGINS = ["machine_plugin"]

TIMEZONE = "America/New_York"
DEFAULT_LANG = "en"
DEFAULT_PAGINATION = 10
DEFAULT_DATE_FORMAT = "%b %-d, %Y"  # Sep 14, 2026

DIRECT_TEMPLATES = ["index", "archives", "tags", "search"]
TEMPLATE_PAGES = {"search.json": "search.json"}

ARTICLE_URL = "posts/{slug}/"
ARTICLE_SAVE_AS = "posts/{slug}/index.html"
PAGE_URL = "{slug}/"
PAGE_SAVE_AS = "{slug}/index.html"
TAG_URL = "tags/{slug}/"
TAG_SAVE_AS = "tags/{slug}/index.html"

# The theme has no category/author views.
CATEGORY_SAVE_AS = ""
CATEGORIES_SAVE_AS = ""
AUTHOR_SAVE_AS = ""
AUTHORS_SAVE_AS = ""

STATIC_PATHS = ["images"]

MENUITEMS = (
    ("posts", "/"),
    ("archive", "/archives.html"),
    ("tags", "/tags.html"),
    ("search", "/search.html"),
    ("about", "/about/"),
)
DISPLAY_PAGES_ON_MENU = False
DISPLAY_CATEGORIES_ON_MENU = False

SOCIAL = (
    ("github", "https://github.com/d0mzw"),
    ("linkedin", "https://linkedin.com/in/d0mzw"),
    ("email", "mailto:dominic.ck.wang@protonmail.com"),
)

INTRO = (
    "I'm a security researcher/engineer. This is where I post notes on vulnerability research, "
    "exploit development, offensive security, and what I'm learning about AI/ML safety & internals."
    " <br><br>"  # the space keeps the page description from reading "internals.Posts"
    "<em>Posts are drafted from my Obsidian notes with AI assistance.</em>"
)

MARKDOWN = {
    "extension_configs": {
        "markdown.extensions.toc": {"toc_depth": "2-3"},
        "markdown.extensions.tables": {},
        "markdown.extensions.footnotes": {},
        "markdown.extensions.attr_list": {},
        "markdown.extensions.md_in_html": {},
        "markdown.extensions.sane_lists": {},
        "pymdownx.superfences": {
            "custom_fences": [
                {
                    "name": "mermaid",
                    "class": "mermaid",
                    "format": "machine_plugin.fence_mermaid_format",
                }
            ]
        },
        "pymdownx.highlight": {
            "css_class": "highlight",
            "use_pygments": True,
            "pygments_lang_class": True,
        },
        "pymdownx.tilde": {},
        "pymdownx.keys": {},
        "pymdownx.magiclink": {},  # link bare URLs, as Obsidian does
        "pymdownx.arithmatex": {"generic": True},  # $...$ and $$...$$ math, drawn by KaTeX
    },
    "output_format": "html5",
}

# Feeds are enabled in publishconf.py.
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

RELATIVE_URLS = False
