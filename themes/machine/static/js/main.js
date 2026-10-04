(function () {
  "use strict";

  var root = document.documentElement;

  /* ---------- Theme toggle ---------- */

  var toggle = document.querySelector(".theme-toggle");
  if (toggle) {
    toggle.addEventListener("click", function () {
      var next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("theme", next); } catch (e) {}
      renderMermaid();
    });
  }

  /* ---------- Mobile menu ---------- */

  var header = document.querySelector(".site-header");
  var menu = document.querySelector(".menu-toggle");
  if (header && menu) {
    var setMenu = function (open) {
      header.classList.toggle("menu-open", open);
      menu.setAttribute("aria-expanded", open ? "true" : "false");
    };
    menu.addEventListener("click", function () {
      setMenu(!header.classList.contains("menu-open"));
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape") setMenu(false);
    });
    document.addEventListener("click", function (e) {
      if (!header.contains(e.target)) setMenu(false);
    });
  }

  /* ---------- Back to top ---------- */

  var toTop = document.querySelector(".to-top");
  if (toTop) {
    toTop.addEventListener("click", function (e) {
      e.preventDefault();
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }

  /* ---------- Search ---------- */

  var input = document.getElementById("search-input");
  if (input) {
    var results = document.getElementById("search-results");
    var empty = document.getElementById("search-empty");
    var posts = [];

    var el = function (tag, cls, text) {
      var node = document.createElement(tag);
      if (cls) node.className = cls;
      if (text) node.textContent = text;
      return node;
    };

    var render = function () {
      var q = input.value.trim().toLowerCase();
      var hits = q
        ? posts.filter(function (p) {
            return (p.title + " " + p.summary + " " + p.tags.join(" ")).toLowerCase().indexOf(q) !== -1;
          })
        : posts;

      results.textContent = "";
      hits.forEach(function (p) {
        var a = el("a");
        a.href = p.url;
        a.appendChild(el("span", "title", p.title));
        a.appendChild(el("span", "sub", p.tags.length ? p.date + " · " + p.tags.join(", ") : p.date));
        results.appendChild(a);
      });
      empty.hidden = hits.length !== 0;
    };

    var params = new URLSearchParams(window.location.search);
    if (params.get("q")) input.value = params.get("q");

    input.addEventListener("input", render);
    input.addEventListener("keydown", function (e) {
      if (e.key === "Enter") {
        var first = results.querySelector("a");
        if (first) window.location.href = first.href;
      }
    });

    fetch(input.getAttribute("data-index"))
      .then(function (r) { return r.json(); })
      .then(function (data) { posts = data; render(); })
      .catch(function () { empty.hidden = false; });
  }

  /* ---------- Math ---------- */

  // pymdownx.arithmatex marks math as <span class="arithmatex">\(...\)</span> and
  // <div class="arithmatex">\[...\]</div>; KaTeX is loaded only on pages that have some.
  var maths = document.querySelectorAll(".arithmatex");
  if (maths.length) {
    var KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16/dist/";
    var css = document.createElement("link");
    css.rel = "stylesheet";
    css.href = KATEX + "katex.min.css";
    document.head.appendChild(css);
    var js = document.createElement("script");
    js.src = KATEX + "katex.min.js";
    js.onload = function () {
      maths.forEach(function (el) {
        var tex = el.textContent.trim();
        var display = el.tagName === "DIV";
        window.katex.render(tex.slice(2, -2), el, { displayMode: display, throwOnError: false });
      });
      // The TOC holds headings as plain text, so math in a heading arrives as \(...\)
      document.querySelectorAll(".post-toc a").forEach(function (a) {
        if (a.textContent.indexOf("\\(") === -1) return;
        var parts = a.textContent.split(/\\\((.*?)\\\)/);
        a.textContent = "";
        parts.forEach(function (part, i) {
          if (i % 2 === 0) { a.appendChild(document.createTextNode(part)); return; }
          var span = document.createElement("span");
          window.katex.render(part, span, { throwOnError: false });
          a.appendChild(span);
        });
      });
    };
    document.head.appendChild(js);
  }

  /* ---------- Mermaid ---------- */

  var blocks = document.querySelectorAll(".mermaid");
  var mermaidReady = null;
  var renderCount = 0;

  function loadMermaid() {
    if (mermaidReady) return mermaidReady;
    mermaidReady = new Promise(function (resolve, reject) {
      var s = document.createElement("script");
      s.src = "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js";
      s.onload = function () { resolve(window.mermaid); };
      s.onerror = reject;
      document.head.appendChild(s);
    });
    return mermaidReady;
  }

  function renderMermaid() {
    if (!blocks.length) return;
    Promise.all([loadMermaid(), document.fonts ? document.fonts.ready : null])
      .then(function (res) {
        var mermaid = res[0];
        var cs = getComputedStyle(root);
        var v = function (name) { return cs.getPropertyValue(name).trim(); };
        mermaid.initialize({
          startOnLoad: false,
          theme: "base",
          flowchart: { padding: 14, useMaxWidth: false },
          themeVariables: {
            fontFamily: "Lato, Helvetica, sans-serif",
            fontSize: "15px",
            primaryColor: v("--bg"),
            primaryBorderColor: v("--fg2"),
            primaryTextColor: v("--fg"),
            lineColor: v("--muted"),
            secondaryColor: v("--entry"),
            tertiaryColor: v("--entry"),
            edgeLabelBackground: v("--entry"),
          },
        });
        blocks.forEach(function (block) {
          var id = "mermaid-" + ++renderCount;
          mermaid
            .render(id, block.getAttribute("data-src"))
            .then(function (out) {
              block.innerHTML = out.svg;
              block.setAttribute("data-processed", "true");
            })
            .catch(function () {
              var stray = document.getElementById("d" + id);
              if (stray) stray.remove();
            });
        });
      })
      .catch(function () {});
  }

  blocks.forEach(function (block) {
    block.setAttribute("data-src", block.textContent);
  });
  renderMermaid();
})();
