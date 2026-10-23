"""Tidy the pages mkdocs-jupyter makes from the guides.

mkdocs-jupyter appends an indented <style> block to every notebook, which
the Markdown pass over the guides turns into a code block showing the CSS
as text at the foot of the page. The same CSS is already in the page's
head, so the block is removed, along with the paragraph tags that end up
around the notebook's closing tags.
"""

import re

_STYLE_BLOCK = re.compile(
    r'<div class="highlight"><pre><span></span><code>\s*&lt;style&gt;.*?&lt;/style&gt;\s*</code></pre></div>',
    re.S,
)
_CLOSING = re.compile(r"<p>(</div> <!-- jp-Notebook -->\s*</div> <!-- jupyter-wrapper -->)</p>")


def on_page_content(html, page, config, files):
    if not page.file.src_uri.endswith(".ipynb"):
        return html
    html = _STYLE_BLOCK.sub("", html)
    return _CLOSING.sub(r"\1", html)
