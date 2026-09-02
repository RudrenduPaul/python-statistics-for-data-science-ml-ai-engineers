# Contributing

## How the repository is organized

Each chapter is a set of three files in `chapters/`:

- `chapter-0N-<slug>.md`: the prose, in Quarto-flavored Markdown. Every interactive figure is a `::: {#fig-slug} ... :::` Quarto figure div wrapping a raw-HTML `<iframe>` pointing at a self-contained page in `_generated/`, so the book never depends on a live Python kernel to render.
- `chapter-0N-<slug>-plots.py`: the plotting code, standalone and runnable on its own. Each function builds one Plotly figure and writes it to `_generated/` as a self-contained interactive HTML page. This is the source of truth for regenerating any figure in the book.
- `chapter-0N-<slug>-plots.ipynb`: an executed, rendered preview of the same code, viewable directly on GitHub, with each figure's static output baked in as an image (GitHub's notebook viewer cannot run the live JavaScript the book itself uses).

## Building locally

```bash
pip install -r requirements.txt
python chapters/chapter-01-descriptive-statistics-plots.py   # regenerate figures for one chapter
quarto render                                                  # build the book locally
```

Requires the [Quarto CLI](https://quarto.org/docs/get-started/) (`brew install --cask quarto`). Every chapter's `-plots.py` script needs to run once, or whenever the figure code changes, to populate `_generated/` before `quarto render`.

## Reporting issues and corrections

Open an issue on this repository: https://github.com/RudrenduPaul/python-statistics-for-data-science-ml-ai-engineers/issues
