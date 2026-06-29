"""Flat composer-style index for DeepJ style conditioning (arxiv:1801.00887)."""
from constants import genre, styles, NUM_STYLES


def flatten_styles():
    """Return [(index, genre_name, composer_slug, data_path), ...]."""
    rows = []
    idx = 0
    for genre_id, genre_name in enumerate(genre):
        for path in styles[genre_id]:
            slug = path.rsplit("/", 1)[-1]
            rows.append((idx, genre_name, slug, path))
            idx += 1
    assert idx == NUM_STYLES, (idx, NUM_STYLES)
    return rows


def list_styles_text():
    lines = ["DeepJ composer style indices (mix with --styles i j …):"]
    for idx, genre_name, slug, path in flatten_styles():
        lines.append(f"  {idx:2d}  {genre_name:9s}  {slug:12s}  ({path})")
    lines.append("")
    lines.append("Genre averages (use --genre):")
    for i, g in enumerate(genre):
        lines.append(f"  {i}  {g}")
    return "\n".join(lines)