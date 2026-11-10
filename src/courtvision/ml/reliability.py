from __future__ import annotations


def reliability_bins(
    y_true: list[int],
    y_prob: list[float],
    bins: int = 5,
) -> list[dict]:
    rows = []
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        bucket = [
            (label, probability)
            for label, probability in zip(y_true, y_prob, strict=True)
            if (probability >= low and probability < high) or (index == bins - 1 and probability == 1.0)
        ]
        if not bucket:
            rows.append(
                {
                    "low": low,
                    "high": high,
                    "count": 0,
                    "mean_predicted": None,
                    "fraction_positive": None,
                }
            )
            continue
        rows.append(
            {
                "low": low,
                "high": high,
                "count": len(bucket),
                "mean_predicted": sum(probability for _label, probability in bucket) / len(bucket),
                "fraction_positive": sum(label for label, _probability in bucket) / len(bucket),
            }
        )
    return rows


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def reliability_svg(bins: list[dict], title: str = "Confiabilidade na validação") -> str:
    width, height = 440, 300
    left, right, top, bottom = 48, 16, 36, 46
    plot_w = width - left - right
    plot_h = height - top - bottom

    def x_of(value: float) -> float:
        return left + max(0.0, min(value, 1.0)) * plot_w

    def y_of(value: float) -> float:
        return top + (1.0 - max(0.0, min(value, 1.0))) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        f"<title>{_escape(title)}</title>",
        '<rect width="100%" height="100%" fill="#f7f4ef"/>',
        f'<line x1="{x_of(0):.1f}" y1="{y_of(0):.1f}" x2="{x_of(1):.1f}" y2="{y_of(1):.1f}" '
        'stroke="#8a8175" stroke-dasharray="4 4"/>',
    ]
    for item in bins:
        if not item["count"] or item["mean_predicted"] is None or item["fraction_positive"] is None:
            continue
        radius = 4 + min(int(item["count"]), 20) * 0.35
        parts.append(
            f'<circle cx="{x_of(float(item["mean_predicted"])):.1f}" '
            f'cy="{y_of(float(item["fraction_positive"])):.1f}" r="{radius:.1f}" '
            'fill="#1f6f5b" fill-opacity="0.85">'
            f'<title>{item["count"]} jogos</title></circle>'
        )
    parts.append(
        f'<text x="{left}" y="22" font-size="14" font-family="sans-serif">{_escape(title)}</text>'
    )
    parts.append(
        f'<text x="{left}" y="{height - 14}" font-size="11" font-family="sans-serif" fill="#5c564e">'
        "Validação interna. Não é resultado externo verificado.</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"
