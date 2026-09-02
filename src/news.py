"""Live news sentiment — informational only, NOT used in the trained model or backtest.
Free news APIs don't provide a usable historical archive, so this can't be validated the
way the technical/fundamental/event features were (see README). Uses VADER, a rule-based
lexicon sentiment scorer (hand-built word list + grammar heuristics, not a trained model
or LLM) — consistent with the project's no-AI/no-LLM constraint."""
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def get_news_sentiment(stock: str, max_items: int = 10) -> dict:
    ticker = yf.Ticker(f"{stock}.NS")
    try:
        items = ticker.news or []
    except Exception:
        items = []

    headlines = []
    for item in items[:max_items]:
        content = item.get("content", item)
        title = content.get("title")
        if title:
            headlines.append({
                "title": title,
                "published": content.get("pubDate"),
                "link": (content.get("canonicalUrl") or {}).get("url") or (content.get("clickThroughUrl") or {}).get("url"),
            })

    if not headlines:
        return {"headlines": [], "mean_compound": 0.0, "label": "No recent news"}

    for h in headlines:
        h["sentiment"] = _analyzer.polarity_scores(h["title"])["compound"]

    mean_compound = sum(h["sentiment"] for h in headlines) / len(headlines)
    if mean_compound >= 0.15:
        label = "Positive"
    elif mean_compound <= -0.15:
        label = "Negative"
    else:
        label = "Neutral"

    return {"headlines": headlines, "mean_compound": round(mean_compound, 3), "label": label}


if __name__ == "__main__":
    import sys
    stock = sys.argv[1].upper() if len(sys.argv) > 1 else "RELIANCE"
    result = get_news_sentiment(stock)
    print(f"News sentiment for {stock}: {result['label']} ({result['mean_compound']:+.3f})\n")
    for h in result["headlines"]:
        print(f"  [{h['sentiment']:+.2f}] {h['title']}")
