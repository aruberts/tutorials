import matplotlib.pyplot as plt
from wordcloud import WordCloud


def generate_word_cloud(text: str):
    """
    Generate and display a word cloud image based on the provided text.

    Args:
        text (str): The input text to generate the word cloud from.

    Returns:
        None
    """
    # Generate a word cloud image
    wordcloud = WordCloud(
        max_words=100, background_color="white", width=1600, height=800
    ).generate(text)

    plt.figure(figsize=(20, 10), facecolor="k")
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()
