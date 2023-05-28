import polars as pl


def email_clean(
    data: pl.DataFrame, col: str, new_col_name: str | None = None
) -> pl.DataFrame:
    """
    Cleans and preprocesses the text in a specified column of a DataFrame containing
    email data, and returns a modified DataFrame.

    Args:
        data (pl.DataFrame): A DataFrame containing email data.
        col (str): The name of the column in the DataFrame to clean and preprocess.
        new_col_name (str | None, optional): The name for the new column with cleaned data. Defaults to None.

    Returns:
        pl.DataFrame: A modified DataFrame with the cleaned and preprocessed text.

    """
    data = data.with_columns(
        pl.col(col)
        .str.replace_all(r"<.*?>", " ")
        .str.replace_all(r"[^a-zA-Z\s]+", " ")
        .str.replace_all(r"\s+", " ")
        .str.to_lowercase()
        .alias(new_col_name if new_col_name is not None else col)
    )

    return data


def tokenise_text(data: pl.DataFrame, col: str, split_token: str = " ") -> pl.DataFrame:
    """
    Tokenizes the text in a specified column of a DataFrame containing email data and returns a modified DataFrame.

    Args:
        data (pl.DataFrame): A DataFrame containing email data.
        col (str): The name of the column in the DataFrame to tokenize.
        split_token (str, optional): The token used to split the text into tokens. Defaults to " ".

    Returns:
        pl.DataFrame: A modified DataFrame with tokenized text.
    """
    data = data.with_columns(
        pl.col(col).str.split(split_token).alias(f"{col}_tokenised")
    )

    return data


def remove_stopwords(
    data: pl.DataFrame, stopwords: set | list, col: str
) -> pl.DataFrame:
    """Removes stopwords from the text in a specified column of a DataFrame containing email data and returns a modified DataFrame.

    Args:
        data (pl.DataFrame): A DataFrame containing email data.
        stopwords (set | list): A set or list of stopwords to be removed from the text.
        col (str): The name of the column in the DataFrame to remove stopwords from.

    Returns:
        pl.DataFrame: A modified DataFrame with stopwords removed from the text.
    """
    data = data.with_columns(
        pl.col(col)
        .arr.eval(
            pl.when(
                (~pl.element().is_in(stopwords)) & (pl.element().str.n_chars() > 2)
            ).then(pl.element())
        )
        .arr.eval(pl.element().drop_nulls())
    )
    return data
