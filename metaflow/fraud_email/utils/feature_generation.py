import polars as pl


def extract_fields(emails: pl.DataFrame) -> pl.DataFrame:
    """
    Extracts specific fields from a DataFrame containing email data and
    returns a modified DataFrame.

    Args:
        emails (pl.DataFrame): A DataFrame containing email data.

    Returns:
        pl.DataFrame: A modified DataFrame with extracted fields.
    """
    email_pattern = r"From:\s*([^<\n\s]+)"
    subject_pattern = r"Subject:\s*(.*)"
    name_email_pattern = r'From:\s*"?([^"<]+)"?\s*<([^>]+)>'

    emails = (
        emails.with_columns(
            pl.col("emails").str.extract(name_email_pattern, 2).alias("sender_email"),
            pl.col("emails").str.extract(name_email_pattern, 1).alias("sender_name"),
            pl.col("emails").str.extract(subject_pattern, 1).alias("subject"),
        )
        .with_columns(
            pl.when(pl.col("sender_email").is_null())
            .then(pl.col("emails").str.extract(email_pattern, 1))
            .otherwise(pl.col("sender_email"))
            .alias("sender_email")
        )
        .with_columns(
            pl.col("emails")
            .str.replace("Status: RO", "Status: O", literal=True)
            .str.split("Status: O")
            .arr.get(1)
            .alias("email_text")
        )
    )

    return emails


def email_features(data: pl.DataFrame, col: str) -> pl.DataFrame:
    """
    Computes additional features for a specified column in a DataFrame
    containing email data and returns a modified DataFrame.

    Args:
        data (pl.DataFrame): A DataFrame containing email data.
        col (str): The name of the column in the DataFrame to compute features for.

    Returns:
        pl.DataFrame: A modified DataFrame with additional computed features.
    """
    data = data.with_columns(
        pl.col(col).str.n_chars().alias(f"{col}_length"),
    ).with_columns(
        (pl.col(col).str.count_match(r"[A-Z]") / pl.col(f"{col}_length")).alias(
            f"{col}_percent_capital"
        ),
        (pl.col(col).str.count_match(r"[^A-Za-z ]") / pl.col(f"{col}_length")).alias(
            f"{col}_percent_digits"
        ),
    )

    return data
