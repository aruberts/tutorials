def load_emails_txt(path: str, split_str: str = "From r  ") -> list[str]:
    """
    Loads emails from a text file and returns them as a list.

    Args:
        path (str): The file path to the text file.
        split_str (str, optional): The string used to split the text file into
                                   individual emails. Defaults to "From r  ".

    Returns:
        list[str]: A list of emails extracted from the text file.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        text = file.read()

    emails = text.split(split_str)

    return emails
