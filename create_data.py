import os
import numpy as np
import pandas as pd
FAKE_PATH = "StoryText 2\\Fake\\finalFake"
SATIRE_PATH = "StoryText 2\\Satire\\finalSatire"
FAKE = 1
SATIRE = 0


def read_text_file(data, file_path, is_fake):
    with open(file_path, 'r', encoding="ISO-8859-1") as f:
        lines = f.readlines()
    parse_article(data, lines, is_fake)


def parse_article(data, lines, is_fake):
    lst = []
    body = ""
    count = 0
    for line in lines:
        if count == 0:
            if len(line.strip()) > 5:
                lst.append(line.strip())
            else:
                lst.append("no_title")
        elif count == 1:
            if "http" in line or "com" in line:
                lst.append(line.strip())
            else:
                lst.append("no_url")
        elif count > 1:
            body += line.strip()
        count += 1
    lst.append(body)
    lst.append(is_fake)
    data.append(lst)


def read_txt_files(data, path, is_fake):
    for file in os.listdir(path):
        if file.endswith(".txt"):
            file_path = f"{path}\{file}"
            read_text_file(data, file_path, is_fake)


def create_df():
    data = []
    read_txt_files(data, FAKE_PATH, FAKE)
    read_txt_files(data, SATIRE_PATH, SATIRE)
    df = pd.DataFrame(data, columns=['title', 'url', 'body', 'fake'])
    df.to_csv("news.csv")

create_df()