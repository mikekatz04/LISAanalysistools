# -*- coding: utf-8 -*-
import os.path
from tarfile import open as opentar

import gdown as gdd

os.makedirs("tutorial/data/", exist_ok=True)

# download MBH
if not os.path.exists("tutorial/data/mbh"):
    url = "https://drive.google.com/file/d/1Rr3pEF2BA2oRfs23FDIgMgoutxTq1qOR/view?usp=share_link"  # noqa: E501
    output_path = "tutorial/data/MBHDemo.tar"
    gdd.download(url, output_path, quiet=False, fuzzy=True)
    with opentar("tutorial/data/MBHDemo.tar", "r") as tar:
        tar.extractall(
            "tutorial/data/mbh"
        )  # specify which folder to extract to
    os.remove("tutorial/data/MBHDemo.tar")


# download UCB
if not os.path.exists("tutorial/data/ucb"):
    # 06 mo
    url = "https://drive.google.com/file/d/1z4M2cZnF5mPsS6ppNsHtceqXfR9Liz99/view?usp=share_link"  # noqa: E501
    output_path = "tutorial/data/UCBDemo_06.tar"
    gdd.download(url, output_path, quiet=False, fuzzy=True)
    with opentar("tutorial/data/UCBDemo_06.tar", "r") as tar:
        tar.extractall(
            "tutorial/data/ucb"
        )  # specify which folder to extract to
    os.remove("tutorial/data/UCBDemo_06.tar")

    # 03 mo
    url = "https://drive.google.com/file/d/1XGYUETwupZe8AiEDd4OMAIzA7kukDh5e/view?usp=share_link"  # noqa: E501
    output_path = "tutorial/data/UCBDemo_03.tar"
    gdd.download(url, output_path, quiet=False, fuzzy=True)
    with opentar("tutorial/data/UCBDemo_03.tar", "r") as tar:
        tar.extractall(
            "tutorial/data/ucb"
        )  # specify which folder to extract to
    os.remove("tutorial/data/UCBDemo_03.tar")
