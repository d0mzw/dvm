import os
import sys

sys.path.append(os.curdir)
from pelicanconf import *  # noqa: E402, F403

# GitHub Actions overrides this with the Pages URL; used for local production builds.
SITEURL = "https://weirdmachine.wtf"
RELATIVE_URLS = False

FEED_ALL_ATOM = "feeds/all.atom.xml"

DELETE_OUTPUT_DIRECTORY = True
