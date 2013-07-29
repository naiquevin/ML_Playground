"""

Convert markdown documents to text and save them in the ../interview
directory if it doesn't already exist

"""

import os
import yaml
from markdown import markdown
from bs4 import BeautifulSoup
import json
import re


GIT_REPO_PATH = '/home/vineet/javascript/usesthis'

OUTPUT_FILE_PATH = 'interviews.json'

KEYTERMS_REGEX = re.compile(r'\[([^\]]+)\]')


def sep_yaml_md(lines):
    in_yaml = False
    metadata = []
    body = []

    lines = (unicode(line, 'utf-8') for line in lines)

    for line in lines:
        if line.startswith('---') and not in_yaml:
            in_yaml = True
        elif line.startswith('---') and in_yaml:
            in_yaml = False
        elif in_yaml:
            metadata.append(line)
        else:
            body.append(line)
    return ('\n'.join(metadata), '\n'.join(body))


def md_to_text(content):
    text = BeautifulSoup(markdown(content)).find_all(text=True)
    return '\n'.join(line for line in text if line not in ['', '\n'])


def parse(filepath):
    with open(filepath) as f:
        data, md = sep_yaml_md(f)
        data = yaml.load(data)
        if data is None:
            print filepath
        body = md_to_text(md)
        keyterms = [t.lower() for t in KEYTERMS_REGEX.findall(md)]
        data.update({'body': body, 'keyterms': keyterms})
    return data


def main():
    posts_path = os.path.join(GIT_REPO_PATH, 'interviews')
    posts = (parse(os.path.join(posts_path, p)) 
             for p in os.listdir(posts_path))
    posts = list(posts)
    with open(OUTPUT_FILE_PATH, 'w') as f:
        json.dump(posts, f, indent=4)


if __name__ == '__main__':
    main()

