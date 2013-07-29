import requests
import json
import sys


GH_API_URL = 'https://api.github.com'

STREAM = sys.stderr


def get_repos(page, keyword, repos):
    if page == 0:
        STREAM.write('\n')
        return repos
    else:
        STREAM.write('{page}..'.format(page=page))
        url = (
            '{api_url}/legacy/repos/search/{keyword}?'
            'start_page={start_page}'
        ).format(api_url=GH_API_URL, keyword=keyword, start_page=page)
        req = requests.get(url)
        if req.status_code == 200:
            return get_repos(page - 1, keyword, req.json()['repositories'] + repos)


if __name__ == '__main__':
    script, lang = sys.argv
    repos = get_repos(10, lang, [])
    print json.dumps(repos)
