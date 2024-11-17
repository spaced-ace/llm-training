import json


def remove_trailing_headers(sections: list[str]) -> list[str]:
    for i in range(len(sections)-1,0,-1):
        if sections[i].startswith('#'):
            sections.pop(i)
        else:
            break
    return sections

def unifiy_headers(sections: list[str]) -> list[str]:
    res = []
    began_unification = False
    for sec in sections:
        header = sec.startswith('#')
        if not header and not began_unification:
            res.append(sec)
        elif header and not began_unification:
            res.append(sec)
            began_unification = True
        elif header and began_unification:
            res[-1] += '\n' + sec
        elif not header and began_unification:
            res[-1] += '\n' + sec
            began_unification = False
        else:
            raise ValueError('I am a bad programmer')
    return res

if __name__ == '__main__':
    with open('important_articles_cleaned.json', 'r') as f:
        articles = json.load(f)

    cleaned = {}
    for article in articles:
        article['sections_en'] = remove_trailing_headers(article['sections_en'])
        article['sections_en'] = unifiy_headers(article['sections_en'])
        article['sections_hu'] = remove_trailing_headers(article['sections_hu'])
        article['sections_hu'] = unifiy_headers(article['sections_hu'])
        cleaned[article['url']] = article['sections_en']
        cleaned[article['url_hu']] = article['sections_hu']

    with open('important_articles_cleaned_unified.json', 'w') as f:
        json.dump(cleaned, f, indent=2)
