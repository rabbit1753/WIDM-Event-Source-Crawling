import requests

def web_contain(url) :
    res = requests.post("http://140.115.54.45:6789/post/crawler/static/html", json={"urls": [url], "cache": False})
    result = eval(res.content.decode("utf-8"))[0]   # str
    return result
