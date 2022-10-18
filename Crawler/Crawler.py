from asyncio.windows_events import NULL
import requests

def web_contain(url) :
    # res = ""
    try:
        res = requests.post("http://140.115.54.45:6789/post/crawler/static/html", json={"urls": [url], "cache": False}, timeout=40)
        if len(eval(res.content.decode("utf-8"))) >= 1 or res == "":
            # print("eval decode",eval(res.content.decode("utf-8")))
            result = eval(res.content.decode("utf-8"))[0]   # str
            return result
        else:
            return False
    except:
        print("It's Timeout!")
        return False
    
