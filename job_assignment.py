seed = ["http://careernthu.conf.asia/",
            "http://fishbar.com.tw",
            "https://imc.ichiayi.com"]

def job_assignment():
    link = seed[0]
    del seed[0]
    return link

if __name__ == '__main__': 
    for i in range(len(seed)):
        link = job_assignment()
        print(link)