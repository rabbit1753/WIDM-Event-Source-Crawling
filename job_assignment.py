# from asyncio.windows_events import NULL
import csv
import pandas as pd

def job_assignment(count):
    seed = pd.read_csv('train.csv')
    link = seed['Event Message Page URL']
    if count >= len(link):
        print("no seed to crawling")
        return 0
    return link[count]

if __name__ == '__main__': 
    for i in range(400):
        link = job_assignment(i)
        if link == 0:
            break
        print(link)