import csv
url_dic = {}
with open('TESW.csv', newline='') as csvfile:
  rows = csv.reader(csvfile)

  for row in rows :
    if row[0] in url_dic :
      url_dic[row[0]] = url_dic[row[0]] + ',' +row[1]
    else :
      url_dic[row[0]] = row[1]

def dis_URL( seed , URL) :
  if URL in url_dic[seed] and URL!= seed:
    return True
  else : 
    return False
