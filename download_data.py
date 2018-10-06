import wget
import pandas as pd
Data_Frame = pd.read_csv('test_images.csv' , error_bad_lines=False  )
Images_link = Data_Frame['Images']

for i in range(10,len(Images_link)):
     url = Images_link[i]
     wget.download(str(url))

print(Data_Frame)
