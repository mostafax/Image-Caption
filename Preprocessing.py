f = open("Flickr8k.token.txt",'r')
w = open("train.txt",'w')
fullText=f.read().split("\n")
for i in range(30000): #first 30000 data are for training
    fullImage = fullText[i]
    imageName=fullImage.split("#")[0]
    imageCaption=fullImage.split("	")[1]
    w.write(imageName)
    w.write("\t")
    w.write("<start>")
    w.write(imageCaption)
    w.write("<end>")
    w.write("\n")
f.close()
w.close()