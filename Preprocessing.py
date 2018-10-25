def createfiles():  # create output files with the desired format of data
    tokenFile = open("Flickr8k.token.txt",'r')
    trainOutputFile = open("train.txt",'w')
    devOutputFile = open("dev.txt",'w')
    testOutputFile = open("test.txt",'w')
    fullText=tokenFile.read().split("\n")  # list of all raw data
    for i in range(40000):  # data contains 8000 photo each has 5 captions
        fullImage = fullText[i]
        imageName=fullImage.split("#")[0]
        imageCaption=fullImage.split("	")[1]
        if i < 30000:  # first 6000*5 image for train
            trainOutputFile.write(imageName +"\t<start>"+imageCaption+"<end>\n")
        elif i<35000: # next 1000*5 image for dev
            devOutputFile.write(imageName +"\t<start>"+imageCaption+"<end>\n")
        else:  # next 1000*5 image for test
            testOutputFile.write(imageName + "\t<start>" + imageCaption + "<end>\n")
    tokenFile.close()
    trainOutputFile.close()
    devOutputFile.close()
    testOutputFile.close()
    return

createfiles()