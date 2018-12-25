# Image-caption Using End2end CNN , LSTM based Model!
 The aim of the project is to generate a caption for images.
 Each image has a story, Image Captioning narrates it.
<img src = "/PretrainedModel/Out.png">

 
 this model is bases on [Show and Tell: A Neural Image Caption Generator
](https://arxiv.org/pdf/1411.4555.pdf)

📖 Documentation
================
## How to Run
**Install the requirements:**
```bash
pip3 install -r requirements.txt 
```
**Running the Model**
```bash
python3 model.py
```

## Results

The results are not bad at all! a lot of test cases gonna be so realistic, but the model still needs more training
<img src = "/PretrainedModel/r1.png">
<img src = "/PretrainedModel/r2.png">
## Paper
This project is an implementation of the [Show and Tell](https://arxiv.org/pdf/1411.4555.pdf), published 2015.

## Dataset
- Dataset used is Flicker8k each image have 5 captions.
- you can request data from here [Flicker8k]
(https://forms.illinois.edu/sec/1713398).
**Sample of the data used**
<img src = "/PretrainedModel/dayaset.png">
## Model Used
<img src = "/PretrainedModel/model.png">

## Experiments

<img src = "/PretrainedModel/expermant.png">

## Future Work
-Training, Training and more Training<br>
-Using Resnet instead of VGG16<br>
-Creating API for production level <br>
-Using Word2Vec embedding.

