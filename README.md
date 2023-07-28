# IMAGE CAPTIONING with VGG16 and GRU model (RNN) 🚙🧑‍🤝‍🧑🐶

### What is image captioning?
Image Captioning is the task of describing the content of an image in words. This task lies at the intersection of computer vision and natural language processing. Usually, this is done by placing an encoder that extract image features on top of a decoder that has the goal to generate a descriptive text sequence.

### COCO 2014 Dataset
COCO is a large-scale object detection, segmentation, and captioning dataset.
* Data are stored in JSON like files;
* Information can be accessed via API;
* Train, validation and test splits are available;


### Preprocessing - 1/2
For developing our project, we only used the training split offered by COCO. In total, we have 83.000 images ca. and we will assign (roughly):
* 70% to the training set;
* 20% to the valisation set;
* 10% to the test set.
### Preprocessing - 2/2
*Image Preprocessing*: dataset of tensors preprocessed ➡️ RGB color, Size 224x224, VGG16 preprocess input.\
*Caption Preprocessing*: vocabulary of words ➡️ Alphanumeric characters, Start/end tokens, Lowercase, Tokenization.

### Implemented solution
1. Encoder - CNN with VGG16
2. Decoder - RNN model with GRU layers

#### CNN: VGG16
![image](https://github.com/stella-cervini/DeepLearning/assets/73294073/dd99f149-aced-49da-a541-8053ef36637b)
Credit: https://neurohive.io/en/popular-networks/vgg16 \
Softmax layer is used for classification problem (not used here). We used this pretrained model to extract image features.
* Our output dim.: (None, 4096)
* Convolutional blocks
* Max Pooling
* Flatten
* Dense

#### RNN model with GRU Layers
Gated recurrent units (GRUs) are a type of Recurrent Neural Network (RNN) and are similar to a long short-term memory (LSTM) with a forget gate.\
Gru has fewer parameters than LSTM, as it lacks an output gate. GRU's performance on certain tasks, like natural language processing was found to be similar to that of LSTM, so they are considered to be a variation on the LSTM.
![image](https://github.com/stella-cervini/DeepLearning/assets/73294073/a645f6d8-e609-4d9b-abdb-87821e765ebd)
Credit: https://www.researchgate.net/figure/Gated-Recurrent-Unit-GRU_fig4_328462205

### Results
![tennis](https://github.com/stella-cervini/DeepLearning/assets/73294073/ddc92ce5-79bd-4d98-b3a5-dab1192ef0ef)

**Predicted caption**: a crowd of people watching a tennis ball [end]\
**Real Caption**: [start] the men are in the middle of a tennis match [end]\
**Bleu score**: 0.1218\
The *Bilingual Evaluation Understudy Score*, or BLEU for short, is a metric for evaluating a
generated sentence to a reference sentence.\
A perfect match results in a score of 1.0, whereas a perfect mismatch results in a score of 0.0.\

### Comments on the results
1. Overall, the model is able to identify the main subjects in the image and generate a meaningful caption. Yet, in some cases they are entirely wrong.
2. Training the model with a large amount of data is computationally expensive and lead to really high xecution times.
3. However, as we can see from the BLEU score, the generated caption is not optimal. In fact, usually it is not higher that 0.2.

### Example of Negative results
![bears](https://github.com/stella-cervini/DeepLearning/assets/73294073/64c8eea0-e52e-43fe-9180-4631f20bf50f)
**Predicted caption**: a dog with a black umbrella in its back [end]\
**Real Caption**: [start] Two cutouts of baby black bears beside a lake [end]\
**Bleu score**: 0.108\
A BLEU score is considered acceptable if it is greater than 0.4. In this case, we settled for a subjective evaluation as, generally, the caption was consistent with the image.

### Failed experiments
***Q&A to our final implementation***\
1. *Why don't use a different CNN?* We used the VGG16. The Resnet50 returned slightly worse results despite taking more time in the execution.
2. *Why didn't change the number of GRU layers?* Less layers did not provide better results while more layers take too much time.
3. *What about the dropout layers?* They are useful to not overfit the model. We did not use the dropout parameter in the GRU layer due to difficulty in interpreting the internal state.
4. *Why don't use LSTM?* In the overall implementation, GRUs fitted better, since LSTMs required a specific input that was difficult to deal with due to dimension contraints.

### Future developments
We are aware of the fact that the results are not perfect. To obtain better results, transformers and the attention mechanism can be implemented.
