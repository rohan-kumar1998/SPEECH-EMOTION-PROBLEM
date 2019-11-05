# SPEECH-EMOTION-PROBLEM

## Results

| Models        |Validation set          | Test set  |
| ------------- |:-------------:| -----:|
| Bidirectional GRU      | 15.42 | 33.19 |
| Bidirectional LSTM      | 6.26      |   31.59 |
| CNN      | 83.47      |   **84.00** |
| GRU+LSTM Ensemble      | 41.92      |   32.55 |
| GRU+CNN Ensemble  | **84.31**      |    83.57 |


## Features 
For each .wav file, a mfcc (Mel-frequency cepstral coefficients) matrix was extracted using the implementation present in torchaudio library. A log (base 2) operator was applied element wise on the mfcc matrix. The nan values corresponding to the negative elements in the matrix were replaced with 0. 

The output dimension of mfcc function in torchaudio library is `[num_channels, n_mfcc, time]` here, number of mfcc features extracted or n_mfcc was taken as 40, while on observation the num_channels was more than 1 (observed was 2). It was then reshaped as `[n_mfcc,time*num_channels]` to be used in the models. 

## Dataset 
The dataset used was Multimodal Emotionlines Dataset (MELD) Poria S. et Al. The dataset contains instances of dialogue utterances from the tv show 'Friends'.

The statistics of dataset provided was - 

| Dataset   |Disgust | Fear | Happy | Neutral | Sad | Total |           
| ------------- |:-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
|Training set      | 232 | 216 | 1609 | 4592 | 705 | 7354 | 
|Validation set     | 28 | 25 | 181 | 517 | 79 |  830 

Some issues faced in the dataset
 *  The dataset is severly unbalanced with roughly **62%** data being from a single class 
 *  Due the size of dataset being huge, and the length of feature matrix being `n_mfcc*time`, technical limitations came into play for loading the dataset onto the gpu, due to limitations on gpu memory and ram. 
 
 To solve this issue an upper limit of 1000 was imposed onto the data set, and a new set **Test set** was created by partitioning the new training set 70:30. 
 
 Hence the new size of training set was 2207 and the size of new test set created was 946. Solving the second problem and somewhat solving the first one too. 
 
 A Training was planned with using only training and val set (i.e before partitioning) but wasn't completed due to restrictions in time. 
 
## Model Architecture 
 A set of 5 models were used to experiment on the dataset. Trying to determine what the one model is learning which the other model is ignoring, and ensembling them to create a model to captures them both. This was done to determine which model is best suited for the emotion detection task. 

### Bidirectional GRU and LSTM Models 
A standard implementation of GRU and LSTM models were done on Pytorch Paszke et Al. and as mentioned in the second issue in the dataset subsection, the sequence or dependency length was much too high, and hence these models were used as an experiment based on their varying properties on dependency lengths. 

#### GRU vs LSTM 
Due to the addition of a few extra gates in LSTM the number of trainable parameters significantly increase, making the training process fairly slow, hence the GRU model could achieve a better accuracy in limited epoch training.  

Even though the LSTM are often considered a more powerful network than GRU, GRUs have been seen to outperform LSTMs on a shorter dependency. 

### CNN model
The major difference between a CNN and a RNN model is the ability of RNN to process the data sequentially, while CNN takes all the data at once. Even though on paper it sounds as if RNN is more human-like in it's data processing, CNN are fairly common in the field of text classification Kim Y. et Al., and hence this model was used in the experiments as well. 

### GRU+CNN Ensemble 

### GRU+LSTM Ensemble 

## Experiment Setup 
To maintain a set of uniformity, similar hyperparamters and number of epochs were chosen. Due the lack of time, the number of epochs were limited to 15, and the best performing model (i.e the weights corresponding to minimum validation loss) was saved. Adam optimizer Kingma et Al. was used to update weights, based on a loss calculated by BCEwithlogitsloss in Pytorch, which is the implementation of binary cross-entropy loss + sigmoid. Batch size was taken as 10 and accuracy was used as the metric for model evaluation. The learning rate was fixed as 1e-5 for every model.

 ## Conclusion
 A few reasons could be pointed out for not achieving a perfect accuracy - 
  *  The dataset even after augmentation was still unbalanced, 216 datapoints in the least dense class (Fear) vs 1000 datapoints in the most dense class (Neutral).
  *  The data, even though classified into different classes were similar in nature as tested via human-evaluation. The similarity lied in the recorded laughter present in the audio clips. 
  *  Even though not confirmed, there could be noise present in the data as well.
  
In the future a speaker separation model can be used to filter out the laughter, to further boost the accuracy, or a model could be trained to isolate the semantics of the spoken words, which would also contain the nature of emotion the speaker is in. 
 

 ## References 
 Poria, S., Hazarika, D., Majumder, N., Naik, G., Cambria, E., & Mihalcea, R. (2018). Meld: A multimodal multi-party dataset for emotion recognition in conversations. arXiv preprint arXiv:1810.02508.
 
 Paszke, Adam, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. "Automatic differentiation in pytorch." (2017).
 
 Kim, Yoon. "Convolutional neural networks for sentence classification." arXiv preprint arXiv:1408.5882 (2014).
 
 Kingma, Diederik P., and Jimmy Ba. "Adam: A method for stochastic optimization." arXiv preprint arXiv:1412.6980 (2014).

 
