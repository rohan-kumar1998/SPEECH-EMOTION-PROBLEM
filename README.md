# SPEECH-EMOTION-PROBLEM

## Results

| Models        |Validation set          | Test set  |
| ------------- |:-------------:| -----:|
| Bidirectional GRU      | 80.52 | **82.10** |
| Bidirectional LSTM      | 83.47      |   **84.00** |
| CNN      | 83.47      |   **84.00** |
| GRU+LSTM Ensemble      | 83.47      |   **84.00** |
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

Another item
⋅⋅* Unordered sub-list. 
