# Gender Recognition using Voice
This project is about building a deep learning model using TensorFlow 2 to recognize gender of a given speaker's audio.

## Requirements
- TensorFlow 2.x.x
- Scikit-learn
- Numpy
- Pandas
- PyAudio
- Librosa

Installing the required libraries:

    pip3 install -r requirements.txt

## Dataset used

[Mozilla's Common Voice](https://www.kaggle.com/mozillaorg/common-voice) large dataset is used here, and some preprocessing has been performed:
- Filtered out invalid samples.
- Filtered only the samples that are labeled in `genre` field.
- Balanced the dataset so that number of female samples are equal to male.
- Used [Mel Spectrogram](https://librosa.github.io/librosa/generated/librosa.feature.melspectrogram.html) feature extraction technique to get a vector of a fixed length from each voice sample

## Training
We can customize the model in [`utils.py`](utils.py) file under the `create_model()` function and then run:

    python train.py

## Testing

[`test.py`](test.py) is the code responsible for testing the audio files or we could also use our own voice:

    python test.py --help

**Output:**

    usage: test.py [-h] [-f FILE]

    Gender recognition script, this will load the model trained, and perform
    inference on a sample provided (either using our own voice or a file)

- For instance, to get gender of the file `test-samples/27-124992-0002.wav`, we can:

      python test.py --file "test-samples/27-124992-0002.wav"

    **Output:**

      Result: male
      Probabilities:     Male: 96.36%    Female: 3.64%
  
  There are some audio samples in [test-samples](test-samples) folder to test with, it is grabbed from [LibriSpeech dataset](http://www.openslr.org/12).
- To make inference on our own voice instead, we need to:
      
      python test.py

    Wait until `"Please speak"` prompt appears and then start talking, it will stop recording when we stop speaking and then display the test results.

    
