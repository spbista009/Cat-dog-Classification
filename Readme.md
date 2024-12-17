
# Cat-Dog Classification

This repository contains a project focused on building a machine learning model to classify images of cats and dogs. The project leverages deep learning techniques to achieve accurate predictions. Below, you'll find details about the project, setup instructions, and usage guidelines.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Setup and Installation](#setup-and-installation)
- [Training the Model](#training-the-model)
- [Usage](#usage)
- [Contributing](#contributing)


---

## Project Overview

The goal of this project is to classify images of cats and dogs using a Convolutional Neural Network (CNN). The key features of the project include:
- Use of a labeled dataset of cat and dog images for training.
- Implementation of data preprocessing techniques such as resizing, normalization, and augmentation.
- Model built using a popular deep learning framework  TensorFlow/Keras o.
- Evaluation of the model's performance with metrics like accuracy and loss.

---

## Dataset

The dataset used for this project consists of images of cats and dogs, from Kaggle. Ensure the dataset is downloaded and organized as follows:

link to datasets: https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/data 

```
/dataset
  /train
    /cats
      cat1.jpg
      cat2.jpg
      ...
    /dogs
      dog1.jpg
      dog2.jpg
      ...
  /test
    /cats
      cat_test1.jpg
      cat_test2.jpg
      ...
    /dogs
      dog_test1.jpg
      dog_test2.jpg
      ...
```

---

## Dependencies

The project requires the following Python libraries:

- Python 3.7+
- TensorFlow 
- NumPy
- Pandas
- Matplotlib
- pickle
- os

Install the dependencies using:

```bash
pip install -r requirements.txt
```

---

## Setup and Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/spbista009/Cat-dog-Classification.git
   ```

2. Navigate to the code directory:
   ```bash
   cd Code
   ```

3. Ensure the dataset is placed in the appropriate directory structure as mentioned above.

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Training the Model, Model Evaluation and Saving

Run the following command to train the model:

```bash
python CatDog.py
```



## Contributing

Contributions are welcome! If you would like to contribute to this project:
1. Fork the repository.
2. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes and push them to your forked repository.
4. Submit a pull request.

---

