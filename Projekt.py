import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report

def preprocess_pandas(data, columns):
    df_ = pd.DataFrame(columns=columns)
    data['Sentence'] = data['Sentence'].str.lower()
    data['Sentence'] = data['Sentence'].replace('[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', '', regex=True)                      # remove emails
    data['Sentence'] = data['Sentence'].replace('((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}', '', regex=True)    # remove IP address
    data['Sentence'] = data['Sentence'].str.replace('[^\w\s]','')                                                       # remove special characters
    data['Sentence'] = data['Sentence'].replace('\d', '', regex=True)                                                   # remove numbers
    for index, row in data.iterrows():
        word_tokens = word_tokenize(row['Sentence'])
        filtered_sent = [w for w in word_tokens if not w in stopwords.words('english')]
        df_ = df_.append({
            "index": row['index'],
            "Class": row['Class'],
            "Sentence": " ".join(filtered_sent[0:])
        }, ignore_index=True)
    return data

# If this is the primary file that is executed (ie not an import of another file)
#if False:
if __name__ == "__main__":
    # get data, pre-process and split
    data = pd.read_csv("amazon_cells_labelled.txt", delimiter='\t', header=None)
    data.columns = ['Sentence', 'Class']
    data['index'] = data.index                                          # add new column index
    columns = ['index', 'Class', 'Sentence']
    data = preprocess_pandas(data, columns)                             # pre-process
    training_data, validation_data, training_labels, validation_labels = train_test_split( # split the data into training, validation, and test splits
        data['Sentence'].values.astype('U'),
        data['Class'].values.astype('int32'),
        test_size=0.10,
        random_state=0,
        shuffle=True
    )

    # vectorize data using TFIDF and transform for PyTorch for scalability
    word_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_features=50000, max_df=0.5, use_idf=True, norm='l2')
    training_data = word_vectorizer.fit_transform(training_data)        # transform texts to sparse matrix
    training_data = training_data.todense()                             # convert to dense matrix for Pytorch
    vocab_size = len(word_vectorizer.vocabulary_)
    validation_data = word_vectorizer.transform(validation_data)
    validation_data = validation_data.todense()
    train_x_tensor = torch.from_numpy(np.array(training_data)).type(torch.FloatTensor)
    train_y_tensor = torch.from_numpy(np.array(training_labels)).long()
    validation_x_tensor = torch.from_numpy(np.array(validation_data)).type(torch.FloatTensor)
    validation_y_tensor = torch.from_numpy(np.array(validation_labels)).long()

# We create our own dataset to load the BoW embedded texts
class BoWDataset(torch.utils.data.Dataset):
    def __init__(self, sentence_vectors, labels):
        self.sentence_vectors = sentence_vectors
        self.labels = labels
        
    def __getitem__(self, index):
        return self.labels[index], self.sentence_vectors[index]
  
    def __len__(self):
        return len(self.labels)

bow_train_data = BoWDataset(train_x_tensor, train_y_tensor)
bow_test_data = BoWDataset(validation_x_tensor, validation_y_tensor)
bow_trainloader = DataLoader(bow_train_data, batch_size=300, shuffle=True)
bow_testloader = DataLoader(bow_test_data, batch_size=100, shuffle=False)

network = nn.Sequential(
    nn.Linear(vocab_size, 1000),
    nn.ReLU(),
    nn.Linear(1000, 50),
    nn.ReLU(),
    nn.Linear(50, 2)
)

optimizer = optim.Adam(network.parameters(), lr=0.1)
loss_function = nn.CrossEntropyLoss()
epochs = 30

for epoch in range(epochs):
    for batch_nr, (labels, data) in enumerate(bow_trainloader):
        prediction = network(data)
        loss = loss_function(prediction, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        #Print the epoch, batch, and loss
        print(
            f'\rEpoch {epoch+1} [{batch_nr+1}/{len(bow_trainloader)}] - Loss: {loss}',
            end=''
        )

print("\n")

while(True):
    text = input("Please give a short review of our product.\n")
    review = text + "\t" + "0" + "\n" + text + "\t" + "0"
    text_file = open("Review.txt", "w")
    text_file.write(review)
    text_file.close()

    reviewdata = pd.read_csv("Review.txt", delimiter='\t', header=None)
    reviewdata.columns = ['Sentence', 'Class']
    reviewdata['index'] = reviewdata.index
    reviewdata = preprocess_pandas(reviewdata, columns)
    review_data, unused1, unused2, unused3 = train_test_split( # split the data into training, validation, and test splits
        reviewdata['Sentence'].values.astype('U'),
        reviewdata['Class'].values.astype('int32'),
        test_size=0.00001,
        random_state=0,
        shuffle=True
    )
    review_data = word_vectorizer.transform(review_data)
    review_data = review_data.todense()
    review_x_tensor = torch.from_numpy(np.array(review_data)).type(torch.FloatTensor)

    prediction = torch.argmax(network(review_x_tensor[0]), dim=-1)

    if prediction.item() == 0:
        print("Sorry to hear that.")
    else:
        print("Wonderful!")