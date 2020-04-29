import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from lib.profile_abs import  Profiling
import os
import torch.utils.data as data
# Device configuration
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Recurrent neural network (many-to-one)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, model_type):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_type = model_type

        if model_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        else:
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
#        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        if self.model_type == 'lstm':
            state_vec, _ = self.rnn(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        else:
            state_vec, _ = self.rnn(x, h0)


        # Decode the hidden state of the last time step
        out = self.fc(state_vec[:, -1, :])
        # dense = self.fc(state_vec)
        return state_vec, out


'''
data : n * 28 * 28  batch_size 

'''
from os.path import expanduser
mnist_dir = os.path.join(expanduser('~') , '.torch')


def get_dt(batch_size=16, n_tr =-1, n_test=-1 ,normalize=True, sklearn_seed=123456):
    '''
        @param 
            n_tr <0 
                n_tr= 60000 
            n_test <0 
                n_test =10000
    '''
    import keras 
    (x_train,y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    index_4_7 = np.logical_or(y_train==4,y_train==7) 
    x_train = x_train [index_4_7]
    y_train = y_train [index_4_7]

    index_4_7 = np.logical_or(y_test==4,y_test==7) 
    x_test = x_test [index_4_7]
    y_test = y_test [index_4_7]

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    if n_tr >0 :
        x_train, _, y_train, _ = train_test_split(
            x_train, y_train, train_size=n_tr, test_size=0, random_state=sklearn_seed)
    else:
        n_tr= len(x_train)# full dataset

    if n_test >0 :
        x_test, _, y_test, _ = train_test_split(
            x_test, y_test, train_size=n_test, test_size=0, random_state=sklearn_seed+1)
    else:
        n_test = len(x_test)# full dataset

    # process x
    if normalize:
        x_train= x_train.reshape(x_train.shape[0],-1)
        x_test= x_test.reshape(x_test.shape[0],-1)

        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        x_train= x_train.reshape(x_train.shape[0],28,28)
        x_test= x_test.reshape(x_test.shape[0],28,28)




    num_steps= int(np.ceil(n_tr / batch_size))

    def fetch_func (seed):
        np.random.seed(seed)
        idx_list = np.array_split(np.random.permutation(n_tr), num_steps)

        for one_idx in idx_list:
            imgs,lbls= x_train[one_idx],y_train[one_idx]

            yield torch.from_numpy(imgs).float(),torch.from_numpy(lbls).long(), one_idx
    def fetch_func_test(seed=None):
        idx_list = np.array_split(np.arange(n_tr), num_steps)

        for one_idx in idx_list:
            imgs,lbls= x_train[one_idx],y_train[one_idx]

            yield torch.from_numpy(imgs).float(),torch.from_numpy(lbls).long()


    return x_train ,y_train, x_test, y_test , fetch_func ,fetch_func_test, num_steps


class TorchMnistiClassifier(Profiling):
    def __init__(self, model_type, save_dir, epoch=10):

        # # Hyper-parameters
        # sequence_length = 28
        # input_size = 28
        # hidden_size = 128
        #
        # num_layers = 2
        # num_classes = 10
        # batch_size = 100
        # num_epochs = 2


        super().__init__(save_dir)
        self.time_steps = 28  # timesteps to unroll
        self.n_units = 128  # hidden LSTM units
        self.n_inputs = 28  # rows of 28 pixels (an mnist img is 28x28)
        self.n_classes = 10  # mnist classes/labels (0-9)
        self.batch_size = 128  # Size of each batch
        self.num_layers = 1 # Number of recurrent layers.
        self.channel = 1
        self.n_epochs = epoch
        self.learning_rate = 0.01



        # Internal
        self._data_loaded = False
        self._trained = False
        self._model_type = model_type

        self.model_path = os.path.join(self.model_dir, model_type + '_' + str(epoch) + '.ckpt')
        self.model = RNN(self.n_inputs, self.n_units, self.num_layers, self.n_classes, self._model_type).to(device)
        if os.path.exists(self.model_path):
            print('Load existing model')
            self.model.load_state_dict( torch.load(os.path.join(self.model_path)))
        else:
            print('Training...')
            print(device)
            self.train()

    def train(self):
        # MNIST dataset

        x_train ,y_train,x_test,y_test, fetch_func ,test_fetch_func, num_steps = get_dt(batch_size=self.batch_size)


        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()

        # Train the model
        total_step = num_steps#len(train_loader)

        #total_step = len(train_loader)
        for epoch in range(self.n_epochs):
            epoch += 1
            for i , (images,labels,_) in enumerate(fetch_func(seed=epoch)):
            #for i, (images, labels) in enumerate(train_loader):
                images = images.reshape(-1, self.time_steps, self.n_inputs).to(device)
                labels = labels.to(device)

                # Forward pass
                _, outputs = self.model(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i + 1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(epoch + 1, self.n_epochs, i + 1, total_step, loss.item()))
            if epoch % 5 == 0 or epoch == self.n_epochs:
                filepath = self.model_dir + '/' + self._model_type + '_%d.ckpt'%(epoch)
                torch.save(self.model.state_dict(), filepath)

            # Test the model
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in test_fetch_func(seed=epoch):
                #for images, labels in test_loader:
                    images = images.reshape(-1, self.time_steps, self.n_inputs).to(device)
                    labels = labels.to(device)
                    _, outputs = self.model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

        # torch.save(self.model.state_dict(), self.model_path)

    def do_profile(self):
        '''
        train_dataset = torchvision.datasets.MNIST(root=mnist_dir,

                                                   train=True,
                                                   transform=transforms.ToTensor(),
                                                   download=True)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=True)
        '''
        x_train ,y_train,x_test,y_test, fetch_func ,test_fetch_func, num_steps = get_dt(batch_size=self.batch_size)
        #return self.predict_loader(train_loader)
        return self.predict_loader(fetch_func)


        # (x_train, y_train), (_, _) = mnist.load_data()
        # return self.predict(x_train)



    def preprocess(self, x_test):
        # x_test = x_test.reshape(x_test.shape[0], self.time_steps, self.n_inputs, self.channel)
        x_test = x_test.astype('float32')
        x_test /= 255
        return x_test
    def predict_loader(self, data_loader):
        with torch.no_grad():
            t1,t2 = [],[]
            for images, labels in data_loader:
                images = images.reshape(-1, self.time_steps, self.n_inputs).to(device)

                state_vec, _ = self.model(images)
                dense = self.model.fc(state_vec)
                softmax = torch.nn.functional.softmax(dense, dim=-1)
                t1.append(softmax.cpu().numpy())
                t2.append(state_vec.cpu().numpy())

        return np.vstack(t1), np.vstack(t2),
    def predict(self, np_data): # n * 28 * 28
        pre_data = self.preprocess(np_data)
        data = torch.from_numpy(pre_data)

        # Test the model
        with torch.no_grad():
            images = data.reshape(-1, self.time_steps, self.n_inputs).to(device)
            state_vec, _ = self.model(images)
            dense = self.model.fc(state_vec)
            # print(dense.shape)
            # _,
            softmax = torch.nn.functional.softmax(dense,dim=-1)
            # _, predicted = torch.max(softmax, dim=-1)
            return  softmax.numpy(), state_vec.numpy() #predicted.numpy(), state_vec.numpy(), softmax.numpy()




if __name__ == "__main__":
    # print(os.path.dirname('a/b/c/a'))
    classifier = TorchMnistiClassifier(model_type='lstm', save_dir='../../data/mnist_rnn_torch')

    '''
    test_dataset = torchvision.datasets.MNIST(root=mnist_dir,
                                              train=False,
                                              transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=100,
                                              shuffle=False)
    '''
    x_train ,y_train,x_test,y_test, fetch_func ,test_fetch_func, num_steps = get_dt()
    test_loader = test_fetch_func()

    y = []
    x = []
    with torch.no_grad():
        correct = 0
        total = 0

        for images, labels in test_loader:
            images = images.reshape(-1, classifier.time_steps, classifier.n_inputs).to(device)
            labels = labels.to(device)
            _, outputs = classifier.model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y.append(predicted.cpu().numpy())
            x.append(labels.cpu().numpy())



        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
    from keras.datasets import mnist
    from torchsummary import summary
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    #
    # # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    outputs = classifier.predict_np(x_test)
    res = outputs[0][:, -1]
    #
    # x0 = np.concatenate(x)
    # y0 = np.concatenate(y)
    same = np.sum(res == y_test)
    # correct = np.sum(x == y0)
    #

    print(same)

