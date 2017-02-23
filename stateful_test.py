import numpy as np
from keras.layers import Dense, Input, merge, Activation, Dropout, Flatten
from keras.models import Model
from keras.layers.recurrent import GRU

# Model
i = Input(batch_shape=(1, 1, 1))
x = GRU(1, stateful=True)(i)
x = Activation('sigmoid')(x)
model = Model(i, x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Data
N_train = 100
max_len = 100
one_indexes = np.random.choice(a=N_train, size=N_train // 2, replace=False)

X_train = np.zeros((N_train, max_len, 1))
Y_train = np.zeros((N_train, 1))

# Set first index in sequence to 1, everything to zero
# We want the last output of the GRU to be 1.
X_train[one_indexes, 0] = 1
Y_train[one_indexes] = 1

print('Train...')
for epoch in range(15):
    mean_tr_acc = []
    mean_tr_loss = []
    for i in range(len(X_train)):
        y_true = Y_train[i]
        x_seq = X_train[i]
        for feature in x_seq:
            tr_loss, tr_acc = model.train_on_batch(
                np.array(feature).reshape([1, 1, 1]),
                np.array(y_true)
            )
            mean_tr_acc.append(tr_acc)
            mean_tr_loss.append(tr_loss)
        model.reset_states()

    print('accuracy training = {}'.format(np.mean(mean_tr_acc)))
    print('loss training = {}'.format(np.mean(mean_tr_loss)))
    print('___________________________________')
