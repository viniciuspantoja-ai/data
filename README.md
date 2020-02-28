# data
Repo for classes data

colab for house pricing prediction: https://colab.research.google.com/drive/182Rt-VT_sOy2ux3pPn2KRxVa-R-jKs0b

colab for train_test_slplit LSTM https://colab.research.google.com/drive/1O_ED6jl5FIu2Q16CI_cidh1-_QjilckV

def load_data(data, seq_len, hours_ahead):
    
    sequence_length = seq_len  + hours_ahead
    result = []
    for i in range(len(data) - sequence_length):
        result.append(data.iloc[i: i + sequence_length])
    
    result = np.array(result)
    row = round(0.75 * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-hours_ahead]
    y_train = train[:, -hours_ahead:]
    x_test = result[int(row):, :-hours_ahead]
    y_test = result[int(row):, -hours_ahead:]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  

    return [x_train, y_train, x_test, y_test]


X_train, y_train, X_test, y_test = load_data(.iloc[20000:]['Sub_metering_1'],24*7, 24)



print(X_train.shape, y_train.shape)
