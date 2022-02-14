import numpy as np
import load_data

X_train, y_train, X_test, y_test = load_data.get_data()
np.save('train_data.npy', X_train)
np.save('train_labels.npy', y_train)
np.save('test_data.npy', X_test)
np.save('test_labels.npy', y_test)
