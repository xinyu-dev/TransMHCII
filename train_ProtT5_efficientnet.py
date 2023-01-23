import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.data import Dataset
import pickle


def get_dataset_partitions(ds, ds_size, train_split=0.8, val_split=0.1, 
                           include_test = True, test_split=0.1, seed = 12345):
    """
    Generic function to shuffle and partition a tf.data.Dataset object into train, val, test
    :param ds: tf.data.Dataset
    :param ds_size: int, size of the dataset (i.e. # of data points)
    :param train_split: float, percentage of the training data. 
    :param val_split: float, percentage of the validation data
    :param include_test: bool, whether to include the test partition.
    :param test_split: float, percentage of test data. Ignored when include_test if False
    return tuple of tf.data.Dataset objects: (train, val) or (train, val, test)
    """
    
    # check
    if include_test:
        assert (train_split + test_split + val_split) == 1
    else:
        assert (train_split + val_split) == 1
    
    # Specify seed to always have the same split distribution between runs
    ds = ds.shuffle(ds_size, seed=seed)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    # split train, val, test
    if include_test:
        train_ds = ds.take(train_size)    
        val_ds = ds.skip(train_size).take(val_size)
        test_ds = ds.skip(train_size).skip(val_size)
        return train_ds, val_ds, test_ds
    
    # split train, val
    else:
        train_ds = ds.take(train_size)
        val_ds = ds.skip(train_size)
        return train_ds, val_ds
    

def load_data(csv_fp, features_fp, batch_size = 256):
    """
    Function to load data
    :param csv_fp: str, path to the input CSV file
    :param features_fp: str, path to the PLM embedded pickle file
    :param batch_size: int, batch size
    :return: tensorflow dataset train_data, tensorflow dataset val_data, shape of input1 (PLM embedded features), shape of input2 (alelles), shape of label, number of unique alleles
    """

    df = pd.read_csv(csv_fp)

    # convert labels
    labels = tf.one_hot(df.cat_multi, depth = len(np.unique(df.cat_multi)))

    # load features from PLM embedding
    esm = pickle.load(open(features_fp, 'rb'))
    # add color dim so that COnv2D works
    esm = np.expand_dims(esm, axis = 3)

    # load alleles
    a_encoded = np.array(df.a_encoded).reshape(-1, 1)
    n_a = len(np.unique(df.a_encoded))
    
    # data
    data = Dataset.from_tensor_slices((
        {
            'input1': tf.convert_to_tensor(esm), # input1: PLM embeddings
            'input2': tf.convert_to_tensor(a_encoded) # input2: integer labels
        },
         labels
    ))
    
    # get partition
    train_data, val_data = get_dataset_partitions(
    ds = data, 
    ds_size = labels.shape[0], 
    train_split = 0.85, 
    val_split = 0.15, 
    include_test = False
)
    
    # shuffle the train data every epoch
    train_data = train_data.shuffle(len(train_data), reshuffle_each_iteration=True)

    # batch all data
    batch_size = batch_size
    train_data = train_data.batch(batch_size = batch_size)
    val_data  = val_data.batch(batch_size = batch_size)
    
    # check shape
    for input_dict, label in train_data:
        input1_shape = input_dict['input1'].shape
        input2_shape = input_dict['input2'].shape
        label_shape = label.shape
        break
        
    return train_data, val_data, input1_shape, input2_shape, label_shape, n_a

if __name__ == '__main__':
    
    # load data
    print('Loading data...')
    train_data, val_data, input1_shape, input2_shape, label_shape, n_a = load_data(
        csv_fp = 'data/input_with_features.csv', 
        features_fp = 'embedded/prot_t5_xl_uniref50.pkl', 
        batch_size = 128
    )
    
    print('Data loaded. Compiling model....')

    # build & compile. Use distributed straining.
    strategy = tf.distribute.MirroredStrategy(devices=["/gpu:1", "/gpu:2", "/gpu:3"]) # add any available GPU here
    with strategy.scope():
        # base
        base = keras.applications.EfficientNetV2B0(
            include_top=False,
            weights="imagenet",
            input_shape= (32, 1024, 3), # (32, feature, 3)
            pooling='avg',
            include_preprocessing=False,
        )
        
        base.trainable = True
        
        input1 = keras.Input(shape=input1_shape[1:], name = 'input1')
        x1 = tf.image.resize(input1, (32, 1024))
        x1 = layers.Conv2D(filters=3, strides = 1, padding = 'same', kernel_size = 3)(x1)
        x1 = base(x1)

        input2 = keras.Input(shape = input2_shape[1:], name = 'input2')
        x2 = keras.layers.Embedding(input_dim = n_a, output_dim = 128)(input2)
        x2 = keras.layers.GlobalAveragePooling1D()(x2)
        x2 = keras.layers.Dense(64, activation = 'relu', kernel_initializer = 'he_uniform')(x2)

        x1 = layers.Concatenate()([x1, x2])
        x1 = layers.Dense(128, kernel_initializer = 'he_uniform', activation='relu')(x1) 
        x1 = layers.Dropout(0.2)(x1)
        outputs = layers.Dense(4, kernel_initializer = 'he_uniform', activation='softmax')(x1)

        model = keras.Model(
                inputs = [input1, input2],
                outputs = [outputs]
            )

        model.compile(
            optimizer= keras.optimizers.Adam(learning_rate = 1e-5) ,
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=[tf.keras.metrics.CategoricalAccuracy()]
        )
    
    def linear_warmup_scheduler(epoch, lr,
                            stage1_start_lr = 1e-5,
                            stage1_end_lr = 1e-3,
                            stage1_epochs = 10,
                            stage2_end_lr = 1e-3,
                            stage2_epochs = 20,
                            stage3_epochs = 70,
                            alpha = 0
                           ):
        # stage 1
        if epoch < stage1_epochs:
            delta = (stage1_end_lr - stage1_start_lr)/stage1_epochs
            return lr + delta

        # stage 2
        if epoch < stage2_epochs + stage1_epochs:
            delta = (stage2_end_lr - stage1_end_lr)/stage2_epochs
            return lr + delta

        # stage 3
        else:
            n = epoch - (stage1_epochs + stage2_epochs)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * (n) / stage3_epochs))
            delta = (1 - alpha) * cosine_decay + alpha
            return stage2_end_lr * delta

    callbacks = [
        keras.callbacks.ModelCheckpoint("models/ProtT5_Uniref/model_ProtT5_efficientnet_v2b0.h5", save_best_only=True, monitor='val_categorical_accuracy', mode = 'max'),
        keras.callbacks.EarlyStopping(patience = 10, min_delta=0.0001, restore_best_weights = True),
        keras.callbacks.LearningRateScheduler(linear_warmup_scheduler),
        keras.callbacks.TensorBoard(log_dir='logs/ProtT5_Uniref/model_ProtT5_efficientnet_v2b0')
    ]
    
    # class weights from sklearn's `compute_class_weight` function
    class_weight = {
        0: 1.380400890868597,
        1: 1.0619883486273465,
        2: 0.9074965835882085,
        3: 0.8116815086432687
    }
    
    history = model.fit(
        train_data,
        validation_data=val_data, 
        epochs=100,
        callbacks = callbacks, 
        batch_size = None, 
        class_weight = class_weight
    )
    
    print('done!')
    
