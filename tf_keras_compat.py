"""TensorFlow/Keras import shim for DeepJ (TF 1.x notebooks through TF 2.x)."""
try:
    import tensorflow as tf
    if hasattr(tf, "keras"):
        from tensorflow.keras.layers import (
            Input, LSTM, Dense, Dropout, Lambda, Reshape, Permute,
            TimeDistributed, RepeatVector, Conv1D, Activation,
            Embedding, Flatten, Concatenate, Add,
        )
        from tensorflow.keras.models import Model
        import tensorflow.keras.backend as K
        from tensorflow.keras import losses
    else:
        raise ImportError("tensorflow.keras missing")
except ImportError:
    from keras.layers import (
        Input, LSTM, Dense, Dropout, Lambda, Reshape, Permute,
        TimeDistributed, RepeatVector, Conv1D, Activation,
        Embedding, Flatten,
    )
    from keras.layers.merge import Concatenate, Add
    from keras.models import Model
    import keras.backend as K
    from keras import losses

__all__ = [
    "Input", "LSTM", "Dense", "Dropout", "Lambda", "Reshape", "Permute",
    "TimeDistributed", "RepeatVector", "Conv1D", "Activation",
    "Embedding", "Flatten", "Concatenate", "Add", "Model", "K", "losses",
]