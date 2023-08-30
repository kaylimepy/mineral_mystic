import tensorflow

class CrystalVision:
    """
    Class for mineral classification using a pre-trained EfficientNetB3 model.
    """
    def __init__(self, img_size=(256, 256)):
        """
        Initialize the MineralMystic class with image size and base model.
        
        Parameters:
            img_size (tuple): Image dimensions (width, height)
        """
        # Define image shape based on the provided size
        self.IMG_SHAPE = img_size + (3,)
        
        # Initialize pre-trained EfficientNetB3 model for feature extraction
        self.base_model = tensorflow.keras.applications.EfficientNetB3(input_shape=self.IMG_SHAPE,
                                                                       include_top=False,
                                                                       weights='imagenet')
        
        # Set the base model to non-trainable
        self.base_model.trainable = False
        
        # Initialize Global Average Pooling layer
        self.global_average_layer = tensorflow.keras.layers.GlobalAveragePooling2D()
        
        # Initialize Dense layer for predictions
        self.prediction_layer = tensorflow.keras.layers.Dense(7, activation='softmax')
        
        # Assemble the complete model
        self._build_model()


    def fine_tune(self, start_layer):
        """
        Unfreeze layers from 'start_layer' and make them trainable for fine-tuning.
        
        Parameters:
            start_layer (int): The layer index from which to start fine-tuning
        """
        # Unfreeze all layers in the base model
        self.base_model.trainable = True
        
        # Freeze layers before the 'start_layer' layer
        for layer in self.base_model.layers[:start_layer]:
            layer.trainable = False

    def compile_model(self, optimizer, loss, metrics):
        """
        Compile the model with the specified optimizer, loss, and metrics.
        
        Parameters:
            optimizer (str or tensorflow.keras.optimizers.Optimizer): The optimizer to use
            loss (str or tensorflow.keras.losses.Loss): The loss function
            metrics (list): List of metrics to evaluate the model
        """
        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=metrics)

    def summary(self):
        """
        Display model architecture summary.
        """
        self.model.summary()
        
    def _build_model(self):
        """
        Assemble the layers to form the complete model.
        """
        # Define input layer
        inputs = tensorflow.keras.Input(shape=self.IMG_SHAPE)
        
        # Stack layers
        x = self.base_model(inputs, training=False)
        x = self.global_average_layer(x)
        x = tensorflow.keras.layers.Dropout(0.2)(x)
        outputs = self.prediction_layer(x)
        
        # Create the model
        self.model = tensorflow.keras.Model(inputs, outputs)
        

