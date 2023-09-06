from crystal_vision import CrystalVision
from data_utils import remove_invalid_and_corrupted_files, get_datasets, get_class_weights

DATA_PATH  = 'temp/minet2'
BATCH_SIZE = 64
IMG_SIZE   = (256, 256)

crystal_vision = CrystalVision()


def run_training_sequence():
    # todo: replace all prints with logging
    print("Running pipeline...")

    remove_invalid_and_corrupted_files(DATA_PATH)
    print("Removed invalid or corrupted files.")

    # Get datasets
    train_dataset, validation_dataset, test_dataset = get_datasets(DATA_PATH, BATCH_SIZE, IMG_SIZE)
    print("Got datasets.")

    # Get class weights
    class_weights = get_class_weights(train_dataset)
    print("Got class weights.")

    # Compile the initial model
    crystal_vision.compile_model(learning_rate=0.001)
    print("Compiled inital model.")
    
    # Train the initial model model
    history = crystal_vision.model.fit(train_dataset, 
                                       epochs=10, 
                                       validation_data=validation_dataset, 
                                       callbacks=crystal_vision.callbacks,
                                       class_weight=class_weights)
    print("Trained initial model.")

    # Fine tune initial model
    crystal_vision.fine_tune(100)
    print("Fine tuned initial model.")

    # Compile again
    crystal_vision.compile_model(learning_rate=0.0001)
    print("Compiled fine tuned model.")
    
    # Train fine tuned model
    history_fine_tuned = crystal_vision.model.fit(train_dataset,
                                                  epochs=25,
                                                  initial_epoch=history.epoch[-1],
                                                  validation_data=validation_dataset,
                                                  callbacks=crystal_vision.callbacks,
                                                  class_weight=class_weights)
    print("Trained fine tuned model.")

    # Save the model
    crystal_vision.save_model('models/crystal_vision_model.h5')
    
    # Evaluate model on validation dataset
    validation_loss, validation_accuracy = crystal_vision.evaluate_model(validation_dataset)

    print(f"Validation accuracy: {round(validation_accuracy * 100, 2)}%")
    print(f"Validation loss: {round(validation_loss, 2)}")


if __name__ == "__main__":
    run_training_sequence()
