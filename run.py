from crystal_vision import CrystalVision
from utils.data_processing import remove_invalid_and_corrupted_files, get_datasets, get_class_weights
from utils.logging import logger

DATA_PATH  = 'temp/minet2'
BATCH_SIZE = 64
IMG_SIZE   = (256, 256)

crystal_vision = CrystalVision()


def run_training_sequence():
    logger.info('Starting training sequence.')

    remove_invalid_and_corrupted_files(DATA_PATH)
    logger.info('Removed invalid and corrupted files.')

    train_dataset, validation_dataset, test_dataset = get_datasets(DATA_PATH, BATCH_SIZE, IMG_SIZE)
    logger.info('Got datasets.')

    class_weights = get_class_weights(train_dataset)
    logger.info('Got class weights.')

    crystal_vision.compile_model(learning_rate=0.001)
    logger.info('Compiled initial model.')
    
    history = crystal_vision.model.fit(train_dataset, 
                                       epochs=10, 
                                       validation_data=validation_dataset, 
                                       callbacks=crystal_vision.callbacks,
                                       class_weight=class_weights)
    logger.info('Trained initial model.')

    crystal_vision.fine_tune(100)
    logger.info('Fine tuned model.')

    crystal_vision.compile_model(learning_rate=0.0001)
    logger.info('Compiled fine tuned model.')
    
    history_fine_tuned = crystal_vision.model.fit(train_dataset,
                                                  epochs=25,
                                                  initial_epoch=history.epoch[-1],
                                                  validation_data=validation_dataset,
                                                  callbacks=crystal_vision.callbacks,
                                                  class_weight=class_weights)
    logger.info('Trained fine tuned model.')

    crystal_vision.save_model('models/crystal_vision_model.h5')
    logger.info('Saved model.')
    
    validation_loss, validation_accuracy = crystal_vision.evaluate_model(validation_dataset)
    logger.info('Evaluated model.')

    logger.info(f"Validation accuracy: {round(validation_accuracy * 100, 2)}%")
    logger.info(f"Validation loss: {round(validation_loss, 2)}")


if __name__ == "__main__":
    run_training_sequence()
