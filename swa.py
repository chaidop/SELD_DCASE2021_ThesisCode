# https://github.com/simon-larsson/keras-swa/blob/master/swa/keras.py
import tensorflow as tf

def on_epoch_end(epoch, model, start_epoch, swa_freq=2, verbose=True):
    start_epoch = start_epoch - 1
    swa_weights = None
    cnt = 0

    epoch = epoch - start_epoch
    if epoch == 0 or (epoch > 0 and epoch % swa_freq == 0):
        if verbose:
            print("\nSaving Weights... ", epoch+start_epoch)
        return update_swa_weights(swa_weights, model, cnt)

def on_train_end(model, swa_weights):
    print("\nThe final model Has Been set...")
    model.set_weights(swa_weights)

def update_swa_weights(swa_weights, model, cnt):
    if swa_weights is None:
        swa_weights = model.get_weights()
    else:
        swa_weights = [
            (swa_w*cnt + w) / (cnt+1)
            for swa_w, w in zip(swa_weights, model.get_weights())]
    cnt += 1

    return swa_weights