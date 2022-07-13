from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


model = load_model('gru_trained.h5')

def try_me(trail):

    x_trail = word_tokenizer.texts_to_sequences([trail])
    x_trail_paded = pad_sequences(
        x_trail, maxlen=MAX_SEQ_LENGTH, padding='pre', truncating='post')

    y_pred = model.predict(x_trail_paded)

    y = []
    for i in range(len(y_pred[0])):
        y.append(y_pred[0][i].argmax())
    output = tag_tokenizer.sequences_to_texts([y])

    return output[0]