import tensorflow as tf
import pickle as pkl
import numpy as np
import pandas as pd

def preprocessing(data):
    data = data.str.normalize('NFD')
    data = data.str.encode('ascii', 'ignore').str.decode('utf-8')
    data = data.str.lower()
    data = data.str.replace(r"([^ a-z.?!¡,¿])", "", regex=True)
    data = data.str.replace(r"([?.!¡,¿])", r" \1 ", regex=True)
    data = data.str.replace(r'[" "]+', " ")
    data = data.str.strip()
    data = '<START> ' + data + ' <END>'
    return data

def predict_seq2seq(input_text, encoder_model, decoder_model, tokenizer_input, tokenizer_target, max_length_input, max_length_target):
    # Encode input
    input_seq = tokenizer_input.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_length_input, padding='post')
    encoder_output = encoder_model.predict(input_seq)
    
    # Start sequence with <START> token
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer_target.word_index['<START>']
    
    # Iteratively predict next tokens
    decoded_sentence = ''
    for _ in range(max_length_target):
        output_tokens = decoder_model.predict([encoder_output, target_seq])
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = tokenizer_target.index_word[sampled_token_index]
        
        if sampled_token == '<END>':
            break
        decoded_sentence += ' ' + sampled_token
        
        # Update target sequence (of length 1)
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
    
    return decoded_sentence

if __name__ == '__main__':
    model = tf.keras.models.load_model('seq2seq.h5')
    with open('tokenizer_x.pkl', 'rb') as f:
        tokenizer_x = pkl.load(f)

    with open('tokenizer_y.pkl', 'rb') as f:
        tokenizer_y = pkl.load(f)

    max_length_x = 51
    max_length_y = 53   
    input_seq = "I'll miss you."
    output_sentence = predict_seq2seq(model, tokenizer_x, tokenizer_y, input_seq, max_length_x, max_length_y)
    print('Input:', input_seq)
    print('Output:', output_sentence)