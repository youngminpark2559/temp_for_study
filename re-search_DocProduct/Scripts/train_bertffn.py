# conda activate py36gputf && \
# cd /mnt/1T-5e7/mycodehtml/bio_health/NLP/Santosh_Gupta/Scripts && \
# rm e.l && python train_bertffn.py \
# 2>&1 | tee -a e.l && code e.l

# ================================================================================
import sys
network_dir='/mnt/1T-5e7/mycodehtml/bio_health/NLP/Santosh_Gupta'
sys.path.insert(0,network_dir)

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

# ================================================================================
import argparse
import os

import tensorflow as tf
import tensorflow.keras.backend as K

from Scripts.dataset import create_dataset_for_bert
from Scripts.models import MedicalQAModelwithBert
from Scripts.loss import qa_pair_loss, qa_pair_cross_entropy_loss
from Scripts.tokenization import FullTokenizer
from Scripts.metrics import qa_pair_batch_accuracy


def train_bertffn(model_path='models/bertffn_crossentropy/bertffn',
                  # data_path='data/mqa_csv',
                  data_path='/mnt/1T-5e7/mycodehtml/bio_health/NLP/Santosh_Gupta/reddit_data.csv',
                  # num_epochs=20,
                  num_epochs=2,
                  num_gpu=1,
                  # batch_size=64,
                  batch_size=4,
                  learning_rate=2e-5,
                  validation_split=0.2,
                  loss='categorical_crossentropy',
                  # pretrained_path='pubmed_pmc_470k/',
                  pretrained_path="/mnt/1T-5e7/mycodehtml/bio_health/NLP/Santosh_Gupta/BioBertFolder/pubmed_pmc_470k",
                  max_seq_len=256):

    # ================================================================================
    tf.compat.v1.disable_eager_execution()

    # ================================================================================
    if loss == 'categorical_crossentropy':
        loss_fn = qa_pair_cross_entropy_loss
    else:
        loss_fn = qa_pair_loss
    
    # ================================================================================
    K.set_floatx('float32')

    # ================================================================================
    vocab_path=os.path.join(pretrained_path, 'vocab.txt')
    # print("vocab_path",vocab_path)
    # /mnt/1T-5e7/mycodehtml/bio_health/NLP/Santosh_Gupta/BioBertFolder/pubmed_pmc_470k/vocab.txt

    tokenizer = FullTokenizer(vocab_path)

    # ================================================================================
    # @ Train data
    d = create_dataset_for_bert(
        data_path, tokenizer=tokenizer, batch_size=batch_size,
        shuffle_buffer=500000, dynamic_padding=True, max_seq_length=max_seq_len)
    
    # ================================================================================
    # @ Eval data
    eval_d = create_dataset_for_bert(
        data_path, tokenizer=tokenizer, batch_size=batch_size,
        mode='eval', dynamic_padding=True, max_seq_length=max_seq_len,
        bucket_batch_sizes=[64, 64, 64])

    # ================================================================================
    ckpt_path=os.path.join(pretrained_path, 'biobert_model.ckpt')

    medical_qa_model = MedicalQAModelwithBert(
        config_file=os.path.join(
            pretrained_path, 'bert_config.json'),
        checkpoint_file=ckpt_path)
    
    # ================================================================================
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    
    # ================================================================================
    medical_qa_model.compile(
        optimizer=optimizer, loss=loss_fn, metrics=[qa_pair_batch_accuracy])

    # ================================================================================
    epochs = num_epochs

    # ================================================================================
    loss_metric = tf.keras.metrics.Mean()

    # ================================================================================
    medical_qa_model.fit(d, epochs=epochs)

    # ================================================================================
    medical_qa_model.summary()

    # ================================================================================
    medical_qa_model.save_weights(model_path)

    # ================================================================================
    medical_qa_model.evaluate(eval_d)

if __name__ == "__main__":

    train_bertffn()
