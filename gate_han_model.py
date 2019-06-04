import tensorflow as tf
import numpy as np
from preprocess import BuildDateset
from util import compute_acc, compute_max_sentence, compute_max_para
from keras.preprocessing import text, sequence
from tensorflow.contrib import rnn



class GateHanModel:
    def __init__(self):
        # 词向量的维度 self.embedding_dim
        self.word_embedding_dim = 50
        # 词汇表
        self.word_dict_size = 2500
        # lstm 输入的维度
        self.lstm_input_dim = 200
        # 输入每个序列最大长度
        self.input_x_max_len = None
        # 输入每个文档的最大句子数量
        self.sentence_max_len = 700
        # batch_size
        self.batch_size = None
        # self.batch_size = 2
        # word lstm hidden 维度
        self.word_lstm_hidden_size = 128
        # sentence lstm hidden 维度
        self.sentence_lstm_hidden_size = 256

        # 分类任务的数量
        self.class_num = 13
        # 学习率
        self.learning_rate = 0.0001

        self.input_x = tf.placeholder(dtype=tf.int64,
                                      shape=[self.batch_size, self.sentence_max_len, self.input_x_max_len])
        self.input_paragraph_len = self.paragraph_len(self.input_x)

        self.target_y = tf.placeholder(dtype=tf.int64, shape=[self.batch_size, None])

        self.idx2word_vec = tf.Variable(
            initial_value=np.random.normal(size=(self.word_dict_size, self.word_embedding_dim)).astype(np.float32), )

        self.word_embedding = tf.nn.embedding_lookup(self.idx2word_vec, self.input_x)
        self.lstm_input = self.word_embedding

        # 第三步 构造 词级别lstm

        fw_lstm_cell = tf.nn.rnn_cell.LSTMCell(self.word_lstm_hidden_size, forget_bias=1.0, state_is_tuple=True,
                                               initializer=tf.orthogonal_initializer())  # cell
        # bw_lstm_cell = tf.nn.rnn_cell.LSTMCell(self.word_lstm_hidden_size, forget_bias=1.0, state_is_tuple=True,
        #                                        initializer=tf.orthogonal_initializer())  # cell

        attention_w = tf.get_variable(name='attention_w', shape=[self.word_lstm_hidden_size, 40])
        attention_u = tf.get_variable(name='attention_u', shape=[40, 1])
        attention_b = tf.get_variable(name='attention_b', shape=[40])

        def build_words_LSTM(input_x, lstm_input):
            """

            """
            encoder_inputs_actual_length = self.sequence_len(input_x)
            encoder_outputs, states = tf.nn.dynamic_rnn(fw_lstm_cell, lstm_input,
                                                        sequence_length=encoder_inputs_actual_length, dtype=tf.float32)

            # outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_lstm_cell,
            #                                              cell_bw=bw_lstm_cell,
            #                                              inputs=lstm_input,
            #                                              sequence_length=encoder_inputs_actual_length,
            #                                              dtype=tf.float32,
            #                                              )
            # encoder_outputs = tf.concat(outputs, 2)

            attention_weight = tf.tensordot(tf.tanh(tf.tensordot(encoder_outputs, attention_w, axes=1) + attention_b),
                                            attention_u, axes=1)

            attention_mask = tf.expand_dims(
                tf.sequence_mask(encoder_inputs_actual_length, maxlen=tf.shape(lstm_input)[1], dtype=tf.float32, ),
                axis=-1
            )
            max_attention_weight = tf.reduce_max(attention_weight, axis=1, keep_dims=True)
            mask_attention_exp = tf.exp((attention_weight - max_attention_weight)) * attention_mask
            exp_sum = tf.reduce_sum(mask_attention_exp, axis=1, keepdims=True)
            attention_weight = tf.div(mask_attention_exp, exp_sum + 1e-6)
            weighted_projection = tf.multiply(encoder_outputs, attention_weight)
            outputs = tf.reduce_sum(weighted_projection, axis=1, )

            return outputs

        sentence_inputs = tf.transpose(tf.map_fn(fn=lambda x: build_words_LSTM(x[0], x[1]),
                                                 elems=(tf.transpose(self.input_x, [1, 0, 2]),
                                                        tf.transpose(self.lstm_input, [1, 0, 2, 3])),
                                                 dtype=tf.float32), [1, 0, 2])
        # 构造句子级别的lstm
        fw_lstm_cell_s = tf.nn.rnn_cell.LSTMCell(self.sentence_lstm_hidden_size, forget_bias=1.0, state_is_tuple=True,
                                                 initializer=tf.orthogonal_initializer(), name='fw_sentence')  # cell
        # bw_lstm_cell_s = tf.nn.rnn_cell.LSTMCell(self.sentence_lstm_hidden_size, forget_bias=1.0, state_is_tuple=True,
        #                                          initializer=tf.orthogonal_initializer(), name='bw_sentence')  # cell
        #
        # outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_lstm_cell_s,
        #                                              cell_bw=bw_lstm_cell_s,
        #                                              inputs=sentence_inputs,
        #                                              sequence_length=self.input_paragraph_len,
        #                                              dtype=tf.float32,
        #                                              )
        # encoder_outputs = tf.concat(outputs, 2)
        encoder_outputs, states = tf.nn.dynamic_rnn(fw_lstm_cell_s, sentence_inputs, sequence_length=self.input_paragraph_len, dtype=tf.float32)
        self.lstm_ouputs = tf.reshape(encoder_outputs, [-1, self.sentence_lstm_hidden_size])

        # 第四步
        self.classification_embedding = tf.Variable(
            initial_value=np.random.normal(size=(self.sentence_lstm_hidden_size, self.class_num + 1)).astype(
                np.float32), )
        self.classification_bias = tf.Variable(
            initial_value=np.random.normal(size=(self.class_num + 1)).astype(np.float32), )
        self.predict_vector = tf.nn.softmax(
            tf.matmul(self.lstm_ouputs, self.classification_embedding) + self.classification_bias)

        self.predict_vector = tf.reshape(self.predict_vector, [-1,  self.sentence_max_len, self.class_num+1])

        self.predict = tf.argmax(self.predict_vector, axis=-1)
        # 计算loss
        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.target_y, logits=self.predict_vector)

        # opt
        optimizer = tf.train.AdamOptimizer()
        grads, vars = zip(*optimizer.compute_gradients(self.loss))
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=5)
        self.opt_op = optimizer.apply_gradients(zip(grads, vars), name="train_op")

    def sequence_len(self, input_x):
        """
        计算句子的实际长度
        :param input_x: 用0填充.统计每个句子的非0 长度
        :return: 返回句子的实际长度
        """
        return tf.cast(tf.reduce_sum(tf.sign(tf.abs(input_x)), axis=1), tf.int32)

    def paragraph_len(self, input_x):
        """
        计算段落的实际长度
        :param input_x: 用0填充.统计每个句子的非0 长度
        :return: 返回句子的实际长度
        """
        return tf.cast(tf.reduce_sum(tf.sign(tf.abs(tf.reduce_sum(input_x, axis=-1))), axis=1), tf.int32)


if __name__ == '__main__':
    model = GateHanModel()
    sess = tf.Session()
    #  切成一段一段的。 比如下面 每段有3句话 一个有4段
    sess.run(tf.global_variables_initializer())
    dt = BuildDateset()
    # data_x = [
    #             [[1, 2, 3, 0, 0],
    #              [1, 2, 3, 0, 0],
    #              [0, 0, 0, 0, 0],
    #              ],
    #             [[4, 5, 0, 0, 0],
    #              [0, 0, 0, 0, 0],
    #              [0, 0, 0, 0, 0], ],
    #             [[1, 2, 3, 0, 0],
    #              [0, 0, 0, 0, 0],
    #              [0, 0, 0, 0, 0],
    #              ],
    #             [[1, 2, 3, 0, 0],
    #              [0, 0, 0, 0, 0],
    #              [0, 0, 0, 0, 0],
    #              ],
    #         ]
    # data_y = [
    #             [1, 1, 0],
    #             [2, 0, 0],
    #             [1, 0, 0],
    #             [1, 0, 0],
    #         ]
    train_x, train_y = dt.get_train()
    val_x, val_y = dt.get_val()
    test_x, test_y = dt.get_test()
    train_x = dt.data_x2seq(train_x)
    train_y = sequence.pad_sequences(train_y, maxlen=700, padding="post")
    train_y = train_y.tolist()
    val_x = dt.data_x2seq(val_x)
    val_y = sequence.pad_sequences(val_y, maxlen=700, padding="post")
    val_y = val_y.tolist()
    test_x = dt.data_x2seq(test_x)
    test_y = sequence.pad_sequences(test_y, maxlen=700, padding="post")
    test_y = test_y.tolist()
    start_index = 0
    end_index = 5
    
    max_val_acc = 0
    patient = 0
    for i in range(5000):
        print(start_index)
        if end_index > 100:
            end_index = 100
            batch_x = train_x[start_index:]
            batch_y = train_y[start_index:]
            start_index = 0
            end_index = 5
        else:
            batch_x = train_x[start_index: end_index]
            batch_y = train_y[start_index: end_index]
            start_index += 5
            end_index += 5

    # for batch_x, batch_y in zip(data_x, data_y):
        _ = sess.run(model.opt_op, feed_dict={
            model.input_x: batch_x,
            model.target_y: batch_y
        })

        if i % 10 == 0:
            loss = sess.run(model.loss, feed_dict={
                model.input_x: train_x,
                model.target_y: train_y
            })
            train_predict = sess.run(model.predict, feed_dict={
                model.input_x: train_x,
                # model.input_x: data_x,
            })
            val_predict = sess.run(model.predict, feed_dict={
                model.input_x: val_x,
                # model.input_x: data_x,
            })
            print(loss)
            compute_acc("training acc: ", train_predict.tolist(), train_y)
            val_acc = compute_acc("val acc: ", val_predict.tolist(), val_y)
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                patient = 0
            else:
                patient += 1
                if patient == 100:
                    print("early stop training")
                    break

    test_predict = sess.run(model.predict, feed_dict={
        model.input_x: test_x,
        # model.input_x: data_x,
    })
    compute_acc("test acc: ", test_predict.tolist(), test_y)


