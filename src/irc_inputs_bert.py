from bert import tokenization
import tensorflow as tf
import tensorflow_hub as hub

sess_config = tf.ConfigProto(allow_soft_placement=True)
sess_config.gpu_options.allow_growth = True


def get_sess_config():
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    return sess_config


def create_tokenizer_from_hub_module(bert_hub_model_handle, sess_config=get_sess_config()):
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        #bert_module = hub.Module(FLAGS.bert_hub_module_handle)
        bert_module = hub.Module(bert_hub_model_handle, trainable=True)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session(config=sess_config) as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                  tokenization_info["do_lower_case"]])
    return tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)


def truncate_seq_pair(tokens_a, tokens_b, max_length):
    assert max_length >= 2
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
        return tokens_a, tokens_b

    half_length = round(max_length / 2 + .01)
    a_end = half_length
    b_end = max_length - half_length
    if len(tokens_a) < a_end:
        b_end = max_length - len(tokens_a)

    if len(tokens_b) < b_end:
        a_end = max_length - len(tokens_b)

    return tokens_a[:a_end], tokens_b[:b_end]


def msg_pairs_to_bert_inputs(bert_hub_model_handle, msg_context, curr_msg, prev_msg, max_seq_length):
    '''
    Inputs:
        msg_context: size = [total_messages]
            a list of original message context
        curr_msg, prev_msg: size = [total_pairs]
            a list of message index

    Outputs:
        bert_inputs: size = [pairs, max_seq_len]
    '''
    tokenizer = create_tokenizer_from_hub_module(bert_hub_model_handle)
    msg_token_ids = []
    for message in msg_context:
        tokens = tokenizer.tokenize(message)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        msg_token_ids.append(input_ids)
    CLS_id = tokenizer.convert_tokens_to_ids(["[CLS]"])[0]
    SEP_id = tokenizer.convert_tokens_to_ids(["[SEP]"])[0]

    feat_ids = []
    feat_mask = []
    feat_seg_ids = []

    for msg1_id, msg2_id in zip(prev_msg, curr_msg):
        input_ids = []
        segment_ids = []

        input_ids.append(CLS_id)
        segment_ids.append(0)

        token_ids_a, token_ids_b = truncate_seq_pair(
            msg_token_ids[msg1_id], msg_token_ids[msg2_id], max_seq_length - 3)

        for token_id in token_ids_a:
            input_ids.append(token_id)
            segment_ids.append(0)

        input_ids.append(SEP_id)
        segment_ids.append(0)

        for token_id in token_ids_b:
            input_ids.append(token_id)
            segment_ids.append(1)

        input_ids.append(SEP_id)
        segment_ids.append(1)

        input_mask = [1] * len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        feat_ids.append(input_ids)
        feat_mask.append(input_mask)
        feat_seg_ids.append(segment_ids)

    return feat_ids, feat_mask, feat_seg_ids
