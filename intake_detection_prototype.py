import tensorflow as tf
import operator
import os
from tensorflow.python.platform import gfile

BATCH_SIZE = 8
SEQ_LENGTH = 30
SEQ_SHIFT = 10
NUM_FEATURES = 1024
NUM_CLASSES = 2
LSTM_UNITS = 128
NUM_SHARDS = 4
DROPOUT = 0.5

DEF_VAL = 0
PAD_VAL = -1

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_enum(
    name='mode', default="train_and_evaluate", enum_values=["load_and_evaluate", "train_and_evaluate"],
    help='What mode should tensorflow be started in')

tf.enable_eager_execution()

def input_parser(serialized_example):
    features = tf.parse_single_example(
        serialized_example, {
            'example/video_id': tf.FixedLenFeature([], dtype=tf.int64),
            'example/label': tf.FixedLenFeature([], dtype=tf.int64),
            'example/prob_1': tf.FixedLenFeature([], dtype=tf.float32),
            'example/fc7': tf.FixedLenFeature([1024], dtype=tf.float32)
    })
    # Convert label to one-hot encoding
    label = tf.cast(features['example/label'], tf.int32)
    fc7 = features['example/fc7']
    return fc7, label

def dataset(data_dir, is_training):
    # Scan for training files
    filenames = gfile.Glob(os.path.join(data_dir, "*.tfrecords"))
    if not filenames:
        raise RuntimeError('No files found.')
    tf.logging.info("Found {0} files.".format(str(len(filenames))))
    # List files
    files = tf.data.Dataset.list_files(filenames)
    # Shuffle files if needed
    if is_training:
        files = files.shuffle(NUM_SHARDS)
    shift = SEQ_SHIFT if is_training else SEQ_LENGTH
    dataset = files.interleave(
        lambda filename:
            tf.data.TFRecordDataset(filename)
            .map(map_func=input_parser)
            .window(size=SEQ_LENGTH, shift=shift, drop_remainder=True)
            .flat_map(lambda f_w, l_w: tf.data.Dataset.zip(
                (f_w.batch(SEQ_LENGTH), l_w.batch(SEQ_LENGTH)))),
        cycle_length=1)
    if is_training:
        dataset = dataset.shuffle(1000).repeat()
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset

def dense_to_sparse(input, eos_token):
    idx = tf.where(tf.not_equal(input, tf.constant(eos_token, input.dtype)))
    values = tf.gather_nd(input, idx)
    shape = tf.shape(input, out_type=tf.int64)
    sparse = tf.SparseTensor(idx, values, shape)
    return sparse

def keras_model():
    inputs = tf.keras.layers.Input(shape=(SEQ_LENGTH, NUM_FEATURES))
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(units=LSTM_UNITS, dropout=DROPOUT,
            recurrent_dropout=DROPOUT, return_sequences=True))(inputs)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(units=LSTM_UNITS, dropout=DROPOUT,
            recurrent_dropout=DROPOUT, return_sequences=True))(x)
    n_classes = NUM_CLASSES + 1
    outputs = tf.keras.layers.Dense(n_classes)(x)
    return tf.keras.Model(inputs, outputs)

def ctc_loss_collapse_all(labels, logits):
    labels = tf.cast(labels, tf.int32)
    labels = tf.reshape(labels, [BATCH_SIZE, SEQ_LENGTH])
    logits = tf.transpose(a=logits, perm=[1, 0, 2])
    logits = tf.reshape(logits, [SEQ_LENGTH, -1, NUM_CLASSES+1])
    loss = tf.nn.ctc_loss(
        labels=dense_to_sparse(labels, eos_token=PAD_VAL),
        inputs=logits,
        sequence_length=tf.fill([BATCH_SIZE], SEQ_LENGTH),
        preprocess_collapse_repeated=True,
        ctc_merge_repeated=False)
    loss = tf.reduce_mean(loss)
    return loss

def ctc_loss_collapse_ones(labels, logits):
    labels = tf.cast(labels, tf.int32)
    labels = tf.reshape(labels, [BATCH_SIZE, SEQ_LENGTH])
    labels, seq_length = collapse_sequences(labels, def_val=DEF_VAL, pad_val=PAD_VAL, replace_with_idle=False)
    logits = tf.transpose(a=logits, perm=[1, 0, 2])
    logits = tf.reshape(logits, [SEQ_LENGTH, -1, NUM_CLASSES+1])
    loss = tf.nn.ctc_loss(
        labels=dense_to_sparse(labels, eos_token=PAD_VAL),
        inputs=logits,
        sequence_length=seq_length,
        preprocess_collapse_repeated=False,
        ctc_merge_repeated=False)
    loss = tf.reduce_mean(loss)
    return loss

def ctc_loss(labels, logits):
    labels = tf.cast(labels, tf.int32)
    labels = tf.reshape(labels, [BATCH_SIZE, SEQ_LENGTH])
    logits = tf.transpose(a=logits, perm=[1, 0, 2])
    logits = tf.reshape(logits, [SEQ_LENGTH, -1, NUM_CLASSES+1])
    loss = tf.nn.ctc_loss(
        labels=dense_to_sparse(labels, eos_token=PAD_VAL),
        inputs=logits,
        sequence_length=tf.fill([BATCH_SIZE], SEQ_LENGTH),
        preprocess_collapse_repeated=False,
        ctc_merge_repeated=False)
    loss = tf.reduce_mean(loss)
    return loss

def collapse_sequences(labels, def_val=1, pad_val=0, replace_with_idle=True, pos='middle'):
    """Collapse sequences of labels, optionally replacing with default value

    Args:
        labels: The labels, which includes default values (e.g, 1) and
            sequences of interest (e.g., [1, 1, 2, 2, 2, 1, 1, 3, 3, 3, 1]).
        def_val: The value which denotes the default value
        replace_with_idle: If true, collapsed values are replaced with the
            idle val. If false, collapsed values are removed (Tensor is padded
            with 0).
        pos: The position relative to the original sequence to keep the
            remaining non-collapsed value.
    """
    # Get general dimensions
    batch_size = labels.get_shape()[0]
    maxlen = labels.get_shape()[1]

    # Find differences in labels
    diff_mask = tf.not_equal(labels[:, 1:], labels[:, :-1])

    # Mask labels that don't equal previous/next label.
    prev_mask = tf.concat([tf.ones_like(labels[:, :1], tf.bool), diff_mask], axis=1)
    next_mask = tf.concat([diff_mask, tf.ones_like(labels[:, :1], tf.bool)], axis=1)

    if replace_with_idle:

        # Masks for non-default vals, sequence starts and ends
        not_default_mask = tf.not_equal(labels, tf.fill(tf.shape(labels), def_val))
        seq_start_mask = tf.logical_and(prev_mask, not_default_mask)
        seq_end_mask = tf.logical_and(next_mask, not_default_mask)

        # Test if there are no sequences in labels
        empty = tf.equal(tf.reduce_sum(tf.cast(not_default_mask, tf.int32)), 0)

        # Sequence val occurrences
        seq_vals = tf.boolean_mask(labels, seq_start_mask)

        # Prepare padded seq vals
        seq_count_per_batch = tf.reduce_sum(tf.cast(seq_start_mask, tf.int32), axis=[1])
        max_seq_count = tf.reduce_max(seq_count_per_batch)
        seq_val_idx_mask = tf.reshape(tf.sequence_mask(seq_count_per_batch, maxlen=max_seq_count), [-1])
        seq_val_idx = tf.boolean_mask(tf.range(tf.size(seq_val_idx_mask)), seq_val_idx_mask)
        seq_val = tf.scatter_nd(
            indices=tf.expand_dims(seq_val_idx, axis=1),
            updates=seq_vals,
            shape=tf.shape(seq_val_idx_mask))
        seq_val = tf.reshape(seq_val, [batch_size, max_seq_count])

        # Prepare padded seq idx
        seq_se_count_per_batch = tf.reduce_sum(
            tf.cast(seq_start_mask, tf.int32) + tf.cast(seq_end_mask, tf.int32), axis=[1])
        max_seq_se_count = tf.reduce_max(seq_se_count_per_batch)
        se_idx_mask = tf.reshape(tf.sequence_mask(seq_se_count_per_batch,
            maxlen=max_seq_se_count), [-1])
        se_idx = tf.boolean_mask(tf.range(tf.size(se_idx_mask)), se_idx_mask)
        start_updates = tf.cast(tf.where(seq_start_mask)[:,1], tf.int32)
        end_updates = tf.cast(tf.where(seq_end_mask)[:,1], tf.int32)
        se_updates = tf.reshape(tf.stack([start_updates, end_updates], axis=1), [-1])
        seq_idx_se = tf.scatter_nd(
            indices=tf.expand_dims(se_idx, axis=1),
            updates=se_updates,
            shape=tf.shape(se_idx_mask))
        seq_idx_se = tf.reshape(seq_idx_se, [batch_size, max_seq_count, 2])
        seq_idx_se = tf.transpose(seq_idx_se, [0, 2, 1]) # [batch_size, start/end, num_vals]

        # For each sequence of seq_val, find the index to collapse to
        def collapse_seq_idx(start, end, pos='middle'):
            if pos == 'middle':
                return tf.floordiv(end + start, 2)
            elif pos == 'start':
                return start
            elif pos == 'end':
                return end
        seq_idx = tf.map_fn(lambda x: collapse_seq_idx(x[0], x[1]), seq_idx_se, dtype=tf.int32)

        # Scatter seq_vals and seq_idx to determine collapsed labels
        updates = tf.map_fn(
            fn=lambda x: tf.scatter_nd(indices=tf.expand_dims(x[0], axis=1),
                updates=x[1], shape=[maxlen]),
            elems=(seq_idx, seq_val),
            dtype=tf.int32)
        updates = tf.where(tf.equal(updates, 0), tf.fill(tf.shape(updates), def_val), updates)

        # Flatten and determine idx
        new_maxlen = maxlen
        flat_idx = tf.range(tf.size(labels))
        flat_updates = tf.reshape(updates, [-1])
        flat_shape = tf.shape(tf.reshape(labels, [-1]))

        seq_length = tf.fill([batch_size], maxlen)

    else:

        # Mask for all def_val in the sequence
        default_mask = tf.equal(labels, tf.fill(tf.shape(labels), def_val))

        # Combine with mask for all first sequence elements
        mask = tf.logical_or(default_mask, prev_mask)
        flat_updates = tf.boolean_mask(labels, mask, axis=0)

        # Determine new sequence lengths / max length
        new_seq_len = tf.reduce_sum(tf.cast(mask, tf.int32), axis=1)
        new_maxlen = tf.reduce_max(new_seq_len)

        # Mask idx
        idx_mask = tf.sequence_mask(new_seq_len, maxlen=new_maxlen)
        flat_idx_mask = tf.reshape(idx_mask, [-1])
        idx = tf.range(tf.size(idx_mask))
        flat_idx = tf.boolean_mask(idx, flat_idx_mask, axis=0)
        flat_shape = tf.shape(flat_idx_mask)

        seq_length = new_seq_len

    flat = tf.scatter_nd(
        indices=tf.expand_dims(flat_idx, axis=1),
        updates=flat_updates + 1,
        shape=flat_shape)

    flat = tf.where(tf.equal(flat, 0), tf.fill(flat_shape, pad_val + 1), flat)
    flat = flat - 1

    # Reshape back to square batch.
    new_shape = [batch_size, new_maxlen]
    result = tf.reshape(flat, new_shape)

    return result, seq_length

def greedy_decode_with_indices(inputs, num_classes, seq_length, pos='middle'):
    """Naive inference by retrieving most likely output at each time-step.

    Args:
        inputs: The prediction in form of logits. [samples, time_steps, num_classes+1]
            Contains an extra class for CTC epsilon at the last index
        num_classes: The number of classes considered in prediction.
        seq_length: Sequence length for each item in inputs.
        pos: How should predictions (excluding 0) be collapsed to 0?
    Returns:
        Tuple
        * the decoded sequence [seq_length]
        * indices of 1 predictions
    """
    def decode(input):
        cat_ids = tf.cast(tf.argmax(input, axis=1), tf.int32)
        cat_ids = tf.where(tf.equal(cat_ids, num_classes),
            tf.zeros([seq_length], tf.int32), cat_ids) # Set epsilons to 0
        row_ids = tf.range(tf.shape(input)[0], dtype=tf.int32)
        idx = tf.stack([row_ids, cat_ids], axis=1)
        return idx[:,1]

    decoded = tf.map_fn(decode, inputs, dtype=tf.int32)
    collapsed, _ = collapse_sequences(decoded, def_val=DEF_VAL, pad_val=PAD_VAL,
        replace_with_idle=True, pos=pos)
    one_indices = tf.where(tf.equal(collapsed, tf.constant(1, tf.int32)))

    return collapsed, one_indices

def f1(labels, logits):

    seq_length = SEQ_LENGTH
    def_val = 0

    # Cast just in case
    labels = tf.cast(tf.reshape(labels, [BATCH_SIZE, SEQ_LENGTH]), tf.int32)
    logits = tf.cast(tf.reshape(logits, [BATCH_SIZE, SEQ_LENGTH, NUM_CLASSES+1]), tf.int32)

    # Get the decoded logits
    predictions, _ = greedy_decode_with_indices(inputs=logits,
        num_classes=NUM_CLASSES, seq_length=seq_length, pos='middle')

    def sequence_masks(labels, def_val, seq_length):
        # Get dimensions
        batch_size = labels.get_shape()[0]

        # Mask elements non-equal to previous elements
        diff_mask = tf.not_equal(labels[:, 1:], labels[:, :-1])
        prev_mask = tf.concat([tf.ones_like(labels[:, :1], tf.bool), diff_mask], axis=1)
        next_mask = tf.concat([diff_mask, tf.ones_like(labels[:, :1], tf.bool)], axis=1)

        # Mask elements that are not def_val
        not_default_mask = tf.not_equal(labels, tf.fill(tf.shape(labels), def_val))

        # Test if there are no sequences
        empty = tf.equal(tf.reduce_sum(tf.cast(not_default_mask, tf.int32)), 0)

        # Mask sequence starts and ends
        seq_start_mask = tf.logical_and(prev_mask, not_default_mask)
        seq_end_mask = tf.logical_and(next_mask, not_default_mask)

        # Scatter seq_val
        seq_count_per_batch = tf.reduce_sum(tf.cast(seq_start_mask, tf.int32), axis=[1])
        max_seq_count = tf.reduce_max(seq_count_per_batch)
        seq_val_idx_mask = tf.reshape(tf.sequence_mask(seq_count_per_batch, maxlen=max_seq_count), [-1])
        seq_val_idx = tf.boolean_mask(tf.range(tf.size(seq_val_idx_mask)), seq_val_idx_mask)
        seq_vals = tf.boolean_mask(labels, seq_start_mask)
        seq_val = tf.scatter_nd(
            indices=tf.expand_dims(seq_val_idx, axis=1),
            updates=seq_vals,
            shape=tf.shape(seq_val_idx_mask))
        seq_val = tf.reshape(seq_val, [batch_size, max_seq_count])

        # Scatter seq_start
        seq_start_idx = tf.where(seq_start_mask)[:,1]
        seq_start = tf.scatter_nd(
            indices=tf.expand_dims(seq_val_idx, axis=1),
            updates=seq_start_idx,
            shape=tf.shape(seq_val_idx_mask))
        seq_start = tf.reshape(seq_start, [batch_size, max_seq_count])

        # Scatter seq_end
        seq_end_idx = tf.where(seq_end_mask)[:,1]
        seq_end = tf.scatter_nd(
            indices=tf.expand_dims(seq_val_idx, axis=1),
            updates=seq_end_idx,
            shape=tf.shape(seq_val_idx_mask))
        seq_end = tf.reshape(seq_end, [batch_size, max_seq_count])

        def batch_seq_masks(starts, ends, length, vals, def_val):
            def seq_mask(start, end, length, val, def_val):
                return tf.concat([
                    tf.fill([start], def_val),
                    tf.fill([end-start+1], val),
                    tf.fill([length-end-1], def_val)], axis=0)
            return tf.map_fn(
                fn=lambda x: seq_mask(x[0], x[1], length, x[2], def_val),
                elems=(starts, ends, vals),
                dtype=tf.int32)

        seq_masks = tf.cond(empty,
            lambda: tf.fill([batch_size, 1, seq_length], def_val),
            lambda: tf.map_fn(
                fn=lambda x: batch_seq_masks(x[0], x[1], seq_length, x[2], def_val),
                elems=(seq_start, seq_end, seq_val),
                dtype=tf.int32))

        return seq_masks, max_seq_count

    # Dimensions
    batch_size = labels.get_shape()[0]

    # Mask of negative ground truth
    neg_mask = tf.equal(labels, def_val)

    # Compute whether labels are empty (no sequences)
    pos_length = tf.reduce_sum(labels)
    empty = tf.cond(tf.equal(pos_length, 0), lambda: True, lambda: False)

    # Derive positive ground truth mask and stack predictions accordingly
    test = sequence_masks(labels, 0, seq_length)
    pos_mask, max_seq_count = sequence_masks(labels, 0, seq_length)
    pos_mask = tf.reshape(pos_mask, [-1, seq_length])
    pred_stacked = tf.reshape(tf.tile(tf.expand_dims(predictions, axis=1), [1, max_seq_count, 1]), [-1, seq_length])

    # Remove empty masks
    empty_mask = tf.greater(tf.reduce_sum(pos_mask, axis=1), 0)
    pos_mask = tf.boolean_mask(pos_mask, empty_mask)
    pred_stacked = tf.boolean_mask(pred_stacked, empty_mask)

    # Calculate number of predictions for pos sequences
    pred_sums = tf.map_fn(
        fn=lambda x: tf.reduce_sum(tf.boolean_mask(x[0], x[1])),
        elems=(pred_stacked, pos_mask), dtype=tf.int32)

    # Calculate true positive, false positive and false negative count
    tp = tf.reduce_sum(tf.map_fn(lambda count: tf.cond(count > 0, lambda: 1, lambda: 0), pred_sums))
    fn = tf.reduce_sum(tf.map_fn(lambda count: tf.cond(count > 0, lambda: 0, lambda: 1), pred_sums))
    fp = tf.cond(empty,
        lambda: 0,
        lambda: tf.reduce_sum(tf.map_fn(lambda count: tf.cond(count > 1, lambda: count-1, lambda: 0), pred_sums)))
    fp += tf.reduce_sum(tf.boolean_mask(predictions, mask=neg_mask))

    tp = tf.cast(tp, tf.float32)
    fn = tf.cast(fn, tf.float32)
    fp = tf.cast(fp, tf.float32)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    return f1

train_iterator = dataset("../data/train_3d_cnn", True).make_one_shot_iterator()
train_features, train_labels = train_iterator.get_next()
train_features = tf.reshape(train_features, [-1, SEQ_LENGTH, NUM_FEATURES])

eval_iterator = dataset("../data/eval_3d_cnn", False).make_one_shot_iterator()
eval_features, eval_labels = eval_iterator.get_next()
eval_features = tf.reshape(eval_features, [-1, SEQ_LENGTH, NUM_FEATURES])
eval_labels = tf.reshape(eval_labels, [-1, SEQ_LENGTH, 1])

def decode(input):
    num_classes = NUM_CLASSES
    seq_length = SEQ_LENGTH
    cat_ids = tf.cast(tf.argmax(input, axis=1), tf.int32)
    cat_ids = tf.where(tf.equal(cat_ids, num_classes),
        tf.zeros([seq_length], tf.int32), cat_ids) # Set epsilons to 0
    row_ids = tf.range(tf.shape(input)[0], dtype=tf.int32)
    idx = tf.stack([row_ids, cat_ids], axis=1)
    return idx[:,1]

print("---")
print("labels")
print(tf.squeeze(train_labels))

checkpoint_path = "run/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
    save_weights_only=True, verbose=1)

model = keras_model()
model.summary()
model.compile(
    optimizer='adam',
    loss=ctc_loss,
    metrics=[f1])

if FLAGS.mode == "train_and_evaluate":
    model.fit(
        x=train_features,
        y=train_labels,
        batch_size=BATCH_SIZE,
        epochs=5,
        steps_per_epoch=100,
        callbacks=[cp_callback])
elif FLAGS.mode == "load_and_evaluate":
    model.load_weights(checkpoint_path)

#result = model.evaluate(eval_features, eval_labels, steps=100)
result = model.predict(eval_features, steps=1)

print("====")
print(result)
print("---")
print("decode")
print(tf.map_fn(decode, result, dtype=tf.int32))

beam_decoded, log_probs = tf.nn.ctc_beam_search_decoder(
    inputs=tf.transpose(result, [1, 0, 2]),
    sequence_length=tf.fill([BATCH_SIZE], SEQ_LENGTH),
    merge_repeated=True
)

print(tf.sparse.to_dense(beam_decoded[0]))
print(log_probs)

#prediction, indices = greedy_decode_with_indices(inputs=result,
#    num_classes=NUM_CLASSES, seq_length=SEQ_LENGTH)
