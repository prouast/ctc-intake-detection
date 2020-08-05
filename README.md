# ctc-intake-detection

Automatic detection of intake gestures with CTC from inertial or video data.

## Data access

For our experiments, we use the [OREBA](http://www.newcastle.edu.au/oreba) and [Clemson](http://cecas.clemson.edu/~ahoover/cafeteria/) datasets available from the respective websites.

## Data preparation

We have separate repositories with utilities to generate `TFRecord` files from the raw [inertial data](https://github.com/prouast/inertial-sensor-processing) and [video data](https://github.com/prouast/video-sensor-processing).

## Usage

Build `tensorflow_ctc_ext_beam_search_decoder` available at [ctc-beam-search-op](https://github.com/prouast/ctc-beam-search-op) for your system and install the wheel via pip, e.g.:

```
pip install tensorflow_ctc_ext_beam_search_decoder-0.1-cp36-cp36m-linux_x86_64.whl
```

Make sure that all other requirements are fulfilled:

```
$ pip install -r requirements.txt
```

Then call `main.py`:

```
$ python main.py
```

The following flags can be set:

| Argument | Description | Default |
| --- | --- | --- |
| --batch_size | Training batch size | 128 |
| --beam_width | Beam width during beam search | 10 |
| --dataset | Which dataset is used {oreba-dis or clemson} | oreba-dis |
| --decode_fn | Select the decode_fn {greedy or beam_search} | beam_search |
| --eval_batch_size | Evaluation batch size | 1 |
| --eval_dir | Directory with evaluation data | data/inert/valid |
| --eval_steps | Eval and save best model after every x steps | 1000 |
| --input_length | Number of input sequence elements | 128 |
| --input_mode | Select input mode {inertial or video} | inertial |
| --label_mode | Select the label mode | label_1 |
| --log_steps | Log after every x steps | 250 |
| --loss_mode | Select loss mode {ctc or crossent} | ctc |
| --lr_base | Base learning rate | 1e-3 |
| --lr_decay_fn | Select learning rate decay fn {exponential or piecewise_constant} | exponential |
| --lr_decay_rate | Learning rate decay rate | 0.9 |
| --mixed_precision | Use mixed precision {True or False} | False |
| --mode | Select mode {train_and_evaluate or predict} | train_and_evaluate |
| --model | Select model {video_resnet_cnn_lstm or inert_resnet_cnn_lstm} | inert_resnet_cnn_lstm |
| --model_ckpt | Model checkpoint for prediction (e.g., model_5000) | None |
| --model_dir | Output directory for model and training stats | run |
| --num_shuffle | Size of the shuffle buffer | 50000 |
| --predict_dir | Output directory for prediction mode | predict |
| --predict_mode | Select aggregation mode for predictions {video_level_average, video_level_concat, batch_level_voted, batch_level_concat, probs} | batch_level_voted |
| --seq_shift | Shift when generating sequences | 2 |
| --train_dir | Directory with training data | data/inert/train |
| --train_epochs | Number of train epochs | 60 |
| --use_def | Use default class for representation | False |
