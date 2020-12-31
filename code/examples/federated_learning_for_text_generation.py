import collections
import os

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

# 1 加载预训练模型
# 我们加载了预先训练后的TensorFlow教程的模型使用RNN充满渴望的执行文本生成 。然而，而不是训练莎士比亚全集 ，我们预先训练从狄更斯的文本模式双城记和圣诞颂歌 。
# 除了vocabulary扩大，我们没有修改原来的教程，所以这个最初的模式是不是国家的最先进的，但它产生的合理预测，并足以满足我们的教程的目的。最终的模型保存tf.keras.models.save_model(include_optimizer=False)
# 我们将使用联合学习微调这款型号为莎士比亚在本教程中，使用TFF提供的数据的联合版本。

# 1.1 生成词汇查找表

# A fixed vocabularly of ASCII chars that occur in the works of Shakespeare and Dickens:
vocab = list('dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r')

# Creating a mapping from unique characters to indices
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

# 1.2 装入预先训练模型，并生成一些文本

def load_model(batch_size):
    urls = {
        1: 'https://storage.googleapis.com/tff-models-public/dickens_rnn.batch1.kerasmodel',
        8: 'https://storage.googleapis.com/tff-models-public/dickens_rnn.batch8.kerasmodel'}
    assert batch_size in urls, 'batch_size must be in ' + str(urls.keys())
    url = urls[batch_size]
    local_file = tf.keras.utils.get_file(os.path.basename(url), origin=url)
    return tf.keras.models.load_model(local_file, compile=False)


def generate_text(model, start_string):
    # From https://www.tensorflow.org/tutorials/sequences/text_generation
    num_generate = 200
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    temperature = 1.0

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(
            predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))


# Text generation requires a batch_size=1 model.
keras_model_batch1 = load_model(batch_size=1)
print('# RESULT 1A')
print(generate_text(keras_model_batch1, 'What of TensorFlow Federated, you ask? '))

# 2 加载和预处理联邦莎士比亚数据
# 该tff.simulation.datasets包提供了多种分割成“客户”，数据集，其中，每个客户端对应一个数据集可能参与联合学习特定的设备上。
# 这些数据集提供逼真的非IID数据分布是重复的模拟训练真正上分散的数据所面临的挑战。部分数据的预处理是使用工具从完成叶片项目 （ github上 ）。

train_data, test_data = tff.simulation.datasets.shakespeare.load_data()

# 所提供的数据集shakespeare.load_data()由字符串序列中的Tensors ，一个用于由一个特定字在一个莎士比亚戏剧所说的每一行。客户端密钥由与角色的名字加入了游戏的名称，
# 因此，例如MUCH_ADO_ABOUT_NOTHING_OTHELLO对应于在剧中无事生非的人物奥赛罗线。
# 注意，在一个真正的联合学习方案的客户从来没有标识或ID进行跟踪，但对于模拟它带有钥匙的数据集的工作是有用的。
# 这里，例如，我们可以看看从李尔王的一些数据：

# Here the play is "The Tragedy of King Lear" and the character is "King".
raw_example_dataset = train_data.create_tf_dataset_for_client(
    'THE_TRAGEDY_OF_KING_LEAR_KING')
# To allow for future extensions, each entry x
# is an OrderedDict with a single key 'snippets' which contains the text.
print('# RESULT 2A')
for x in raw_example_dataset.take(2):
    print(x['snippets'])

# 我们现在使用tf.data.Dataset转换为训练RNN加载上面的字符准备这个数据。

# Input pre-processing parameters
SEQ_LENGTH = 100
BATCH_SIZE = 8
BUFFER_SIZE = 100  # For dataset shuffling

# Construct a lookup table to map string chars to indexes,
# using the vocab loaded above:
table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=vocab, values=tf.constant(list(range(len(vocab))),
                                       dtype=tf.int64)),
    default_value=0)


def to_ids(x):
    s = tf.reshape(x['snippets'], shape=[1])
    chars = tf.strings.bytes_split(s).values
    ids = table.lookup(chars)
    return ids


def split_input_target(chunk):
    input_text = tf.map_fn(lambda x: x[:-1], chunk)
    target_text = tf.map_fn(lambda x: x[1:], chunk)
    return (input_text, target_text)


def preprocess(dataset):
    return (
        # Map ASCII chars to int64 indexes using the vocab
        dataset.map(to_ids)
            # Split into individual chars
            .unbatch()
            # Form example sequences of SEQ_LENGTH +1
            .batch(SEQ_LENGTH + 1, drop_remainder=True)
            # Shuffle and form minibatches
            .shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
            # And finally split into (input, target) tuples,
            # each of length SEQ_LENGTH.
            .map(split_input_target))

# 注意，在上述批次形成的原始序列的和形成，我们使用drop_remainder=True为了简单起见。这意味着，没有任何字符（客户端）至少(SEQ_LENGTH + 1) * BATCH_SIZE文本字符将空的数据集。
# 一个典型的办法来解决，这将是垫特殊令牌批次，然后掩盖损失不采取填充令牌考虑。
# 这将在一定程度上例复杂，所以在本教程中，我们只使用全批次，如标准教程 。然而，在联邦环境这一问题更为显著，因为许多用户可能有小的数据集。
# 现在我们可以我们进行预处理raw_example_dataset ，并检查类型：

example_dataset = preprocess(raw_example_dataset)
print('# RESULT 2B')
print(example_dataset.element_spec)

# 3 编译对预处理后的数据模型和试验
# 我们加载了一个未编译keras模型，但为了运行keras_model.evaluate ，我们需要一个损失和指标进行编译。我们还将在编译的优化，这将用作该设备上优化的联合学习。
# 原教程没有烧焦级精度（其中最高的概率被提上了正确的下一个字符预测的分数）。这是一个有用的指标，所以我们添加。
# 但是，我们需要定义一个新的度量类这一点，因为我们的预测有等级3（logits的每个向量BATCH_SIZE * SEQ_LENGTH预测），以及SparseCategoricalAccuracy预计只有2个预测。

class FlattenedCategoricalAccuracy(tf.keras.metrics.SparseCategoricalAccuracy):

    def __init__(self, name='accuracy', dtype=tf.float32):
        super().__init__(name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, [-1, 1])
        y_pred = tf.reshape(y_pred, [-1, len(vocab), 1])
        return super().update_state(y_true, y_pred, sample_weight)

# 现在，我们可以编译模型，并评估对我们的example_dataset。

BATCH_SIZE = 8  # The training and eval batch size for the rest of this tutorial.
keras_model = load_model(batch_size=BATCH_SIZE)
keras_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[FlattenedCategoricalAccuracy()])

# Confirm that loss is much lower on Shakespeare than on random data
loss, accuracy = keras_model.evaluate(example_dataset.take(5), verbose=0)
print('# RESULT 3A')
print(
    'Evaluating on an example Shakespeare character: {a:3f}'.format(a=accuracy))

# As a sanity check, we can construct some completely random data, where we expect
# the accuracy to be essentially random:
random_guessed_accuracy = 1.0 / len(vocab)
print('Expected accuracy for random guessing: {a:.3f}'.format(
    a=random_guessed_accuracy))
random_indexes = np.random.randint(
    low=0, high=len(vocab), size=1 * BATCH_SIZE * (SEQ_LENGTH + 1))
data = collections.OrderedDict(
    snippets=tf.constant(
        ''.join(np.array(vocab)[random_indexes]), shape=[1, 1]))
random_dataset = preprocess(tf.data.Dataset.from_tensor_slices(data))
loss, accuracy = keras_model.evaluate(random_dataset, steps=10, verbose=0)
print('Evaluating on completely random data: {a:.3f}'.format(a=accuracy))


# 4 微调与联盟学习的模型
# TFF串行化所有TensorFlow计算，使他们有可能在非Python环境中运行（即使在目前，只有用Python实现模拟运行时可用）。
# 虽然我们在急切模式，（TF 2.0）上运行，目前TFF连载TensorFlow通过构建"的背景下内部必要的OPS计算with tf.Graph.as_default() "语句。
# 因此，我们需要提供一个功能，TFF可以用它来介绍我们的模型，把它控制的图形.我们这样做，如下所示：

# Clone the keras_model inside `create_tff_model()`, which TFF will
# call to produce a new copy of the model inside the graph that it will
# serialize. Note: we want to construct all the necessary objects we'll need
# _inside_ this method.
def create_tff_model():
    # TFF uses an `input_spec` so it knows the types and shapes
    # that your model expects.
    input_spec = example_dataset.element_spec
    keras_model_clone = tf.keras.models.clone_model(keras_model)
    return tff.learning.from_keras_model(
        keras_model_clone,
        input_spec=input_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[FlattenedCategoricalAccuracy()])

# 现在，我们准备建立一个联合平均化迭代的过程，我们将用它来改善模型（关于联邦平均算法的详细信息，请参阅本文从分散的数据深层网络进行通信的，高效的学习 ）。
# 我们使用编译Keras模型每一轮的联合训练后执行标准（非联盟）评估。做着模拟联合学习，有一个标准的测试数据集时，这是用于研究目的。
# 在实际生产设置同样的技术可用于采取与联盟学习训练的模型，并评估他们在一个集中的基准数据集用于测试或质量保证目的。

# This command builds all the TensorFlow graphs and serializes them:
fed_avg = tff.learning.build_federated_averaging_process(
    model_fn=create_tff_model,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(lr=0.5))

# 这是一个最简单的循环，在这里我们为一个轮单批单客户端上运行联盟平均：

state = fed_avg.initialize()
state, metrics = fed_avg.next(state, [example_dataset.take(5)])
train_metrics = metrics['train']
print('# RESULT 4A')
print('loss={l:.3f}, accuracy={a:.3f}'.format(
    l=train_metrics['loss'], a=train_metrics['accuracy']))

# 现在，让我们写一个稍微更有趣的训练和评估循环。
# 所以，这个模拟仍然运行比较快，我们每一轮，只有考虑两种minibatches每个训练上相同的三个客户。

def data(client, source=train_data):
    return preprocess(source.create_tf_dataset_for_client(client)).take(5)

clients = [
    'ALL_S_WELL_THAT_ENDS_WELL_CELIA', 'MUCH_ADO_ABOUT_NOTHING_OTHELLO',
]

print('# RESULT 4B')

train_datasets = [data(client) for client in clients]

# We concatenate the test datasets for evaluation with Keras by creating a
# Dataset of Datasets, and then identity flat mapping across all the examples.
test_dataset = tf.data.Dataset.from_tensor_slices(
    [data(client, test_data) for client in clients]).flat_map(lambda x: x)

# 所产生的模型的初始状态fed_avg.initialize()是基于用于Keras模式，而不是加载是，权重的随机初始化，由于clone_model()不克隆的权重。
# 要开始从预训练模型训练，我们在服务器状态直接从所加载的模型设定的模型权重。

NUM_ROUNDS = 5

# The state of the FL server, containing the model and optimization state.
state = fed_avg.initialize()

# Load our pre-trained Keras model weights into the global model state.
state = tff.learning.state_with_new_model_weights(
    state,
    trainable_weights=[v.numpy() for v in keras_model.trainable_weights],
    non_trainable_weights=[
        v.numpy() for v in keras_model.non_trainable_weights
    ])


def keras_evaluate(state, round_num):
    # Take our global model weights and push them back into a Keras model to
    # use its standard `.evaluate()` method.
    keras_model = load_model(batch_size=BATCH_SIZE)
    keras_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[FlattenedCategoricalAccuracy()])
    tff.learning.assign_weights_to_keras_model(keras_model, state.model)
    loss, accuracy = keras_model.evaluate(example_dataset, steps=2, verbose=0)
    print('\tEval: loss={l:.3f}, accuracy={a:.3f}'.format(l=loss, a=accuracy))


for round_num in range(NUM_ROUNDS):
    print('Round {r}'.format(r=round_num))
    keras_evaluate(state, round_num)
    state, metrics = fed_avg.next(state, train_datasets)
    train_metrics = metrics['train']
    print('\tTrain: loss={l:.3f}, accuracy={a:.3f}'.format(
        l=train_metrics['loss'], a=train_metrics['accuracy']))

print('Final evaluation')
keras_evaluate(state, NUM_ROUNDS + 1)

# 使用默认的变化，我们做得还不够培训，使一个很大的区别，但如果你对更多的莎士比亚数据训练时间越长，你应该看到在更新后的模型生成的文本的风格差异：

# Set our newly trained weights back in the originally created model.
keras_model_batch1.set_weights([v.numpy() for v in keras_model.weights])
# Text generation requires batch_size=1
print('# RESULT 4C')
print(generate_text(keras_model_batch1, 'What of TensorFlow Federated, you ask? '))
