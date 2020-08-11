import nest_asyncio
import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

# 1 准备联合数据集
print('1 准备联合数据集')

# 为了进行演示，我们将模拟一个场景，其中有来自10个用户的数据，每个用户都提供了如何识别不同数字的知识。这是关于为非独立同分布的 ，因为它得到。
# 首先，让我们加载标准MNIST数据：

mnist_train, mnist_test = tf.keras.datasets.mnist.load_data()

print([(x.dtype, x.shape) for x in mnist_train])

# 数据以Numpy数组的形式出现，一个带有图像，另一个带有数字标签，第一个维都遍历各个示例。
# 让我们编写一个帮助程序函数，该函数以与将联邦序列馈入TFF计算的方式兼容的方式对其进行格式化，即作为列表列表-外部列表覆盖用户（数字），内部列表覆盖用户的数据批次。每个客户的顺序。
# 按照惯例，我们将每个批次构造为一对名为x和y的张量，每个张量具有领先的批次尺寸。
# 同时，我们还将每个图像展平为784个元素的向量，并将其中的像素重新缩放到0..1范围，这样我们就不必在模型逻辑上进行数据转换了。

NUM_EXAMPLES_PER_USER = 1000
BATCH_SIZE = 100

def get_data_for_digit(source, digit):
    output_sequence = []
    all_samples = [i for i, d in enumerate(source[1]) if d == digit]
    for i in range(0, min(len(all_samples), NUM_EXAMPLES_PER_USER), BATCH_SIZE):
        batch_samples = all_samples[i:i + BATCH_SIZE]
        output_sequence.append({
            'x':
                np.array([source[0][i].flatten() / 255.0 for i in batch_samples],
                         dtype=np.float32),
            'y':
                np.array([source[1][i] for i in batch_samples], dtype=np.int32)
        })
    return output_sequence

federated_train_data = [get_data_for_digit(mnist_train, d) for d in range(10)]

federated_test_data = [get_data_for_digit(mnist_test, d) for d in range(10)]

# 作为快速检查，让我们看一下第五个客户端（对应于数字5 ）提供的最后一批数据中的Y张量。

print(federated_train_data[5][-1]['y'])

from matplotlib import pyplot as plt

plt.imshow(federated_train_data[5][-1]['x'][-1].reshape(28, 28), cmap='gray')
plt.grid(False)
plt.show()

# 2 关于结合TensorFlow和TFF
print('2 关于结合TensorFlow和TFF')

# 3 定义损失函数
print('3 定义损失函数')

BATCH_SPEC = collections.OrderedDict(
    x=tf.TensorSpec(shape=[None, 784], dtype=tf.float32),
    y=tf.TensorSpec(shape=[None], dtype=tf.int32))
BATCH_TYPE = tff.to_type(BATCH_SPEC)

print(BATCH_TYPE)

MODEL_SPEC = collections.OrderedDict(
    weights=tf.TensorSpec(shape=[784, 10], dtype=tf.float32),
    bias=tf.TensorSpec(shape=[10], dtype=tf.float32))
MODEL_TYPE = tff.to_type(MODEL_SPEC)

print(MODEL_TYPE)