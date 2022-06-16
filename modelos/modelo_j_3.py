seed = 42
gamma = 0.99
batch_size = 256 #256 max
max_steps = 30000
memory_size = 100000
learning_rate = 0.0005
epsilon = 1.0
epsilon_min = 0.1
random_episodes = 2000
update_after_actions = 16
update_q_after_episodes = 200
episodes_to_epsilon_min = 8000
delta_epsilon = (epsilon - epsilon_min) / episodes_to_epsilon_min
loss_function = keras.losses.Huber()
optimizer = keras.optimizers.Adam(learning_rate= learning_rate)

def create_q_model():
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(84, 84, 4))

    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(32,(8,8), strides=2, activation="relu")(inputs)
    layer2 = layers.Conv2D(64,(4,4), strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64,(2,2), strides=1, activation="relu")(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(1024, activation="relu")(layer4)
    action = tf.keras.layers.Dense(4, activation="linear")(layer5)

    return keras.Model(inputs=inputs, outputs=action)