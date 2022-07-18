max_episodes = 10000
gamma = 0.99
batch_size = 32 #256 max
max_steps = 10000
memory_size = 100000
learning_rate = 1e-3
epsilon = 1.0
epsilon_min = 0.1
random_episodes = 0
update_after_episodes = 1
update_q_after_episodes = 10
episodes_to_epsilon_min = 5000
delta_epsilon = (epsilon - epsilon_min) / episodes_to_epsilon_min
loss_function = keras.losses.Huber()
optimizer = keras.optimizers.Adam(learning_rate= learning_rate)

def create_q_model():
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(64, 64, 4))

    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation='relu')(inputs)
    layer2 = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation='relu')(layer1)
    layer3 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu')(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(256, activation='relu')(layer4)
    action = tf.keras.layers.Dense(4, activation="linear")(layer5)

    return keras.Model(inputs=inputs, outputs=action)