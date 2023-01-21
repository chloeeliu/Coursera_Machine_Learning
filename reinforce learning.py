# UNQ_C1
# GRADED CELL

# Create the Q-Network.
q_network = Sequential([
    ### START CODE HERE ### 
        Input(shape=state_size),                      
        Dense(units=64, activation='relu'),            
        Dense(units=64, activation='relu'),            
        Dense(units=num_actions, activation='linear'),
    
    ### END CODE HERE ### 
    ])

# Create the target Q^-Network.
target_q_network = Sequential([
    ### START CODE HERE ### 
        Input(shape=state_size),                       
        Dense(units=64, activation='relu'),            
        Dense(units=64, activation='relu'),            
        Dense(units=num_actions, activation='linear'), 
        
    ### END CODE HERE ###
    ])

### START CODE HERE ### 
optimizer =Adam(learning_rate=ALPHA) 
### END CODE HERE ###


# UNQ_C2
# GRADED FUNCTION: calculate_loss

def compute_loss(experiences, gamma, q_network, target_q_network):
    """ 
    Calculates the loss.
    
    Args:
      experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
      gamma: (float) The discount factor.
      q_network: (tf.keras.Sequential) Keras model for predicting the q_values
      target_q_network: (tf.keras.Sequential) Keras model for predicting the targets
          
    Returns:
      loss: (TensorFlow Tensor(shape=(0,), dtype=int32)) the Mean-Squared Error between
            the y targets and the Q(s,a) values.
    """
    
    # Unpack the mini-batch of experience tuples.
    states, actions, rewards, next_states, done_vals = experiences
    
    # Compute max Q^(s,a).
    max_qsa = tf.reduce_max(target_q_network(next_states), axis=-1)
    
    # Set y = R if episode terminates, otherwise set y = R + γ max Q^(s,a).
    ### START CODE HERE ### 
    y_targets = rewards + (gamma * max_qsa * (1 - done_vals))
    ### END CODE HERE ###
    
    # Get the q_values.
    q_values = q_network(states)
    q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]),
                                                tf.cast(actions, tf.int32)], axis=1))
        
    # Compute the loss.
    ### START CODE HERE ### 
    loss = MSE(y_targets, q_values)
    ### END CODE HERE ### 
    
    return loss
