import tensorflow as tf
import tensorflow.contrib.layers as layers

from utils.general import get_logger
from utils.test_env import EnvTest
from q1_schedule import LinearExploration, LinearSchedule
from q2_linear import Linear


from configs.q3_nature import config


class NatureQN(Linear):
    """
    Implementing DeepMind's Nature paper. Here are the relevant urls.
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    """
    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor) 
                shape = (batch_size, img height, img width, nchannels)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        # this information might be useful
        num_actions = self.env.action_space.n

        ##############################################################
        """
        TODO: implement the computation of Q values like in the paper
                https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
                https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

              you may find the section "model architecture" of the appendix of the 
              nature paper particulary useful.

              store your result in out of shape = (batch_size, num_actions)

        HINT: 
            - You may find the following functions useful:
                - tf.layers.conv2d
                - tf.layers.flatten
                - tf.layers.dense

            - Make sure to also specify the scope and reuse

        """
        ##############################################################
        ################ YOUR CODE HERE - 10-15 lines ################ 

        #pass
        """
        The input to the neural network consists of an 84x84x4 image produced by the preprocessing
        map w.
        The first hidden layer convolves 32 filters of 8x8 with stride 4 with the
        input image and applies a rectifier nonlinearity.
        The second hidden layer convolves 64 filters of 4x4 with stride 2, again followed by a rectifier nonlinearity.
        This is followed by a third convolutional layer that convolves 64 filters of 3x3 with
        stride 1 followed by a rectifier.
        The final hidden layer is fully-connected and consists
        of 512 rectifier units.
        The output layer is a fully-connected linear layer with a
        single output for each valid action.
        The number of valid actions varied between 4
        and 18 on the games we considered.
        """
        with tf.variable_scope(scope, reuse=reuse) as _:
            out = layers.conv2d(inputs=state,
                                     num_outputs=32,
                                     kernel_size=[8, 8],
                                     padding="same",
                                     stride=4)
            out = layers.conv2d(inputs=out,
                                     num_outputs=64,
                                     kernel_size=[4, 4],
                                     padding="same",
                                     stride=2)
            out = layers.conv2d(inputs=out,
                                     num_outputs=64,
                                     kernel_size=[3, 3],
                                     padding="same",
                                     stride=1)
            out = layers.flatten(out)
            out = layers.fully_connected(inputs=out,
                                                   num_outputs=512)
            out = layers.fully_connected(inputs=out,
                                                    num_outputs=num_actions,
                                                    activation_fn = None)

        ##############################################################
        ######################## END YOUR CODE #######################
        return out


"""
Use deep Q network for test environment.
"""
if __name__ == '__main__':
    env = EnvTest((80, 80, 1))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = NatureQN(env, config)
    model.run(exp_schedule, lr_schedule)
