#! /usr/bin/python


import math
import numpy as np
import tensorflow as tf
from keras import losses, optimizers, layers, Model

import rospy

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from gazebo_msgs.srv import GetWorldProperties, DeleteModel
from gazebo_msgs.msg import LinkStates


class ForagerController:
    @staticmethod
    def run():
        rospy.init_node('forager_controller')
        fc = ForagerController()
        rospy.spin()


    def __init__(self):
        self.num_actions = 4

        self.steps_per_episode = 150
        self.action_probs_history = []
        self.critic_value_history = []
        self.rewards_history = []

        self.episode_reward = 0
        self.running_reward = 0

        self.gamma = 0.99  # Discount factor for past rewards
        self.eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

        self.optimizer = optimizers.Adam(learning_rate=0.01)
        self.huber_loss = losses.Huber()
        self.tape = tf.GradientTape(persistent=True)

        self.model = self.get_model()


        self.bridge = CvBridge()

        self.world_properties = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)
        self.delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)

        self.link_states = None
        self.sub_link_states = rospy.Subscriber("/gazebo/link_states", LinkStates, self.save_link_states)
        self.sub_image = rospy.Subscriber("/forager/camera/image_raw", Image, self.actor_critic)
        self.pub_twist = rospy.Publisher('/forager/cmd_vel', Twist, queue_size=1)


    def get_model(self):
        inputs = layers.Input(shape=(3, 400, 400))

        conv_1 = layers.Convolution2D(16, (4, 4), padding="same", activation='relu')(inputs)
        conv_2 = layers.Convolution2D(16, (4, 4), padding="same", activation='relu')(conv_1)
        pool_1 = layers.MaxPooling2D(pool_size=(2, 2))(conv_2)

        flat = layers.Flatten()(pool_1)
        common = layers.Dense(256, activation='relu')(flat)
        action = layers.Dense(self.num_actions, activation="softmax")(common)
        critic = layers.Dense(1)(common)

        return Model(inputs=inputs, outputs=[action, critic])


    def actor_critic(self, image: Image):
        with self.tape as tape:
            cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='rgb8')
            np_image = np.asarray(cv_image)
            np_image = np.moveaxis(np_image, -1, 0)
            np_image = np_image[np.newaxis, ...]

            # Predict action probabilities and estimated future rewards
            # from environment state
            action_probs, critic_value =self.model(np_image)
            self.critic_value_history.append(critic_value[0, 0])

            # Sample action from action probability distribution
            action = np.random.choice(self.num_actions, p=np.squeeze(action_probs))
            self.twist(action)
            self.action_probs_history.append(tf.math.log(action_probs[0, action]))

            # Get reward for previous step
            if len(self.action_probs_history) > 1:
                reward = self.get_reward()
                self.rewards_history.append(reward)
                self.episode_reward += reward
            
            if len(self.rewards_history) >= self.steps_per_episode:
                # Update running reward to check condition for solving
                self.running_reward = 0.05 * self.episode_reward + (1 - 0.05) * self.running_reward

                # Calculate expected value from rewards
                # - At each timestep what was the total reward received after that timestep
                # - Rewards in the past are discounted by multiplying them with gamma
                # - These are the labels for our critic
                returns = []
                discounted_sum = 0
                for r in self.rewards_history[::-1]:
                    discounted_sum = r + self.gamma * discounted_sum
                    returns.insert(0, discounted_sum)

                # Normalize
                returns = np.array(returns)
                returns = (returns - np.mean(returns)) / (np.std(returns) + self.eps)
                returns = returns.tolist()

                # Calculating loss values to update our network
                history = zip(self.action_probs_history[:-1], self.critic_value_history[:-1], returns)
                actor_losses = []
                critic_losses = []
                for log_prob, value, ret in history:
                    # At this point in history, the critic estimated that we would get a
                    # total reward = `value` in the future. We took an action with log probability
                    # of `log_prob` and ended up recieving a total reward = `ret`.
                    # The actor must be updated so that it predicts an action that leads to
                    # high rewards (compared to critic's estimate) with high probability.
                    diff = ret - value
                    actor_losses.append(-log_prob * diff)  # actor loss

                    # The critic must be updated so that it predicts a better estimate of
                    # the future rewards.
                    critic_losses.append(
                        self.huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
                    )

                # Backpropagation
                loss_value = sum(actor_losses) + sum(critic_losses)
                grads = tape.gradient(loss_value, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                # Clear the loss and reward history
                self.action_probs_history = self.action_probs_history[-1:]
                self.critic_value_history = self.critic_value_history[-1:]
                self.rewards_history.clear()


    def twist(self, action: int):
        msg = Twist()
        if action == 0:
            msg.linear.x = -1.0
        elif action == 1:
            msg.linear.x = 1.0 
        elif action == 2:
            msg.angular.z = -1.0
        elif action == 3:
            msg.angular.z = 1.0
        
        self.pub_twist.publish(msg)


    def save_link_states(self, link_states: LinkStates):
        self.link_states = link_states


    def get_reward(self):
        foragerer_ind = self.link_states.name.index('foragerer::chassis')
        foragerer_pos = self.link_states.pose[foragerer_ind].position

        model_names = self.world_properties().model_names
        model_names = list(filter(lambda name: name.startswith('goal'), model_names))

        achieved = 0
        for name in model_names:
            goal_ind = self.link_states.name.index(f'{name}::link')
            goal_pos = self.link_states.pose[goal_ind].position
            
            dist = math.sqrt((foragerer_pos.x - goal_pos.x)**2 + (foragerer_pos.y - goal_pos.y)**2)
            if dist < 2.5:
                achieved += 1
                self.delete_model(name)

        if achieved > 0:
            return 10.0 * achieved
        else:
            return 0.0


if __name__ == '__main__':
    ForagerController.run()
