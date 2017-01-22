import tensorflow as tf

from rl import A3CAgent, ACAgentRunner
from util import *
from music import target_compositions, NOTES_PER_BAR

# Agent imports
import gym
import random
from rl import discount, make_summary

target_compositions += load_melodies('data/edm/edm_c')

batch_cache = {}

# Build custom cloning agent


class CloneAgentRunner(ACAgentRunner):

    def train(self, sess, coord, env_name, writer, gamma):
        try:
            print("Pre-processing CloneAgentRunner...")
            episode_count = 0

            def build_cache(targets):
                # Prebuild state input batches because we're cloning
                # A list of minibatches of input data.
                minibatches = [[[] for _ in self.model.model.inputs]]

                # Initialize first state
                state = self.preprocess(None, (targets[0], 0))
                self.memory.reset(state)

                # Bookkeeping
                for i, state in enumerate(self.memory.to_states()):
                    minibatches[-1][i].append(state)

                for t_index, target in enumerate(targets):
                    # Exclude first and last targets
                    if t_index == 0 and t_index < len(targets) - 2:
                        continue
                    # Simulate state transition
                    # TODO: Refactor so we don't need to pass env.
                    state = self.preprocess(None, (target, i % NOTES_PER_BAR))
                    self.memory.remember(state)

                    # Bookkeeping
                    for i, state in enumerate(self.memory.to_states()):
                        minibatches[-1][i].append(state)

                    if t_index % self.batch_size == 0:
                        minibatches.append([[]
                                            for _ in self.model.model.inputs])
                return minibatches

            # Reset per-episode vars
            terminal = False
            total_reward = 0
            step_count = 0
            # Index of current minibatch
            b_index = 0
            # Reset composition
            comp_id = random.randint(0, len(target_compositions) - 1)
            targets = target_compositions[comp_id]
            if comp_id not in batch_cache:
                batch_cache[comp_id] = build_cache(targets)
            minibatches = batch_cache[comp_id]

            print("Training CloneAgentRunner...")

            while not coord.should_stop():
                # Run a training minibatch batch
                input_batch = minibatches[b_index]

                # Batch predict
                *probs, values = sess.run(
                    self.model.model.outputs,
                    dict(zip(self.model.model.inputs, input_batch))
                )

                # De-batchify values
                values = values.T[0].tolist()

                # Actions
                actions = [(np.random.choice(len(p), p=p),) for p in probs[0]]
                # Compute rewards
                target_index = b_index * self.batch_size
                # TODO: Scale the reward
                rewards = [1 if a[0] == targets[target_index + i + 1]
                           else 0 for i, a in enumerate(actions)]

                b_index += 1
                terminal = b_index >= len(minibatches)
                total_reward += sum(rewards)

                step_count += 1

                if terminal:
                    bootstrap = 0
                else:
                    # Bootstrap from last state
                    bootstrap = sess.run(
                        self.model.value,
                        # TODO: Check if this theory is right.
                        # Input data of last state
                        dict(zip(
                            self.model.model.inputs,
                            [[input_data[-1]] for input_data in input_batch]
                        ))
                    )[0][0]

                # Here we take the rewards and values from the exp, and use them to
                # generate the advantage and discounted returns.
                # The advantage function uses "Generalized Advantage
                # Estimation"
                discounted_rewards = discount(rewards, gamma, bootstrap)
                value_plus = np.array(values + [bootstrap])
                advantages = discount(
                    rewards + gamma * value_plus[1:] - value_plus[:-1], gamma)

                # Train network
                v_l, p_l, e_l, g_n, v_n, _ = sess.run([
                    self.model.value_loss,
                    self.model.policy_loss,
                    self.model.entropy,
                    self.model.grad_norms,
                    self.model.var_norms,
                    self.model.train
                ],
                    {
                    **dict(zip(self.model.model.inputs, input_batch)),
                        **dict(zip(self.model.actions, zip(*actions))),
                        **
                    {
                            self.model.target_v: discounted_rewards,
                            self.model.advantages: advantages
                    }
                }
                )

                if terminal:
                    # Record metrics
                    writer.add_summary(
                        make_summary({
                            'rewards': total_reward,
                            'lengths': step_count,
                            'value_loss': v_l,
                            'policy_loss': p_l,
                            'entropy_loss': e_l,
                            'grad_norm': g_n,
                            'value_norm': v_n,
                            'mean_values': np.mean(values)
                        }),
                        episode_count
                    )

                    episode_count += 1

                    # Reset per-episode vars
                    terminal = False
                    total_reward = 0
                    step_count = 0
                    # Index of current minibatch
                    b_index = 0
                    # Reset composition
                    comp_id = random.randint(0, len(target_compositions) - 1)
                    targets = target_compositions[comp_id]
                    if comp_id not in batch_cache:
                        batch_cache[comp_id] = build_cache(targets)
                    minibatches = batch_cache[comp_id]

        except Exception as e:
            # Report exceptions to the coordinator.
            coord.request_stop(e)

with tf.Session() as sess, tf.device('/cpu:0'):
    agent = make_agent()
    agent.agents = [] # TODO
    agent.add_agent(CloneAgentRunner)

    try:
        agent.load(sess)
        print('Loading last saved session')
    except:
        print('Starting new session')

    agent.compile(sess)
    agent.train(sess, 'music-clone-v0').join()
