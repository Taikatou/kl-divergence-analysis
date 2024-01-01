import onnx
import onnxruntime as ort
from mlagents_envs.environment import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt
import torch.distributions as distributions

# Initialize the Unity Environment
env_path = 'F:\\coma_env\\topdown-research.exe'
unity_env = UnityEnvironment(file_name=env_path)
unity_env.reset()
behavior_name = list(unity_env.behavior_specs)[0]
spec = unity_env.behavior_specs[behavior_name]

# Load ONNX models
ppo_model_path = "C:\\Users\\conor\\Documents\\GitHub\\ml-agents-3\\COMAKoalaGun.onnx"
sac_model_path = "C:\\Users\\conor\\Documents\\GitHub\\ml-agents-3\\LearningKoala.onnx"

ppo_session = ort.InferenceSession(ppo_model_path)
sac_session = ort.InferenceSession(sac_model_path)

# Function to get input shape from ONNX model for obs_3 and obs_4
def get_input_shape(onnx_model, input_name):
    for input in onnx_model.graph.input:
        if input.name == input_name:
            shape = [dim.dim_value for dim in input.type.tensor_type.shape.dim]
            return tuple(shape)

# Get input shapes for obs_3 and obs_4
obs_3_shape = get_input_shape(onnx.load(ppo_model_path), "obs_3")
obs_4_shape = get_input_shape(onnx.load(ppo_model_path), "obs_4")

# Function to perform inference using ONNX models
def infer_onnx_model(session, observations, action_masks):
    print(observations)
    ort_inputs = {
        "obs_0": observations[0].astype(np.float32),
        "obs_1": observations[1].astype(np.float32),
        "obs_2": observations[2].astype(np.float32),
        "obs_3": np.zeros(obs_3_shape, dtype=np.float32),
        "obs_4": np.zeros(obs_4_shape, dtype=np.float32),
        "action_masks": action_masks.astype(np.float32)
    }
    ort_outs = session.run(None, ort_inputs)
    return ort_outs


# Function to compute KL divergence
def compute_kl_divergence(probs1, probs2):
    kl_div = distributions.kl_divergence(distributions.Categorical(probs=probs1),
                                         distributions.Categorical(probs=probs2)).mean()
    return kl_div.item()


# Run multiple episodes and collect data
num_episodes = 10
all_ppo_probs, all_sac_probs = [], []

for episode in range(num_episodes):
    unity_env.reset()
    decision_steps, terminal_steps = unity_env.get_steps(behavior_name)

    while len(terminal_steps.agent_id) == 0:  # Continue until the episode is finished
        observations = decision_steps.obs
        action_masks = np.ones(
            (len(decision_steps), 9))  # Assuming all actions are available

        print(observations)
        ppo_probs = infer_onnx_model(ppo_session, observations, action_masks)
        sac_probs = infer_onnx_model(sac_session, observations, action_masks)

        all_ppo_probs.append(np.exp(ppo_probs[0]))
        all_sac_probs.append(np.exp(sac_probs[0]))

        action = np.random.randint(spec.action_spec.discrete_size, size=(len(decision_steps),))
        print(action)
        unity_env.set_actions(behavior_name, action)
        unity_env.step()
        decision_steps, terminal_steps = unity_env.get_steps(behavior_name)

# Aggregate and compute KL divergence
all_ppo_probs = np.concatenate(all_ppo_probs, axis=0)
all_sac_probs = np.concatenate(all_sac_probs, axis=0)
kl_divergence = compute_kl_divergence(all_ppo_probs, all_sac_probs)

# Output and plot KL divergence
print(f"KL Divergence after {num_episodes} episodes: {kl_divergence}")
plt.figure(figsize=(10, 6))
plt.bar(['PPO vs SAC'], [kl_divergence])
plt.ylabel('KL Divergence')
plt.title(f'KL Divergence between PPO and SAC Agents after {num_episodes} Episodes')
plt.show()
