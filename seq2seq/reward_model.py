'''
Our reward model should accunt for multiple factors: 

- Diagnostic accuracy: You want to minimize over-testing and under-testing, so the reward should encourage ordering tests that provide useful diagnostic information.
- Cost: Consider the cost of ordering tests. Minimize the total cost, which includes the cost of individual tests as well as any penalties for over-testing.
- Rare Occurrence Penalty: Assign higher penalties to larger order sets since they indicate inefficient resource allocation. This could be a term that increases as the number of tests ordered increases.
- Resilience: Consider incorporating a term that encourages resilience to environmental factors. This could involve penalizing actions that are sensitive to environmental fluctuations.
'''

import torch 
import torch.nn as nn 

# Implementing just a simple Linear regression model for now
class LinearRewardModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRewardModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # Linear layer with one output

    def forward(self, x):
        reward = self.linear(x)
        return reward


if __name__ == '__main__':
    # example data
    reward_model = LinearRewardModel(input_size=80)
    order_set_data = torch.rand(1, 80)  # Example order set data
    print(f"sample order set: {order_set_data}")
    reward = reward_model(order_set_data)
    print("Reward:", reward.item())







