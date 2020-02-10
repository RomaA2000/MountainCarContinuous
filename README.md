# MountainCarContinuous
I divided the continuous set of values into a discrete grid, and applied the standard RL algorithm with a Q-function.
Then I trained the model by 1000 epochs and got a 92 mean score, but a mean result above 80 is already achieved by 100.

Here is an example of learning outcomes:

![learning results](learning_results)

If you increase the number of grid elements, you will have to wait a little longer, but the result improves and becomes more stable. Only the number of epochs affects time since the current state in the state grid is searched by binary search.

![learning results smaller grid](learning_results_smaller_grid)
