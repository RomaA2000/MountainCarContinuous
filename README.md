# MountainCarContinuous
I divided the continuous set of values into a discrete grid, and applied the standard RL algorithm with a Q-function.
Then I trained the model by 1000 epochs and got a 92 mean score, but a mean result above 80 is already achieved by 100.

If you turn on .env parameter, simulation won't stop until the car reaches the finish flag, and because of this first results will be ~-1500, but mean at all training will also be ~92.

![learning results with env](plots_continuous/learning_results_with_env)

Here is an example of learning outcomes without .env parameter (last epoch finish position is ~0.45, so car reaches the flag):

![learning results](plots_continuous/learning_results)

If you increase the number of grid elements, you will have to wait a little longer, but the result improves and becomes more stable. Only the number of epochs affects time since the current state in the state grid is searched by binary search.

![learning results smaller grid](plots_continuous/learning_results_smaller_grid)
