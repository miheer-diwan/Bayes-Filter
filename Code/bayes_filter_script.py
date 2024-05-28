from bayes_filter import BayesFilter

bayes_filter = BayesFilter()

# Task 1: action "do nothing" and measurement "door open"
action_do_nothing = 0
measurement_door_open = 1
measurement_door_closed = 0
threshold = 0.9999
_, _, iteration_count_task1 = bayes_filter.bayes_filter(action_do_nothing, measurement_door_open, threshold)
print("\nTask 1:\nIterations required:", iteration_count_task1)

# Task 2:  action "push" and measurement "door open"
action_push = 1
_, _, iteration_count_task2 = bayes_filter.bayes_filter(action_push, measurement_door_open, threshold)
print("\nTask 2:\nIterations required:", iteration_count_task2)

# Task 3:  action "push" and measurement "door open"
action_push = 1
bel_open, bel_closed, iteration_count_task3 = bayes_filter.bayes_filter(action_push, measurement_door_closed, threshold)
print("\nTask 3: \n bel_open:", bel_open,
      "\n bel_closed:", bel_closed,
      "\n Iterations required: ", iteration_count_task3)
