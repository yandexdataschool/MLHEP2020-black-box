import numpy as np

__all__ = [
  'eval_optimization'
]

def eval_optimization(
  optimiser_class, f, x0, bounds,
  moving_cost=1, measuring_cost=1, budget=128,
  progress=None
):
  optimiser = optimiser_class(x0)

  current_x = np.zeros_like(x0)

  history_x = list()
  history_f = list()

  budget_left = budget

  max_iterations = int(np.floor(budget_left / measuring_cost))

  progress = progress(total=budget) if progress is not None else None

  for _ in range(max_iterations):
    x = optimiser.ask()
    x = np.array(x, dtype=np.float64)
    assert x.shape == x0.shape

    x = np.clip(x, bounds[:, 0], bounds[:, 1])

    distance = np.max(np.abs(current_x - x))

    cost = moving_cost * distance + measuring_cost

    if budget_left < cost:
      if progress is not None:
        progress.update(budget - progress.n)

      break

    if progress is not None:
      progress.update(int(budget - budget_left) - progress.n)

    budget_left -= cost
    value = f(x)
    current_x = x

    optimiser.tell(x, value)

    history_x.append(np.copy(x))
    history_f.append(np.copy(value))

  if progress is not None:
    progress.close()

  return np.array(history_x), np.array(history_f)