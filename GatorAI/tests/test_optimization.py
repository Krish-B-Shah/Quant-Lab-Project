from GatorAI.src.optimization.optimizer import mean_variance_optimize
import pandas as pd


def test_optimizer_outputs_weights():
	ret = pd.DataFrame({"A": [0.01, -0.01, 0.0], "B": [0.0, 0.02, -0.01]})
	w = mean_variance_optimize(ret)
	assert abs(w.sum() - 1.0) < 1e-6
	assert set(w.index) == {"A", "B"}
