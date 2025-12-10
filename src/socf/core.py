import itertools
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.environ import value, Objective


def _generate_scenarios(disturbances):
    """
    Generate all disturbance scenarios, restricted to single-factor changes
    (including the nominal point).

    Returns
    -------
    combos : list[dict]
        List of disturbance dictionaries.
    scenario_names : list[str]
        Human-readable labels for each scenario.
    nominal : dict
        The nominal disturbance dictionary (first combination).
    """
    params = list(disturbances.keys())

    # All combinations (cartesian product)
    all_combos = [
        dict(zip(params, vals))
        for vals in itertools.product(*(disturbances[p] for p in params))
    ]

    # Nominal: first combo by construction
    nominal = all_combos[0]

    # Single-factor changes only (including nominal)
    combos = [
        c for c in all_combos
        if sum(c[p] != nominal[p] for p in params) <= 1
    ]

    # Name each scenario
    scenario_names = []
    for c in combos:
        diffs = [(p, c[p]) for p in params if c[p] != nominal[p]]
        if not diffs:
            scenario_names.append("Nominal")
        else:
            p, v = diffs[0]
            scenario_names.append(f"{p}={v}")

    return combos, scenario_names, nominal


def _solve_base_objectives(build_model, combos, solver):
    """
    Solve the base economic objective J for each scenario without
    user-imposed design constraints.

    Returns
    -------
    J_list : list[float]
        Optimal J for each scenario (same order as `combos`).
    """
    J_list = []
    for c in combos:
        m = build_model()
        for p, val in c.items():
            getattr(m, p).set_value(val)
        solver.solve(m, tee=False)
        J_list.append(value(m.J))
    return J_list


def _evaluate_metrics(build_model, metrics, combos, scenario_names, solver):
    """
    Evaluate user metrics for each scenario at the true optimum.

    Returns
    -------
    results_df : pandas.DataFrame
        rows = metrics, columns = scenarios.
    """
    results = []
    for c in combos:
        m = build_model()
        for p, val in c.items():
            getattr(m, p).set_value(val)
        solver.solve(m, tee=False)

        # Compute all metrics for this scenario
        results.append([fn(m, c) for fn in metrics.values()])

    results_df = pd.DataFrame(
        results,
        index=scenario_names,
        columns=metrics.keys()
    ).T.round(4)

    return results_df


def _build_raw_loss_matrix(build_model, user_designs, combos, scenario_names, J_list, solver):
    """
    Compute raw loss values (no infeasibility handling / ranking yet)
    for each user design under each scenario.

    Returns
    -------
    loss_matrix : pandas.DataFrame
        rows = scenarios, columns = "Loss with <design label>".
    """
    col_labels = [f"Loss with {lbl}" for lbl in user_designs]
    loss_matrix = pd.DataFrame(index=scenario_names, columns=col_labels, dtype=object)

    for lbl, apply_design in user_designs.items():
        col = f"Loss with {lbl}"

        for i, (name, c) in enumerate(zip(scenario_names, combos)):
            J_nom = J_list[i]

            m = build_model()
            apply_design(m)  # impose design constraint

            # Assume first objective is economic objective J
            all_objs = list(m.component_data_objects(Objective))
            if not all_objs:
                raise ValueError("Model has no Objective defined.")
            obj = all_objs[0]
            is_min = obj.is_minimizing()

            for p, val in c.items():
                getattr(m, p).set_value(val)

            try:
                solver.solve(m, tee=False)
                if is_min:
                    loss = round(value(m.J) - J_nom, 2)
                else:
                    loss = round(J_nom - value(m.J), 2)
            except Exception:
                loss = np.nan

            loss_matrix.at[name, col] = loss

    return loss_matrix


def _mark_infeasible_designs(loss_matrix):
    """
    Mark any design that yields negative loss in at least one scenario
    as 'infeasible' for all its scenarios.
    """
    for col in loss_matrix:
        numeric_col = pd.to_numeric(loss_matrix[col], errors="coerce")
        if (numeric_col < 0).any():
            loss_matrix[col] = "infeasible"
    return loss_matrix


def _append_average_and_ranking(loss_matrix):
    """
    Compute average loss and ranking, placing infeasible designs last.

    Modifies `loss_matrix` in place and returns it.
    """
    numeric = loss_matrix.replace("infeasible", np.nan).apply(pd.to_numeric)
    avg_loss = numeric.mean()

    # Infeasible designs have at least one NaN entry
    infeas = numeric.isnull().any()

    rank_vec = avg_loss.copy()
    max_ok = rank_vec[~infeas].max()
    rank_vec[infeas] = max_ok + 1  # push infeasible after all feasible

    rankings = rank_vec.rank(method="dense", ascending=True)

    loss_matrix.loc["Average loss"] = avg_loss.round(2)
    loss_matrix.loc["Ranking"] = rankings

    return loss_matrix


def run_self_optimizing_model(build_model, disturbances, user_designs, metrics):
    """
    Run the self-optimizing control model over a set of disturbance
    scenarios and user designs.

    Parameters
    ----------
    build_model : callable
        Function that returns a fresh Pyomo ConcreteModel.
    disturbances : dict
        Mapping parameter names to lists of values to test.
    user_designs : dict
        Mapping design labels to functions that apply constraints to a model.
    metrics : dict
        Mapping metric names to callables (model, scenario_dict) -> float.

    Returns
    -------
    results_df : pandas.DataFrame
        Optimal metric values for each scenario (rows=metrics, cols=scenarios).
    loss_matrix : pandas.DataFrame
        Losses for each design under each scenario, with average and ranking.
    """
    # 1. Disturbance scenarios (single-factor changes)
    combos, scenario_names, nominal = _generate_scenarios(disturbances)

    # 2. Solver (change here if you want a different solver/opts)
    solver = pyo.SolverFactory("ipopt")

    # 3. Base objective values J for each scenario
    J_list = _solve_base_objectives(build_model, combos, solver)

    # 4. Controlled metrics under each scenario at true optimum
    results_df = _evaluate_metrics(build_model, metrics, combos, scenario_names, solver)

    # 5. Loss matrix for each user design
    loss_matrix = _build_raw_loss_matrix(
        build_model=build_model,
        user_designs=user_designs,
        combos=combos,
        scenario_names=scenario_names,
        J_list=J_list,
        solver=solver,
    )

    # 6. Mark negative-loss designs as infeasible
    loss_matrix = _mark_infeasible_designs(loss_matrix)

    # 7. Add average loss and ranking rows
    loss_matrix = _append_average_and_ranking(loss_matrix)

    return results_df, loss_matrix
