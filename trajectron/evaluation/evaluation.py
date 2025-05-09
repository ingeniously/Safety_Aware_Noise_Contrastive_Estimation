import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import binary_dilation
from scipy.stats import gaussian_kde
from utils import prediction_output_to_trajectories
import visualization
from matplotlib import pyplot as plt

def compute_ade(predicted_trajs, gt_traj):
    error = np.linalg.norm(predicted_trajs - gt_traj, axis=-1)
    ade = np.mean(error, axis=-1)
    return ade.flatten()


def compute_fde(predicted_trajs, gt_traj):
    final_error = np.linalg.norm(predicted_trajs[:, :, -1] - gt_traj[-1], axis=-1)
    return final_error.flatten()


def compute_kde_nll(predicted_trajs, gt_traj):
    kde_ll = 0.
    log_pdf_lower_bound = -20
    num_timesteps = gt_traj.shape[0]
    num_batches = predicted_trajs.shape[0]

    for batch_num in range(num_batches):
        for timestep in range(num_timesteps):
            try:
                kde = gaussian_kde(predicted_trajs[batch_num, :, timestep].T)
                pdf = np.clip(kde.logpdf(gt_traj[timestep].T), a_min=log_pdf_lower_bound, a_max=None)[0]
                kde_ll += pdf / (num_timesteps * num_batches)
            except np.linalg.LinAlgError:
                kde_ll = np.nan

    return -kde_ll


def compute_obs_violations(predicted_trajs, map):
    obs_map = map.data

    interp_obs_map = RectBivariateSpline(range(obs_map.shape[1]),
                                         range(obs_map.shape[0]),
                                         binary_dilation(obs_map.T, iterations=4),
                                         kx=1, ky=1)

    old_shape = predicted_trajs.shape
    pred_trajs_map = map.to_map_points(predicted_trajs.reshape((-1, 2)))

    traj_obs_values = interp_obs_map(pred_trajs_map[:, 0], pred_trajs_map[:, 1], grid=False)
    traj_obs_values = traj_obs_values.reshape((old_shape[0], old_shape[1]))
    num_viol_trajs = np.sum(traj_obs_values.max(axis=1) > 0, dtype=float)

    return num_viol_trajs


def interpolate_traj(traj, num_interp=4):
    '''
    Add linearly interpolated points of a trajectory
    '''
    sz = traj.shape
    dense = np.zeros((sz[0], (sz[1] - 1) * (num_interp + 1) + 1, 2))
    dense[:, :1, :] = traj[:, :1]

    for i in range(num_interp+1):
        ratio = (i + 1) / (num_interp + 1)
        dense[:, i+1::num_interp+1, :] = traj[:, 0:-1] * (1 - ratio) + traj[:, 1:] * ratio

    return dense


def compute_col(predicted_traj, predicted_trajs_all, thres=0.2, num_interp=4):
    '''
    Input:
        predicted_trajs: predicted trajectory of the primary agents
        predicted_trajs_all: predicted trajectory of all agents in the scene
    '''
    ph = predicted_traj.shape[0]
    dense_all = interpolate_traj(predicted_trajs_all, num_interp)
    dense_ego = interpolate_traj(predicted_traj[None, :], num_interp)
    distances = np.linalg.norm(dense_all - dense_ego, axis=-1)
    mask = distances[:, 0] > 0
    return distances[mask].min(axis=0) < thres


def compute_batch_statistics(prediction_output_dict,
                             dt,
                             max_hl,
                             ph,
                             node_type_enum,
                             kde=True,
                             obs=False,
                             map=None,
                             prune_ph_to_future=False,
                             best_of=False,
                             col=False):

    (prediction_dict,
     _,
     futures_dict) = prediction_output_to_trajectories(prediction_output_dict,
                                                       dt,
                                                       max_hl,
                                                       ph,
                                                       prune_ph_to_future=prune_ph_to_future)

    batch_error_dict = dict()
    for node_type in node_type_enum:
        batch_error_dict[node_type] =  {'ade': list(), 'fde': list(), 'col_joint': list(), 'col_truth': list(), 'col_cross': list(), 'kde': list(), 'obs_viols': list()}

    for t in prediction_dict.keys():

        if col:
            prediction_joint = list()
            futures_joint = list()
            for node in prediction_dict[t].keys():
                prediction_joint.append(prediction_dict[t][node][0,0])
                futures_joint.append(futures_dict[t][node])
            prediction_joint = np.stack(prediction_joint, axis=0)
            futures_joint = np.stack(futures_joint, axis=0)

        for node in prediction_dict[t].keys():
            ade_errors = compute_ade(prediction_dict[t][node], futures_dict[t][node])
            fde_errors = compute_fde(prediction_dict[t][node], futures_dict[t][node])
            if col:
                idx_neighbors = abs(futures_joint[:, 0, 0] - futures_dict[t][node][None, 0, 0]) > 1e-8
                if idx_neighbors.sum() > 0:
                    num_interp = 4
                    col_joint = compute_col(prediction_dict[t][node][0,0], prediction_joint[idx_neighbors], num_interp=num_interp).astype(float)
                    col_cross = compute_col(prediction_dict[t][node][0,0], futures_joint[idx_neighbors], num_interp=num_interp).astype(float)
                    col_truth = compute_col(futures_dict[t][node], futures_joint[idx_neighbors], num_interp=num_interp)
                    col_joint[col_truth] = float('nan')
                    col_cross[col_truth] = float('nan')
                    col_truth = col_truth.astype(float)
                    if col_truth.any(): 
                        # skip frames where the groud truth observations lead to collisions 
                        ade_errors[:] = float('nan')
                        fde_errors[:] = float('nan')
                else:
                    col_joint = np.array([float('nan')] * (56))
                    col_truth = np.array([float('nan')] * (56))
                    col_cross = np.array([float('nan')] * (56))
            else:
                col_joint = 0
                col_truth = 0
                col_cross = 0
            if kde:
                kde_ll = compute_kde_nll(prediction_dict[t][node], futures_dict[t][node])
            else:
                kde_ll = 0
            if obs:
                obs_viols = compute_obs_violations(prediction_dict[t][node], map)
            else:
                obs_viols = 0
            if best_of:
                ade_errors = np.min(ade_errors, keepdims=True)
                fde_errors = np.min(fde_errors, keepdims=True)
                kde_ll = np.min(kde_ll)
            batch_error_dict[node.type]['ade'].extend(list(ade_errors))
            batch_error_dict[node.type]['fde'].extend(list(fde_errors))
            batch_error_dict[node.type]['col_joint'].extend([col_joint])
            batch_error_dict[node.type]['col_truth'].extend([col_truth])
            batch_error_dict[node.type]['col_cross'].extend([col_cross])
            batch_error_dict[node.type]['kde'].extend([kde_ll])
            batch_error_dict[node.type]['obs_viols'].extend([obs_viols])

    return batch_error_dict


def log_batch_errors(batch_errors_list, log_writer, namespace, curr_iter, bar_plot=[], box_plot=[]):
    for node_type in batch_errors_list[0].keys():
        for metric in batch_errors_list[0][node_type].keys():
            metric_batch_error = []
            for batch_errors in batch_errors_list:
                metric_batch_error.extend(batch_errors[node_type][metric])

            if len(metric_batch_error) > 0:
                log_writer.add_histogram(f"{metric}", metric_batch_error, curr_iter)
                log_writer.add_scalar(f"{metric}_mean", np.mean(metric_batch_error), curr_iter)
                log_writer.add_scalar(f"{metric}_median", np.median(metric_batch_error), curr_iter)

                if metric in bar_plot:
                    pd = {'dataset': [namespace] * len(metric_batch_error),
                                  metric: metric_batch_error}
                    kde_barplot_fig, ax = plt.subplots(figsize=(5, 5))
                    visualization.visualization_utils.plot_barplots(ax, pd, 'dataset', metric)
                    log_writer.add_figure(f"{metric}_bar_plot", kde_barplot_fig, curr_iter)

                if metric in box_plot:
                    mse_fde_pd = {'dataset': [namespace] * len(metric_batch_error),
                                  metric: metric_batch_error}
                    fig, ax = plt.subplots(figsize=(5, 5))
                    visualization.visualization_utils.plot_boxplots(ax, mse_fde_pd, 'dataset', metric)
                    log_writer.add_figure(f"{metric}_box_plot", fig, curr_iter)


def print_batch_errors(batch_errors_list, namespace, curr_iter):
    for node_type in batch_errors_list[0].keys():
        for metric in batch_errors_list[0][node_type].keys():
            metric_batch_error = []
            for batch_errors in batch_errors_list:
                metric_batch_error.extend(batch_errors[node_type][metric])

            if len(metric_batch_error) > 0:
                print(f"{curr_iter}: {metric}_mean", np.mean(metric_batch_error))
                print(f"{curr_iter}: {metric}_median", np.median(metric_batch_error))