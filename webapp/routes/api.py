from __future__ import annotations

import csv
import io
import json
import math
import os
import threading
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from flask import Response, jsonify, redirect, render_template, request, stream_with_context, url_for
from torch.utils.data import DataLoader, random_split

from core.fl.algorithms import run_centralized, run_fedavg, run_fqngd, run_kiqfl
from core.fl.config import apply_comparison_defaults
from core.fl.data import load_dataset
from core.fl.utils import set_seed
from core.models import QNN_Regressor
from webapp.routes import bp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
last_simulation_results = {}
selected_feature_config = {}
_run_lock = threading.Lock()
_active_run_id = 0
_stop_run_id = None  # type: int | None


def _begin_run() -> int:
    """Start a new run id; any previous stream will see a mismatch and exit."""
    global _active_run_id, _stop_run_id
    with _run_lock:
        _active_run_id += 1
        _stop_run_id = None
        return _active_run_id


def _request_stop() -> None:
    """Stop the currently active run after its current epoch."""
    global _stop_run_id
    with _run_lock:
        _stop_run_id = _active_run_id


def _should_stop(run_id: int) -> bool:
    with _run_lock:
        if run_id != _active_run_id:
            return True
        return _stop_run_id is not None and _stop_run_id == run_id


def _build_checkpoint(run_type: str, params: dict, completed_epochs) -> dict:
    params = dict(params or {})
    total = params.get('global_epochs') or params.get('total_epochs')
    return {
        'type': run_type,
        'params': params,
        'completed_epochs': completed_epochs,
        'total_epochs': total,
        'stopped': True,
    }


def _sse(payload: dict) -> str:
    """Encode one SSE data frame; coerce NumPy scalars for json.dumps."""
    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
    return f"data: {json.dumps(payload, default=default)}\n\n"


def _sse_headers() -> dict:
    return {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }

@bp.route('/start_simulation', methods=['POST'])
def start_simulation():
    """Start single algorithm simulation"""
    # Invalidate any previous SSE training so Resume → Start is not stuck behind it.
    _begin_run()

    fl_type = request.form.get('fl_type')
    dataset = request.form.get('dataset')
    num_clients = int(request.form.get('num_clients', 5))
    local_epochs = int(request.form.get('local_epochs', 3))
    global_epochs = int(request.form.get('global_epochs', 50))
    num_qubits = int(request.form.get('num_qubits', 8))
    num_layers = int(request.form.get('num_layers', 2))
    learning_rate = float(request.form.get('learning_rate', 0.01))
    sporadic_p = float(request.form.get('sporadic_p', 0.1))
    theta_walk = float(request.form.get('theta_walk', math.pi / 6))
    phi_walk = float(request.form.get('phi_walk', 0.0))
    nisq_error_rate = float(request.form.get('nisq_error_rate', 0.01))
    data_fraction = float(request.form.get('data_fraction', 0.1))
    quantum_encoding = request.form.get('quantum_encoding', 'amplitude')
    ansatz = request.form.get('ansatz', 'ry_cx')
    measurement = request.form.get('measurement', 'probability')
    differential = request.form.get('differential', 'parameter_shift')
    loss_function = request.form.get('loss_function', 'cross_entropy')
    optimizer_choice = request.form.get('optimizer', 'adam')

    return render_template('simulation_results.html',
                          fl_type=fl_type,
                          dataset=dataset,
                          num_clients=num_clients,
                          data_type='iid',
                          local_epochs=local_epochs,
                          global_epochs=global_epochs,
                          num_qubits=num_qubits,
                          num_layers=num_layers,
                          learning_rate=learning_rate,
                          sporadic_p=sporadic_p,
                          theta_walk=theta_walk,
                          phi_walk=phi_walk,
                          nisq_error_rate=nisq_error_rate,
                          data_fraction=data_fraction,
                          quantum_encoding=quantum_encoding,
                          ansatz=ansatz,
                          measurement=measurement,
                          differential=differential,
                          loss_function=loss_function,
                          optimizer_choice=optimizer_choice)


@bp.route('/start_comparison', methods=['POST'])
def start_comparison():
    """Start algorithm comparison"""
    _begin_run()

    dataset = request.form.get('dataset', 'mnist')
    num_clients = int(request.form.get('num_clients', 5))
    local_epochs = int(request.form.get('local_epochs', 3))
    global_epochs = int(request.form.get('global_epochs', 50))
    num_qubits = int(request.form.get('num_qubits', 8))
    num_layers = int(request.form.get('num_layers', 2))
    learning_rate = float(request.form.get('learning_rate', 0.01))
    sporadic_p = float(request.form.get('sporadic_p', 0.1))
    theta_walk = float(request.form.get('theta_walk', math.pi / 6))
    phi_walk = float(request.form.get('phi_walk', 0.0))
    nisq_error_rate = float(request.form.get('nisq_error_rate', 0.01))
    data_fraction = float(request.form.get('data_fraction', 0.1))
    
    # Get selected algorithms
    algorithms = request.form.getlist('algorithms')
    if not algorithms:
        algorithms = ['kiqfl', 'fedavg', 'fqngd', 'centralized']
    
    return render_template('comparison_results.html',
                          dataset=dataset,
                          num_clients=num_clients,
                          local_epochs=local_epochs,
                          global_epochs=global_epochs,
                          num_qubits=num_qubits,
                          num_layers=num_layers,
                          learning_rate=learning_rate,
                          sporadic_p=sporadic_p,
                          theta_walk=theta_walk,
                          phi_walk=phi_walk,
                          nisq_error_rate=nisq_error_rate,
                          data_fraction=data_fraction,
                          algorithms=','.join(algorithms))


@bp.route('/stop_simulation', methods=['POST'])
def stop_simulation():
    """Ask the active SSE training loop to stop after the current epoch."""
    data = request.get_json(silent=True) or {}
    _request_stop()
    checkpoint = _build_checkpoint(
        data.get('run_type') or 'classification',
        data.get('params') or {},
        data.get('completed_epochs', 0),
    )
    return jsonify(checkpoint)


@bp.route('/stream_simulation')
def stream_simulation():
    """Stream single algorithm simulation"""
    fl_type = request.args.get('fl_type', 'quantum_federated')
    dataset_name = request.args.get('dataset', 'mnist')
    num_clients = int(request.args.get('num_clients', 5))
    local_epochs = int(request.args.get('local_epochs', 3))
    global_epochs = int(request.args.get('global_epochs', 50))
    num_qubits = int(request.args.get('num_qubits', 8))
    num_layers = int(request.args.get('num_layers', 2))
    learning_rate = float(request.args.get('learning_rate', 0.01))
    sporadic_p = float(request.args.get('sporadic_p', 0.1))
    theta_walk = float(request.args.get('theta_walk', math.pi / 6))
    phi_walk = float(request.args.get('phi_walk', 0.0))
    nisq_error_rate = float(request.args.get('nisq_error_rate', 0.01))
    data_fraction = float(request.args.get('data_fraction', 0.1))

    stream_params = {
        'fl_type': fl_type,
        'dataset': dataset_name,
        'num_clients': num_clients,
        'local_epochs': local_epochs,
        'global_epochs': global_epochs,
        'num_qubits': num_qubits,
        'num_layers': num_layers,
        'learning_rate': learning_rate,
        'sporadic_p': sporadic_p,
        'theta_walk': theta_walk,
        'phi_walk': phi_walk,
        'nisq_error_rate': nisq_error_rate,
        'data_fraction': data_fraction,
    }

    def generate():
        global last_simulation_results
        run_id = _begin_run()

        try:
            yield _sse({'status': 'Connected — loading dataset (first run may download MNIST)…'})
            if _should_stop(run_id):
                yield _sse({
                    'complete': True,
                    'stopped': True,
                    'checkpoint': _build_checkpoint('classification', stream_params, 0),
                })
                return

            trainset, testset, input_channels, image_size, num_classes = load_dataset(
                dataset_name, data_fraction
            )
            if _should_stop(run_id):
                yield _sse({
                    'complete': True,
                    'stopped': True,
                    'checkpoint': _build_checkpoint('classification', stream_params, 0),
                })
                return

            yield _sse({'status': f'Dataset ready ({dataset_name}). Starting training…'})

            config = {
                'num_clients': num_clients,
                'local_epochs': local_epochs,
                'global_epochs': global_epochs,
                'num_qubits': num_qubits,
                'num_layers': num_layers,
                'learning_rate': learning_rate,
                'sporadic_p': sporadic_p,
                'theta_walk': theta_walk,
                'phi_walk': phi_walk,
                'nisq_error_rate': nisq_error_rate,
                'input_channels': input_channels,
                'image_size': image_size,
                'num_classes': num_classes
            }

            results = []
            runners = {
                'federated': run_fedavg,
                'quantum_federated': run_kiqfl,
                'fqngd': run_fqngd,
                'centralized': run_centralized,
            }
            runner = runners.get(fl_type)
            if runner is None:
                yield _sse({'error': f'Unknown fl_type: {fl_type}'})
                return

            for result in runner(trainset, testset, config):
                results.append(result)
                yield _sse(result)
                if _should_stop(run_id):
                    last_simulation_results = {'single': results}
                    done = result.get('global_epoch', len(results))
                    yield _sse({
                        'complete': True,
                        'stopped': True,
                        'checkpoint': _build_checkpoint('classification', stream_params, done),
                    })
                    return

            if _should_stop(run_id):
                last_simulation_results = {'single': results}
                yield _sse({
                    'complete': True,
                    'stopped': True,
                    'checkpoint': _build_checkpoint(
                        'classification', stream_params, len(results)
                    ),
                })
                return

            last_simulation_results = {'single': results}
            yield _sse({'complete': True})

        except Exception as e:
            yield _sse({'error': str(e)})

    return Response(stream_with_context(generate()), headers=_sse_headers())


@bp.route('/stream_comparison')
def stream_comparison():
    """Stream algorithm comparison results"""
    dataset_name = request.args.get('dataset', 'mnist')
    num_clients = int(request.args.get('num_clients', 5))
    local_epochs = int(request.args.get('local_epochs', 3))
    global_epochs = int(request.args.get('global_epochs', 50))
    num_qubits = int(request.args.get('num_qubits', 8))
    num_layers = int(request.args.get('num_layers', 2))
    learning_rate = float(request.args.get('learning_rate', 0.01))
    sporadic_p = float(request.args.get('sporadic_p', 0.1))
    theta_walk = float(request.args.get('theta_walk', math.pi / 6))
    phi_walk = float(request.args.get('phi_walk', 0.0))
    nisq_error_rate = float(request.args.get('nisq_error_rate', 0.01))
    data_fraction = float(request.args.get('data_fraction', 0.1))
    algorithms = request.args.get('algorithms', 'kiqfl,fedavg,fqngd,centralized').split(',')

    stream_params = {
        'dataset': dataset_name,
        'num_clients': num_clients,
        'local_epochs': local_epochs,
        'global_epochs': global_epochs,
        'num_qubits': num_qubits,
        'num_layers': num_layers,
        'learning_rate': learning_rate,
        'sporadic_p': sporadic_p,
        'theta_walk': theta_walk,
        'phi_walk': phi_walk,
        'nisq_error_rate': nisq_error_rate,
        'data_fraction': data_fraction,
        'algorithms': ','.join(a.strip() for a in algorithms if a.strip()),
    }

    def generate():
        global last_simulation_results
        all_results = {}
        run_id = _begin_run()

        try:
            yield _sse({'status': 'Connected — loading dataset (first run may download MNIST)…'})
            trainset, testset, input_channels, image_size, num_classes = load_dataset(
                dataset_name, data_fraction
            )
            if _should_stop(run_id):
                yield _sse({
                    'complete': True,
                    'stopped': True,
                    'checkpoint': _build_checkpoint('comparison', stream_params, 0),
                })
                return

            yield _sse({
                'status': (
                    f'Dataset ready. Running {len([a for a in algorithms if a.strip()])} '
                    f'algorithm(s), {global_epochs} round(s) — first epoch can take minutes on CPU.'
                )
            })

            config = {
                'num_clients': num_clients,
                'local_epochs': local_epochs,
                'global_epochs': global_epochs,
                'num_qubits': num_qubits,
                'num_layers': num_layers,
                'learning_rate': learning_rate,
                'sporadic_p': sporadic_p,
                'theta_walk': theta_walk,
                'phi_walk': phi_walk,
                'nisq_error_rate': nisq_error_rate,
                'input_channels': input_channels,
                'image_size': image_size,
                'num_classes': num_classes
            }

            runners = {
                'fedavg': run_fedavg,
                'kiqfl': run_kiqfl,
                'fqngd': run_fqngd,
                'centralized': run_centralized,
            }

            for algo in algorithms:
                if _should_stop(run_id):
                    break
                algo = algo.strip().lower()
                runner = runners.get(algo)
                if runner is None:
                    yield _sse({'status': f'Skipping unknown algorithm: {algo}'})
                    continue

                results = []
                yield _sse({'status': f'Starting {algo.upper()}…'})

                for result in runner(trainset, testset, config):
                    results.append(result)
                    yield _sse(result)
                    if _should_stop(run_id):
                        all_results[algo] = results
                        last_simulation_results = all_results
                        done = result.get('global_epoch', len(results))
                        yield _sse({
                            'complete': True,
                            'stopped': True,
                            'checkpoint': _build_checkpoint('comparison', stream_params, done),
                        })
                        return

                all_results[algo] = results

            last_simulation_results = all_results
            if _should_stop(run_id):
                yield _sse({
                    'complete': True,
                    'stopped': True,
                    'checkpoint': _build_checkpoint('comparison', stream_params, 0),
                })
            else:
                yield _sse({'complete': True})

        except Exception as e:
            import traceback
            yield _sse({'error': str(e), 'traceback': traceback.format_exc()})

    return Response(stream_with_context(generate()), headers=_sse_headers())


@bp.route('/download_csv')
def download_csv():
    """Download results as CSV"""
    global last_simulation_results
    
    if not last_simulation_results:
        return "No simulation results available.", 400
    
    si = io.StringIO()
    writer = csv.writer(si)
    
    # Check if comparison or single
    if 'single' in last_simulation_results:
        results = last_simulation_results['single']
        writer.writerow(['Method', 'Global Epoch', 'Loss', 'Accuracy', 'Fidelity'])
        for res in results:
            writer.writerow([
                res.get('method', 'Unknown'),
                res['global_epoch'],
                res.get('loss', res.get('mse', 0)),
                res.get('accuracy', 0),
                res.get('fidelity', 0)
            ])
    else:
        writer.writerow(['Method', 'Global Epoch', 'Loss', 'Accuracy', 'Fidelity'])
        for method, results in last_simulation_results.items():
            for res in results:
                writer.writerow([
                    res.get('method', method),
                    res['global_epoch'],
                    res.get('loss', res.get('mse', 0)),
                    res.get('accuracy', 0),
                    res.get('fidelity', 0)
                ])
    
    output = si.getvalue()
    return Response(
        output,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=simulation_results.csv"}
    )


@bp.route('/download_comparison_csv')
def download_comparison_csv():
    """Download comparison results as formatted CSV"""
    global last_simulation_results
    
    if not last_simulation_results:
        return "No results available.", 400
    
    si = io.StringIO()
    writer = csv.writer(si)
    
    # Header
    writer.writerow(['Epoch'] + [f'{m}_Loss' for m in last_simulation_results.keys()] + 
                    [f'{m}_Accuracy' for m in last_simulation_results.keys()] +
                    [f'{m}_Fidelity' for m in last_simulation_results.keys()])
    
    # Find max epochs
    max_epochs = max(len(r) for r in last_simulation_results.values())
    
    for epoch in range(max_epochs):
        row = [epoch + 1]
        for method in last_simulation_results.keys():
            results = last_simulation_results[method]
            if epoch < len(results):
                row.append(results[epoch].get('loss', 0))
            else:
                row.append('')
        for method in last_simulation_results.keys():
            results = last_simulation_results[method]
            if epoch < len(results):
                row.append(results[epoch].get('accuracy', 0))
            else:
                row.append('')
        for method in last_simulation_results.keys():
            results = last_simulation_results[method]
            if epoch < len(results):
                row.append(results[epoch].get('fidelity', 0))
            else:
                row.append('')
        writer.writerow(row)
    
    output = si.getvalue()
    return Response(
        output,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=comparison_results.csv"}
    )


# ========================================================================================
# REGRESSION ROUTES (Custom Dataset)
# ========================================================================================

@bp.route('/get_headers', methods=['POST'])
def get_headers():
    if 'upload_dataset' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['upload_dataset']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        df = pd.read_csv(file)
        headers = df.columns.tolist()
        df.to_csv("temp_uploaded_data.csv", index=False)
        return jsonify({'headers': headers})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/start_simulation_custom_final', methods=['POST'])
def start_simulation_custom_final():
    selected_features = {}
    for key in request.form:
        if key.startswith("include_"):
            col_name = key[len("include_"):]
            selected_features[col_name] = request.form.get(key)
    
    output_feature = request.form.get("output_feature")
    
    global selected_feature_config
    selected_feature_config = {
        "selected_features": selected_features,
        "output_feature": output_feature
    }
    
    return redirect(url_for('main.simulation_regression'))


@bp.route('/simulation_regression')
def simulation_regression():
    global selected_feature_config
    return render_template('simulation_results_regression.html', 
                          selected_feature_config=selected_feature_config)


@bp.route('/stream_simulation_regression')
def stream_simulation_regression():
    global selected_feature_config
    
    if not os.path.exists("temp_uploaded_data.csv") or not selected_feature_config:
        return "No data or configuration available.", 400
    
    df = pd.read_csv("temp_uploaded_data.csv")
    input_columns = [col for col, inc in selected_feature_config["selected_features"].items() if inc == "yes"]
    output_column = selected_feature_config["output_feature"]
    
    if output_column not in df.columns:
        return "Output column not found.", 400
    
    X = df[input_columns].values.astype(np.float32)
    y = df[output_column].values.astype(np.float32).reshape(-1, 1)
    
    num_qubits = int(request.args.get('num_qubits', 8))
    num_layers = int(request.args.get('num_layers', 2))
    global_epochs = int(request.args.get('global_epochs', 50))
    learning_rate = float(request.args.get('learning_rate', 0.01))
    
    # Pad and normalize
    target_dim = 2 ** num_qubits
    if X.shape[1] > target_dim:
        X = X[:, :target_dim]
    elif X.shape[1] < target_dim:
        X = np.concatenate([X, np.zeros((X.shape[0], target_dim - X.shape[1]))], axis=1)
    
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    X = X / norms
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    model = QNN_Regressor(n=num_qubits, L=num_layers, use_qaoa=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no"
    }
    
    def generate():
        for epoch in range(global_epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
            
            model.eval()
            total_loss = 0
            total_samples = 0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    total_loss += loss.item() * X_batch.size(0)
                    total_samples += X_batch.size(0)
            
            avg_loss = total_loss / total_samples
            result = {'global_epoch': epoch + 1, 'mse': avg_loss, 'fidelity': 0.0}
            yield f"data: {json.dumps(result)}\n\n"
            time.sleep(0.1)
        
        yield f"data: {json.dumps({'complete': True})}\n\n"
    
    return Response(generate(), headers=headers)


