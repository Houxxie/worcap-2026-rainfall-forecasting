"""Monthly map datasets and chronological epoch selection."""
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from .network import RainfallUNet, seed_everything
from research.common.inputs import require, write_json


def split_inner(training, reference, validation_months=24):
    training, reference = pd.DatetimeIndex(training), pd.DatetimeIndex(reference)
    validation = training[-validation_months:]
    cutoff = validation[0] - pd.DateOffset(months=4)
    fit = training[training <= cutoff]
    refs = reference[reference <= cutoff]
    require(len(fit) >= 24 and fit[-1] < validation[0], 'Insufficient chronological inner training.')
    return fit, refs, validation


class MonthlyMaps(Dataset):
    def __init__(self, context, dates, directory, training=False, scaling=None):
        self.dates = pd.DatetimeIndex(dates)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        d = context.data
        n, h, w = len(self.dates), d.rain.sizes['lat'], d.rain.sizes['lon']
        self.x = np.lib.format.open_memmap(self.directory / 'features.npy', mode='w+', dtype='float32', shape=(n, 31, h, w))
        self.y = np.lib.format.open_memmap(self.directory / 'target.npy', mode='w+', dtype='float32', shape=(n, h, w))
        self.base = np.lib.format.open_memmap(self.directory / 'climate.npy', mode='w+', dtype='float32', shape=(n, h, w))
        totals, squares, pixels = np.zeros(31), np.zeros(31), 0
        for i, t in enumerate(self.dates):
            x = context.features(t, training)
            self.x[i] = x
            self.base[i] = x[22]
            self.y[i] = d.rain.sel(time=t).values
            if scaling is None:
                require(training, 'Only training maps may fit scaling.')
                flat = x.reshape(31, -1).astype('float64')
                totals += flat.sum(1)
                squares += np.square(flat).sum(1)
                pixels += flat.shape[1]
        if scaling is None:
            mean = totals / pixels
            scale = np.sqrt(np.maximum(squares / pixels - mean ** 2, 0))
            self.scaling = dict(mean=mean.astype('float32'), scale=np.maximum(scale, 1e-6).astype('float32'))
        else:
            self.scaling = {k: np.array(v, copy=True) for k, v in scaling.items()}
        for arr in [self.x, self.y, self.base]:
            require(all(np.isfinite(arr[i]).all() for i in range(n)), 'Invalid map cache.')
            arr.flush()

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, i):
        x = (self.x[i] - self.scaling['mean'][:, None, None]) / self.scaling['scale'][:, None, None]
        return torch.from_numpy(x), torch.from_numpy(self.y[i] - self.base[i]), torch.from_numpy(self.base[i].copy())

    def close(self):
        # Only files created by this instance; no recursive or input-directory deletion.
        for arr in [self.x, self.y, self.base]:
            arr.flush()
            arr._mmap.close()
        for name in ['features.npy', 'target.npy', 'climate.npy']:
            (self.directory / name).unlink()
        self.directory.rmdir()


def inference_maps(model, dataset, device, batch_size=2):
    model.eval()
    out = []
    with torch.inference_mode():
        for x, _, base in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
            anomaly = model(x.to(device)).cpu()
            out.append(torch.clamp(base + anomaly, min=0).numpy())
    result = np.concatenate(out)
    require(np.isfinite(result).all(), 'Nonfinite network output.')
    return result


def fit_network(train, config, folder, device, validation=None, epochs=None):
    seed_everything(config['seed'])
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    model = RainfallUNet(widths=tuple(config['architecture']['channels'])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['optimizer']['learning_rate'], weight_decay=config['optimizer']['weight_decay'])
    generator = torch.Generator().manual_seed(config['seed'])
    loader = DataLoader(train, batch_size=config['batch_size'], shuffle=True, generator=generator, num_workers=0)
    limit = epochs if epochs is not None else config['max_epochs']
    history, best, best_epoch, stale = [], float('inf'), 0, 0
    for epoch in range(1, limit + 1):
        t0, total, count = time.perf_counter(), 0., 0
        model.train()
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = torch.mean((prediction - y) ** 2)
            require(torch.isfinite(loss).item(), 'Training loss diverged.')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['optimizer']['gradient_clip_norm'], error_if_nonfinite=True)
            optimizer.step()
            total += loss.item() * y.numel()
            count += y.numel()
        row = dict(epoch=epoch, training_mse=total / count, seconds=time.perf_counter() - t0)
        if validation is not None:
            pred = inference_maps(model, validation, device, config['batch_size'])
            truth = validation.y.astype('float64')
            rmse = float(np.sqrt(np.mean((pred.astype('float64') - truth) ** 2)))
            row['inner_rmse'] = rmse
            if rmse < best:
                best, best_epoch, stale = rmse, epoch, 0
            else:
                stale += 1
        history.append(row)
        write_json(folder / 'training_history.json', history)
        print(folder.name, row, flush=True)
        if validation is not None and stale >= config['patience']:
            break
    selected = best_epoch if validation is not None else limit
    # Inner weights are never evaluated on the outer block. Refit starts from seed.
    torch.save(dict(state_dict=model.state_dict(), widths=config['architecture']['channels'],
        epochs_run=epoch, selected_epochs=selected, seed=config['seed']), folder / 'network.pt')
    np.savez_compressed(folder / 'scaling.npz', **train.scaling)
    write_json(folder / 'training.json', dict(selected_epochs=selected,
        parameter_count=sum(p.numel() for p in model.parameters()), train_months=len(train),
        train_start=str(train.dates[0].date()), train_end=str(train.dates[-1].date()),
        inner_selection=validation is not None, device=str(device)))
    return model, selected
