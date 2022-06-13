# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Represents a model repository, including pre-trained models and bags of models.
A repo can either be the main remote repository stored in AWS, or a local repository
with your own models.
"""

from hashlib import sha256
from pathlib import Path
import typing as tp

import torch
import yaml

from .apply import BagOfModels, Model
from .states import load_model


AnyModel = tp.Union[Model, BagOfModels]


class ModelLoadingError(RuntimeError):
    pass


def check_checksum(path: Path, checksum: str):
    sha = sha256()
    with open(path, 'rb') as file:
        while True:
            buf = file.read(2**20)
            if not buf:
                break
            sha.update(buf)
    actual_checksum = sha.hexdigest()[:len(checksum)]
    if actual_checksum != checksum:
        raise ModelLoadingError(f'Invalid checksum for file {path}, '
                                f'expected {checksum} but got {actual_checksum}')


class ModelOnlyRepo:
    """Base class for all model only repos.
    """
    def has_model(self, sig: str) -> bool:
        raise NotImplementedError()

    def get_model(self, sig: str) -> Model:
        raise NotImplementedError()


class RemoteRepo(ModelOnlyRepo):
    def __init__(self, root_url: str, remote_files: tp.List[str]):
        if not root_url.endswith('/'):
            root_url += '/'
        self._models: tp.Dict[str, str] = {}
        for file in remote_files:
            sig, checksum = file.split('.')[0].split('-')
            assert sig not in self._models
            self._models[sig] = root_url + file

    def has_model(self, sig: str) -> bool:
        return sig in self._models

    def get_model(self, sig: str) -> Model:
        try:
            url = self._models[sig]
        except KeyError:
            raise ModelLoadingError(f'Could not find a pre-trained model with signature {sig}.')
        pkg = torch.hub.load_state_dict_from_url(url, map_location='cpu', check_hash=True)
        return load_model(pkg)


class LocalRepo(ModelOnlyRepo):
    def __init__(self, root: Path):
        self.root = root
        self.scan()

    def scan(self):
        self._models = {}
        self._checksums = {}
        for file in self.root.iterdir():
            if file.suffix == '.th':
                if '-' in file.stem:
                    xp_sig, checksum = file.stem.split('-')
                    self._checksums[xp_sig] = checksum
                else:
                    xp_sig = file.stem
                if xp_sig in self._models:
                    raise ModelLoadingError(
                        f'Duplicate pre-trained model exist for signature {xp_sig}. '
                        'Please delete all but one.')
                self._models[xp_sig] = file

    def has_model(self, sig: str) -> bool:
        return sig in self._models

    def get_model(self, sig: str) -> Model:
        try:
            file = self._models[sig]
        except KeyError:
            raise ModelLoadingError(f'Could not find pre-trained model with signature {sig}.')
        if sig in self._checksums:
            check_checksum(file, self._checksums[sig])
        return load_model(file)


class BagOnlyRepo:
    """Handles only YAML files containing bag of models, leaving the actual
    model loading to some Repo.
    """
    def __init__(self, root: Path, model_repo: ModelOnlyRepo):
        self.root = root
        self.model_repo = model_repo
        self.scan()

    def scan(self):
        self._bags = {}
        for file in self.root.iterdir():
            if file.suffix == '.yaml':
                self._bags[file.stem] = file

    def has_model(self, name: str) -> bool:
        return name in self._bags

    def get_model(self, name: str) -> BagOfModels:
        try:
            yaml_file = self._bags[name]
        except KeyError:
            raise ModelLoadingError(f'{name} is neither a single pre-trained model or '
                                    'a bag of models.')
        bag = yaml.safe_load(open(yaml_file))
        signatures = bag['models']
        models = [self.model_repo.get_model(sig) for sig in signatures]
        weights = bag.get('weights')
        segment = bag.get('segment')
        return BagOfModels(models, weights, segment)


class AnyModelRepo:
    def __init__(self, model_repo: ModelOnlyRepo, bag_repo: BagOnlyRepo):
        self.model_repo = model_repo
        self.bag_repo = bag_repo

    def has_model(self, name_or_sig: str) -> bool:
        return self.model_repo.has_model(name_or_sig) or self.bag_repo.has_model(name_or_sig)

    def get_model(self, name_or_sig: str) -> AnyModel:
        if self.model_repo.has_model(name_or_sig):
            return self.model_repo.get_model(name_or_sig)
        else:
            return self.bag_repo.get_model(name_or_sig)
