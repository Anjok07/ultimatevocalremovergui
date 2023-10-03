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
    """A class representing a remote repository with models."""
    def __init__(self, models: tp.Dict[str, str]):
        """Initialize the RemoteRepo with a dictionary of models."""
        self._models = models

    def has_model(self, sig: str) -> bool:
        """Check if a model with the given signature exists in the repository."""
        return sig in self._models

    def get_model(self, sig: str) -> Model:
        """Get a Model object based on the given signature."""
        try:
            url = self._models[sig]
        except KeyError:
            raise ModelLoadingError(f'Could not find a pre-trained model with signature {sig}.')
        pkg = torch.hub.load_state_dict_from_url(url, map_location='cpu', check_hash=True)
        return load_model(pkg)


class LocalRepo(ModelOnlyRepo):
    """
    A class representing a local repository for pre-trained models.
    Inherits from the 'ModelOnlyRepo' class.
    """
    def __init__(self, root: Path):
        """
        Initialize a new 'LocalRepo' instance with the root path.

        Parameters:
            root (Path): The root path of the repository.
        """
        self.root = root
        self.scan()

    def scan(self):
        """
        Scan the repository and populate the models and checksums dictionaries.

        """
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
                    print('Whats xp? ', xp_sig)
                    raise ModelLoadingError(
                        f'Duplicate pre-trained model exist for signature {xp_sig}. '
                        'Please delete all but one.')
                self._models[xp_sig] = file

    def has_model(self, sig: str) -> bool:
        """
        Check if a model with the specified signature exists.

        Parameters:
            sig (str): The model signature.

        Returns:
            bool: True if the model exists, False otherwise.
        """
        return sig in self._models

    def get_model(self, sig: str) -> Model:
        """
        Retrieve a model with the specified signature.

        Parameters:
            sig (str): The model signature.

        Returns:
            Model: The retrieved model.

        Raises:
            ModelLoadingError: If the model with the specified signature cannot be found.
        """
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
        """Initialize the BagOnlyRepo object.

        Parameters:
            root (Path): The root directory to scan for YAML files.
            model_repo (ModelOnlyRepo): The model repository object.
        """
        self.root = root
        self.model_repo = model_repo
        self.scan()

    def scan(self):
        """Scan the root directory for YAML files and store them in a dictionary.
        """
        self._bags = {}
        for file in self.root.iterdir():
            if file.suffix == '.yaml':
                self._bags[file.stem] = file

    def has_model(self, name: str) -> bool:
        """Check if a model exists in the bag.

        Parameters:
            name (str): The name of the model.

        Returns:
            bool: True if the model exists, False otherwise.
        """
        return name in self._bags

    def get_model(self, name: str) -> BagOfModels:
        """Retrieve a bag of models from a YAML file.

        Parameters:
            name (str): The name of the YAML file.

        Returns:
            BagOfModels: The bag of models.
        """
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
    """This class represents a repository for models."""
    def __init__(self, model_repo: ModelOnlyRepo, bag_repo: BagOnlyRepo):
        """Initialize the AnyModelRepo with a model repository and a bag repository."""
        self.model_repo = model_repo
        self.bag_repo = bag_repo

    def has_model(self, name_or_sig: str) -> bool:
        """Check if a model exists in the repository.

        This method checks if a model with the specified name or signature exists in the
        model repository or the bag repository. It returns True if the model exists and
        False otherwise.

        Parameters:
            name_or_sig (str): The name or signature of the model.

        Returns:
            bool: True if the model exists, False otherwise.
        """
        return self.model_repo.has_model(name_or_sig) or self.bag_repo.has_model(name_or_sig)

    def get_model(self, name_or_sig: str) -> AnyModel:
        """Get a model from the repository.

        This method retrieves a model with the specified name or signature from the model
        repository or the bag repository. If the model exists in the model repository, it
        is returned. Otherwise, the model is retrieved from the bag repository.

        Parameters:
            name_or_sig (str): The name or signature of the model.

        Returns:
            AnyModel: The retrieved model.
        """
        print('name_or_sig: ', name_or_sig)
        if self.model_repo.has_model(name_or_sig):
            return self.model_repo.get_model(name_or_sig)
        else:
            return self.bag_repo.get_model(name_or_sig)
