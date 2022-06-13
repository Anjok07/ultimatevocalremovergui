# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Loading pretrained models.
"""

import logging
from pathlib import Path
import typing as tp

from dora.log import fatal

from .hdemucs import HDemucs
from .repo import RemoteRepo, LocalRepo, ModelOnlyRepo, BagOnlyRepo, AnyModelRepo, ModelLoadingError  # noqa

logger = logging.getLogger(__name__)
ROOT_URL = "https://dl.fbaipublicfiles.com/demucs/mdx_final/"
REMOTE_ROOT = Path(__file__).parent / 'remote'

SOURCES = ["drums", "bass", "other", "vocals"]


def demucs_unittest():
    model = HDemucs(channels=4, sources=SOURCES)
    return model


def add_model_flags(parser):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("-s", "--sig", help="Locally trained XP signature.")
    group.add_argument("-n", "--name", default="mdx_extra_q",
                       help="Pretrained model name or signature. Default is mdx_extra_q.")
    parser.add_argument("--repo", type=Path,
                        help="Folder containing all pre-trained models for use with -n.")


def get_model(name: str,
              repo: tp.Optional[Path] = None):
    """`name` must be a bag of models name or a pretrained signature
    from the remote AWS model repo or the specified local repo if `repo` is not None.
    """
    if name == 'demucs_unittest':
        return demucs_unittest()
    model_repo: ModelOnlyRepo
    if repo is None:
        remote_files = [line.strip()
                        for line in (REMOTE_ROOT / 'files.txt').read_text().split('\n')
                        if line.strip()]
        model_repo = RemoteRepo(ROOT_URL, remote_files)
        bag_repo = BagOnlyRepo(REMOTE_ROOT, model_repo)
    else:
        if not repo.is_dir():
            fatal(f"{repo} must exist and be a directory.")
        model_repo = LocalRepo(repo)
        bag_repo = BagOnlyRepo(repo, model_repo)
    any_repo = AnyModelRepo(model_repo, bag_repo)
    return any_repo.get_model(name)


def get_model_from_args(args):
    """
    Load local model package or pre-trained model.
    """
    return get_model(name=args.name, repo=args.repo)
