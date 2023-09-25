# -*- coding: utf-8 -*-

"""
Matchering - Audio Matching and Mastering Python Library
Copyright (C) 2016-2022 Sergree

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
import soundfile as sf


class Result:
    def __init__(
        self, file: str, subtype: str, use_limiter: bool = True, normalize: bool = True
    ):
        _, file_ext = os.path.splitext(file)
        file_ext = file_ext[1:].upper()
        if not sf.check_format(file_ext):
            raise TypeError(f"{file_ext} format is not supported")
        if not sf.check_format(file_ext, subtype):
            raise TypeError(f"{file_ext} format does not have {subtype} subtype")
        self.file = file
        self.subtype = subtype
        self.use_limiter = use_limiter
        self.normalize = normalize


def pcm16(file: str) -> Result:
    return Result(file, "PCM_16")

def pcm24(file: str) -> Result:
    return Result(file, "FLOAT")

def save_audiofile(file: str, wav_set="PCM_16") -> Result:
    return Result(file, wav_set)
