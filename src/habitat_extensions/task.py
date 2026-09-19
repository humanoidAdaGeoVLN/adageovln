"""Project-local Habitat dataset extensions for RxR VLN-CE."""

import gzip
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import attr
from habitat.config.default_structured_configs import DatasetConfig
from habitat.core.dataset import Dataset
from habitat.core.registry import registry
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.nav import NavigationGoal
from habitat.tasks.vln.vln import VLNEpisode
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig


DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"
ALL_LANGUAGES_MASK = "*"
ALL_ROLES_MASK = "*"


@attr.s(auto_attribs=True)
class ExtendedInstructionData:
    instruction_text: str = attr.ib(
        default=None, validator=not_none_validator
    )
    instruction_id: Optional[str] = attr.ib(default=None)
    language: Optional[str] = attr.ib(default=None)
    annotator_id: Optional[str] = attr.ib(default=None)
    edit_distance: Optional[float] = attr.ib(default=None)
    timed_instruction: Optional[
        List[Dict[str, Union[float, str]]]
    ] = attr.ib(default=None)
    instruction_tokens: Optional[List[str]] = attr.ib(default=None)
    split: Optional[str] = attr.ib(default=None)


@attr.s(auto_attribs=True, kw_only=True)
class VLNExtendedEpisode(VLNEpisode):
    goals: Optional[List[NavigationGoal]] = attr.ib(default=None)
    reference_path: Optional[List[List[float]]] = attr.ib(default=None)
    instruction: ExtendedInstructionData = attr.ib(
        default=None, validator=not_none_validator
    )
    trajectory_id: Optional[Union[int, str]] = attr.ib(default=None)


@registry.register_dataset(name="RxRVLNCE-v1")
class RxRVLNCEDatasetV1(Dataset):
    """Load RxR VLN-CE episodes by annotation role and language."""

    episodes: List[VLNExtendedEpisode]
    annotation_roles: List[str] = ["guide", "follower"]
    supported_languages: List[str] = [
        "en-US",
        "en-IN",
        "hi-IN",
        "te-IN",
    ]

    def __init__(self, config: Optional[DictConfig] = None) -> None:
        self.episodes = []
        self.config = config

        if config is None:
            return

        self.roles = self.extract_roles_from_config(config)
        self.languages = list(config.languages)

        for role in self.roles:
            dataset_path = config.data_path.format(
                split=config.split, role=role
            )
            with gzip.open(dataset_path, "rt") as dataset_file:
                self.from_json(
                    dataset_file.read(), scenes_dir=config.scenes_dir
                )

        if ALL_LANGUAGES_MASK not in self.languages:
            requested_languages = set(self.languages)
            unsupported_languages = requested_languages.difference(
                self.supported_languages
            )
            if unsupported_languages:
                raise ValueError(
                    "Unsupported RxR languages: "
                    f"{sorted(unsupported_languages)}"
                )
            self.episodes = [
                episode
                for episode in self.episodes
                if episode.instruction.language in requested_languages
            ]

        self.episodes = list(
            filter(self.build_content_scenes_filter(config), self.episodes)
        )

    @classmethod
    def check_config_paths_exist(cls, config: DictConfig) -> bool:
        return os.path.exists(config.scenes_dir) and all(
            os.path.exists(
                config.data_path.format(split=config.split, role=role)
            )
            for role in cls.extract_roles_from_config(config)
        )

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)

        for episode_data in deserialized["episodes"]:
            episode = VLNExtendedEpisode(**episode_data)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]
                episode.scene_id = os.path.join(
                    scenes_dir, episode.scene_id
                )

            episode.instruction = ExtendedInstructionData(
                **episode.instruction
            )
            episode.instruction.split = self.config.split

            if episode.goals is not None:
                for goal_index, goal in enumerate(episode.goals):
                    episode.goals[goal_index] = NavigationGoal(**goal)

            self.episodes.append(episode)

    @classmethod
    def extract_roles_from_config(cls, config: DictConfig) -> List[str]:
        if ALL_ROLES_MASK in config.roles:
            return list(cls.annotation_roles)

        roles = list(config.roles)
        unsupported_roles = set(roles).difference(cls.annotation_roles)
        if unsupported_roles:
            raise ValueError(
                f"Unsupported RxR roles: {sorted(unsupported_roles)}"
            )
        return roles


@dataclass
class RxRVLNCEDatasetConfig(DatasetConfig):
    type: str = "RxRVLNCE-v1"
    split: str = "train"
    scenes_dir: str = "data/scene_datasets/"
    roles: List[str] = field(default_factory=lambda: ["guide"])
    languages: List[str] = field(default_factory=lambda: ["en-US"])
    data_path: str = (
        "data/datasets/rxr/{split}/{split}_{role}.json.gz"
    )


ConfigStore.instance().store(
    package="habitat.dataset",
    group="habitat/dataset",
    name="rxrvlnce_v1",
    node=RxRVLNCEDatasetConfig,
)
