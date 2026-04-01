from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    maps: tuple[str, ...]
    dynamic_obstacle_count: int = 0
    mapping_mode: str = "ground_truth"
    min_goal_distance: float = 160.0

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "maps": list(self.maps),
            "dynamic_obstacle_count": self.dynamic_obstacle_count,
            "mapping_mode": self.mapping_mode,
            "min_goal_distance": self.min_goal_distance,
        }


@dataclass(frozen=True)
class Curriculum:
    name: str
    stages: tuple[CurriculumStage, ...]


def list_curricula() -> list[str]:
    return sorted(CURRICULA.keys())


def resolve_curriculum(
    name: str,
    maps_override: list[str] | None = None,
) -> Curriculum:
    if not name:
        raise ValueError("curriculum name must be non-empty")
    if name not in CURRICULA:
        raise ValueError(f"unsupported curriculum: {name!r}")

    curriculum = CURRICULA[name]
    if not maps_override:
        return curriculum

    maps = tuple(dict.fromkeys(maps_override))
    stages = tuple(
        CurriculumStage(
            name=stage.name,
            maps=maps,
            dynamic_obstacle_count=stage.dynamic_obstacle_count,
            mapping_mode=stage.mapping_mode,
            min_goal_distance=stage.min_goal_distance,
        )
        for stage in curriculum.stages
    )
    return Curriculum(name=curriculum.name, stages=stages)


def single_stage_curriculum(
    maps: list[str],
    dynamic_obstacle_count: int,
    mapping_mode: str,
    min_goal_distance: float,
) -> Curriculum:
    unique_maps = tuple(dict.fromkeys(maps)) or ("default",)
    return Curriculum(
        name="custom",
        stages=(
            CurriculumStage(
                name="baseline",
                maps=unique_maps,
                dynamic_obstacle_count=dynamic_obstacle_count,
                mapping_mode=mapping_mode,
                min_goal_distance=min_goal_distance,
            ),
        ),
    )


CURRICULA: dict[str, Curriculum] = {
    "easy": Curriculum(
        name="easy",
        stages=(
            CurriculumStage(
                name="easy-start",
                maps=("default",),
                dynamic_obstacle_count=0,
                mapping_mode="ground_truth",
                min_goal_distance=120.0,
            ),
            CurriculumStage(
                name="easy-multi-map",
                maps=("default", "blocks"),
                dynamic_obstacle_count=1,
                mapping_mode="ground_truth",
                min_goal_distance=150.0,
            ),
        ),
    ),
    "standard": Curriculum(
        name="standard",
        stages=(
            CurriculumStage(
                name="ground-truth-start",
                maps=("default", "blocks"),
                dynamic_obstacle_count=0,
                mapping_mode="ground_truth",
                min_goal_distance=140.0,
            ),
            CurriculumStage(
                name="ground-truth-dynamic",
                maps=("default", "maze", "blocks"),
                dynamic_obstacle_count=1,
                mapping_mode="ground_truth",
                min_goal_distance=180.0,
            ),
            CurriculumStage(
                name="sensor-driven-finish",
                maps=("default", "maze", "blocks"),
                dynamic_obstacle_count=2,
                mapping_mode="sensor_driven",
                min_goal_distance=200.0,
            ),
        ),
    ),
    "robust": Curriculum(
        name="robust",
        stages=(
            CurriculumStage(
                name="ground-truth-start",
                maps=("default", "blocks"),
                dynamic_obstacle_count=0,
                mapping_mode="ground_truth",
                min_goal_distance=150.0,
            ),
            CurriculumStage(
                name="ground-truth-dynamic",
                maps=("default", "maze", "blocks"),
                dynamic_obstacle_count=2,
                mapping_mode="ground_truth",
                min_goal_distance=190.0,
            ),
            CurriculumStage(
                name="sensor-driven-dynamic",
                maps=("default", "maze", "blocks"),
                dynamic_obstacle_count=3,
                mapping_mode="sensor_driven",
                min_goal_distance=210.0,
            ),
            CurriculumStage(
                name="generalization",
                maps=("generated_easy", "generated_medium", "maze"),
                dynamic_obstacle_count=3,
                mapping_mode="sensor_driven",
                min_goal_distance=220.0,
            ),
        ),
    ),
}
