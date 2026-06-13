import json
import tempfile
import unittest
from pathlib import Path

from src.datasets.physicalai_openh import PhysicalAIOpenHDataset, PhysicalAIOpenHError


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class TestPhysicalAIOpenHMetadata(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.leaf = "Surgical/example/task"
        meta = self.root / self.leaf / "meta"
        info = {
            "codebase_version": "v2.1",
            "robot_type": "testbot",
            "total_episodes": 5,
            "total_frames": 50,
            "total_tasks": 2,
            "chunks_size": 1000,
            "fps": 30,
            "splits": {"train": "0:3", "val": "3:4", "test": "4:5"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "observation.images.color": {"dtype": "video", "shape": [16, 16, 3]},
                "observation.state": {"dtype": "float32", "shape": [7]},
                "action": {"dtype": "float32", "shape": [4]},
                "timestamp": {"dtype": "float32", "shape": [1]},
            },
        }
        modality = {
            "video": {"color": {"original_key": "observation.images.color"}},
            "state": {"joint_angles": {"original_key": "observation.state"}},
            "action": {"eef": {"original_key": "action"}},
        }
        episodes = [
            {"episode_index": 0, "tasks": ["suturing"], "length": 10},
            {"episode_index": 1, "tasks": ["suturing"], "length": 10},
            {"episode_index": 2, "tasks": ["cutting"], "length": 10},
            {"episode_index": 3, "tasks": ["cutting"], "length": 10},
            {"episode_index": 4, "tasks": ["suturing"], "length": 10},
        ]
        write_json(meta / "info.json", info)
        write_json(meta / "modality.json", modality)
        write_jsonl(
            meta / "tasks.jsonl",
            [{"task_index": 0, "task": "suturing"}, {"task_index": 1, "task": "cutting"}],
        )
        write_jsonl(meta / "episodes.jsonl", episodes)

    def tearDown(self):
        self.tmp.cleanup()

    def test_parses_lerobot_v21_metadata_and_existing_splits(self):
        dataset = PhysicalAIOpenHDataset(
            dataset_roots=[self.leaf],
            local_repo_root=str(self.root),
            split="train",
            fps=4,
            frames_per_clip=4,
            require_action=True,
            require_state=True,
        )

        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset.datasets[0].video_keys, ["observation.images.color"])
        self.assertEqual(dataset.datasets[0].action_keys, ["action"])
        self.assertEqual(dataset.datasets[0].state_keys, ["observation.state"])
        self.assertEqual([episode.episode_index for episode in dataset.episodes], [0, 1, 2])

        val_dataset = PhysicalAIOpenHDataset(
            dataset_roots=[self.leaf],
            local_repo_root=str(self.root),
            split="val",
            fps=4,
            frames_per_clip=4,
        )
        self.assertEqual([episode.episode_index for episode in val_dataset.episodes], [3])

    def test_offline_default_requires_cache_or_local_root(self):
        with self.assertRaisesRegex(PhysicalAIOpenHError, "local_files_only=True"):
            PhysicalAIOpenHDataset(
                dataset_roots=[self.leaf],
                split="train",
                fps=4,
                frames_per_clip=4,
            )

    def test_cache_dir_alias_reads_local_mirror(self):
        dataset = PhysicalAIOpenHDataset(
            dataset_roots=[self.leaf],
            cache_dir=str(self.root),
            split="train",
            fps=4,
            frames_per_clip=4,
        )

        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset.required_metadata_files()[0], f"{self.leaf}/meta/episodes.jsonl")

    def test_task_and_embodiment_filters_are_episode_safe(self):
        dataset = PhysicalAIOpenHDataset(
            dataset_roots=[self.leaf],
            local_repo_root=str(self.root),
            split="train",
            fps=4,
            frames_per_clip=4,
            task_filter=["cutting"],
            embodiment_filter=["testbot"],
        )

        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.episodes[0].episode_index, 2)
        self.assertEqual(dataset.episodes[0].tasks, ("cutting",))

    def test_frame_step_sampling_uses_exact_stride(self):
        dataset = PhysicalAIOpenHDataset(
            dataset_roots=[self.leaf],
            local_repo_root=str(self.root),
            split="train",
            frame_step=2,
            frames_per_clip=4,
            random_clip_sampling=False,
        )

        self.assertEqual(dataset._sample_indices(dataset.episodes[0], video_fps=30).tolist(), [0, 2, 4, 6])

    def test_legacy_layout_fails_clearly(self):
        legacy_leaf = "Surgical/legacy/task"
        write_json(self.root / legacy_leaf / "meta" / "episode_000000_meta.json", {"length": 10})

        with self.assertRaisesRegex(PhysicalAIOpenHError, "missing meta/info.json"):
            PhysicalAIOpenHDataset(
                dataset_roots=[legacy_leaf],
                local_repo_root=str(self.root),
                split="train",
                fps=4,
                frames_per_clip=4,
            )

    def test_fallback_split_is_deterministic_when_info_has_no_splits(self):
        info_path = self.root / self.leaf / "meta" / "info.json"
        info = json.loads(info_path.read_text())
        info.pop("splits")
        write_json(info_path, info)

        first = PhysicalAIOpenHDataset(
            dataset_roots=[self.leaf],
            local_repo_root=str(self.root),
            split="val",
            fps=4,
            frames_per_clip=4,
            val_fraction=0.4,
            seed=7,
        )
        second = PhysicalAIOpenHDataset(
            dataset_roots=[self.leaf],
            local_repo_root=str(self.root),
            split="val",
            fps=4,
            frames_per_clip=4,
            val_fraction=0.4,
            seed=7,
        )

        self.assertEqual(
            [episode.episode_index for episode in first.episodes],
            [episode.episode_index for episode in second.episodes],
        )
        self.assertEqual(len(first), 2)

    def test_missing_declared_validation_split_does_not_overlap_train(self):
        info_path = self.root / self.leaf / "meta" / "info.json"
        info = json.loads(info_path.read_text())
        info["splits"] = {"train": "0:5"}
        write_json(info_path, info)

        with self.assertRaisesRegex(PhysicalAIOpenHError, "zero episodes"):
            PhysicalAIOpenHDataset(
                dataset_roots=[self.leaf],
                local_repo_root=str(self.root),
                split="val",
                fps=4,
                frames_per_clip=4,
                val_fraction=0.4,
            )

    def test_explicit_embodiment_split_holds_out_robot_type(self):
        other_leaf = "Surgical/example/other_task"
        source_meta = self.root / self.leaf / "meta"
        target_meta = self.root / other_leaf / "meta"
        target_meta.mkdir(parents=True, exist_ok=True)
        for name in ("modality.json", "tasks.jsonl", "episodes.jsonl"):
            (target_meta / name).write_text((source_meta / name).read_text())
        info = json.loads((source_meta / "info.json").read_text())
        info["robot_type"] = "otherbot"
        write_json(target_meta / "info.json", info)

        train = PhysicalAIOpenHDataset(
            dataset_roots=[self.leaf, other_leaf],
            local_repo_root=str(self.root),
            split="train",
            fps=4,
            frames_per_clip=4,
            split_strategy="embodiment",
            split_embodiments={"train": ["testbot"], "test": ["otherbot"]},
        )
        test = PhysicalAIOpenHDataset(
            dataset_roots=[self.leaf, other_leaf],
            local_repo_root=str(self.root),
            split="test",
            fps=4,
            frames_per_clip=4,
            split_strategy="embodiment",
            split_embodiments={"train": ["testbot"], "test": ["otherbot"]},
        )

        self.assertEqual({episode.embodiment for episode in train.episodes}, {"testbot"})
        self.assertEqual({episode.embodiment for episode in test.episodes}, {"otherbot"})


if __name__ == "__main__":
    unittest.main()
