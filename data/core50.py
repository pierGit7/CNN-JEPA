from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

from PIL import Image

from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torchvision.datasets.vision import VisionDataset


class CORe50(VisionDataset):
    """`CORe50 <https://vlomonaco.github.io/core50/>`_ dataset (128x128 version), labelled by object category.

    CORe50 has 50 handheld objects from 10 categories (5 objects each), recorded in 11 sessions with different
    backgrounds and lighting, ~300 consecutive video frames per object and session. The standard split by session is
    used: sessions 3, 7 and 10 are the test set, the other 8 the train set.

    Args:
        root (str or ``pathlib.Path``): Directory containing (or to download into) ``core50_128x128/``.
        split (string, optional): ``"train"`` (default) or ``"test"``.
        frame_step (int, optional): Keep one frame every ``frame_step`` frames of each (session, object) sequence.
            Consecutive frames are near-duplicates, so this shrinks the dataset (~165k images in total) with little
            loss of information. Defaults to 1 (all frames).
        download (bool, optional): If ``True``, downloads and extracts the dataset (~5.9 GB zip) into ``root``.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed
            version, e.g. ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    Attributes:
        classes (list): Category names, the target is the index into this list.
    """

    _URL = "http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip"
    _DIRNAME = "core50_128x128"
    _TEST_SESSIONS = (3, 7, 10)
    _TRAIN_SESSIONS = (1, 2, 4, 5, 6, 8, 9, 11)
    _NUM_OBJECTS = 50
    _OBJECTS_PER_CATEGORY = 5
    classes = [
        "plug adapter",
        "mobile phone",
        "scissors",
        "light bulb",
        "can",
        "glasses",
        "ball",
        "marker",
        "cup",
        "remote control",
    ]

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        frame_step: int = 1,
        download: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        self._split = verify_str_arg(split, "split", ["train", "test"])
        self._frame_step = int(frame_step)
        if self._frame_step < 1:
            raise ValueError(f"frame_step must be >= 1, got {frame_step}")
        self._image_root = Path(self.root) / self._DIRNAME

        if download and not self._check_exists():
            download_and_extract_archive(self._URL, str(self.root))

        if not self._check_exists():
            raise RuntimeError(
                f"Dataset not found in {self._image_root}. Use download=True, or download {self._URL} manually and "
                f"extract it into {self.root}, so that {self._image_root}/s1/o1/C_01_01_000.png exists."
            )

        sessions = self._TRAIN_SESSIONS if self._split == "train" else self._TEST_SESSIONS
        self.sessions = list(sessions)
        self._samples = []
        for session in sessions:
            for obj in range(1, self._NUM_OBJECTS + 1):
                frames = sorted((self._image_root / f"s{session}" / f"o{obj}").glob("*.png"))
                label = (obj - 1) // self._OBJECTS_PER_CATEGORY
                self._samples.extend((str(path), label) for path in frames[:: self._frame_step])

        if not self._samples:
            raise RuntimeError(f"No images found in {self._image_root} for sessions {self.sessions}.")

        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

    def _check_exists(self) -> bool:
        return (self._image_root / "s1").is_dir()

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        path, label = self._samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self) -> int:
        return len(self._samples)

    def extra_repr(self) -> str:
        return f"Split: {self._split}, sessions: {self.sessions}, frame_step: {self._frame_step}"
