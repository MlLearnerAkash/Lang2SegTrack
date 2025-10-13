from dataclasses import dataclass, field
from typing import Any, List, Tuple
from typing import Dict, Optional

import numpy


@dataclass
class TrackedObject:
    object_id: int
    label: str
    initial_prompt: Any
    insert_frame_index: int
    confidence: float
    bbox: Tuple[int, int, int, int]
    mask: numpy.ndarray
    last_updated_frame: int





class ObjectInfoManager:
    """
    管理多个被跟踪对象的信息。

    该管理器通过唯一的 object_id 来添加、更新和检索对象信息。
    """

    def __init__(self):
        """
        初始化对象信息管理器。
        """
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self._next_object_id: int = 0

        print("The Object Information Manager is initialized.")

    def add_object(self, initial_prompt: Any, insert_frame_index: int,
                   mask: numpy.ndarray, bbox: Tuple[int, int, int, int] | None = None,
                   label: str | None = None, confidence: float | None = None) -> int:
        object_id = self._next_object_id

        new_object = TrackedObject(
            object_id=object_id,
            label=label,
            initial_prompt=initial_prompt,
            insert_frame_index=insert_frame_index,
            confidence=confidence,
            bbox=bbox,
            mask=mask,
            last_updated_frame=insert_frame_index
        )

        self.tracked_objects[object_id] = new_object
        self._next_object_id += 1

        print(f"Frame {insert_frame_index}: Add new obj -> ID: {object_id}, Label: '{label}'")
        return object_id

    def update_object(self, object_id: int, mask: numpy.ndarray,
                      bbox: Tuple[int, int, int, int] | None = None, confidence: float | None = None, last_updated_frame: int | None = None):
        """更新最后更新帧索引"""
        if object_id not in self.tracked_objects:
            raise KeyError(f"Error: Attempt to update an object that does not exist. ID: {object_id}")

        obj = self.tracked_objects[object_id]
        obj.confidence = confidence
        obj.bbox = bbox
        obj.mask = mask
        obj.last_updated_frame = last_updated_frame

    def get_objects_by_label(self, label: str) -> List[TrackedObject]:
        """根据标签查询对象"""
        return [obj for obj in self.tracked_objects.values() if obj.label == label]

    def cleanup_lost_objects(self, current_frame: int, max_lost_frames: int = 30):
        """清理长时间未更新的对象"""
        lost_ids = []
        for obj_id, obj in self.tracked_objects.items():
            if current_frame - obj.last_updated_frame > max_lost_frames:
                lost_ids.append(obj_id)

        for obj_id in lost_ids:
            self.remove_object(obj_id)

        return lost_ids

    def get_object_info(self, object_id: int) -> Optional[TrackedObject]:
        """
        获取指定ID的对象信息。

        Args:
            object_id (int): 对象的ID。

        Returns:
            Optional[TrackedObject]: 如果找到则返回 TrackedObject 实例，否则返回 None。
        """
        return self.tracked_objects.get(object_id)

    def get_all_objects_info(self) -> Dict[int, TrackedObject]:
        """
        获取当前所有被跟踪对象的信息。

        Returns:
            Dict[int, TrackedObject]: 一个字典，键是对象ID，值是 TrackedObject 实例。
        """
        return self.tracked_objects

    def remove_object(self, object_id: int):
        """
        当一个对象跟踪丢失或结束时，将其从管理器中移除。

        Args:
            object_id (int): 要移除的对象的ID。

        Raises:
            KeyError: 如果提供的 object_id 不存在。
        """
        if object_id not in self.tracked_objects:
            raise KeyError(f"Error: Attempt to remove a non-existent object. ID: {object_id}")

        del self.tracked_objects[object_id]
        print(f"The object has been removed. ID: {object_id}")

    def __str__(self):
        """
        返回管理器当前状态的字符串表示。
        """
        if not self.tracked_objects:
            return "There are currently no objects being tracked."

        info_lines = [f"Manager Status: Tracking {len(self.tracked_objects)} objects."]
        for obj_id, obj in self.tracked_objects.items():
            info_lines.append(f"  - {obj}")
        return "\n".join(info_lines)