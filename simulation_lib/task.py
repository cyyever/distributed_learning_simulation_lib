import uuid

type TaskIDType = uuid.UUID

type OptionalTaskIDType = TaskIDType | None


def get_task_id() -> TaskIDType:
    return uuid.uuid4()
