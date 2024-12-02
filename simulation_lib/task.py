import uuid

type TaskIDType = uuid.UUID

type OptionalTaskIDType = TaskIDType | None


def get_task_id() -> OptionalTaskIDType:
    return uuid.uuid4()
