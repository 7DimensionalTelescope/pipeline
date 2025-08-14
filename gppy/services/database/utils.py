import uuid


def generate_id() -> str:
    return str(uuid.uuid4())


def generate_frame_id() -> str:
    """Generate a unique frame ID for calibration frame QA data"""
    return str(uuid.uuid4())
