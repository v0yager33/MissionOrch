"""本地轨迹采集；默认开启，无外部上传。"""
from .core import configure, emit, get_recorder, scope, trace_call


def configure_default():
    from missionorch_lc.core.settings import PROJECT_ROOT, get_settings

    settings = get_settings()
    directory = settings.trajectory_dir.expanduser()
    if not directory.is_absolute():
        directory = PROJECT_ROOT / directory
    return configure("missionorch-lc", directory, settings.trajectory_enabled)


__all__ = ["configure", "emit", "get_recorder", "scope", "trace_call"]
